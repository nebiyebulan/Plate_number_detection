import torch
import cv2
import  matplotlib.pyplot as plt
import pytesseract
import os
from datetime import datetime
import numpy as np
import time
from Db_Handler import Db_Handler
import pprint
from PlateInfo import PlateInfo
import pymongo
from pymongo import MongoClient


model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # or yolov5m, yolov5x, custom

def draw(boxes,plates,frame):

    for i in range(len(boxes)):
        box=boxes[i]
        text=plates[i]
        box = [int(x) for x in box]
        x1, y1, x2, y2 = box
        color = (255, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (x1, y1 - 20), font,
                    1, color, 2, cv2.LINE_AA)
    return frame

if __name__ == '__main__':

    cap = cv2.VideoCapture('1.mp4')     # Dosyadan video okumak için
    prev_frame_time = 0                   # son kareyi işlediğimiz zamanı kaydetmek için kullanılır
    new_frame_time = 0                    # geçerli kareyi işlediğimiz zamanı kaydetmek için kullanılır

    count = 0
    while(cap.isOpened()):

        ret, frame = cap.read()

        if not ret:                             # video bittiyse veya Video Girişi yoksa
            break

        # fps = 1/(new_frame_time-prev_frame_time)# fps, belirli bir zaman aralığında işlenen kare sayısı olacaktır. çünkü çoğu zaman 0.001 saniyelik bir hata olacak
        # prev_frame_time = new_frame_time
        # fps = int(fps)   						# fps'yi tam sayıya dönüştürme
        # fps = str(fps)     						# kare üzerinde görüntüleyebilmemiz için fps'yi dizeye dönüştürme
        # cv2.putText(frame, fps, (7, 35), font, 1, (100, 255, 0), 3, cv2.LINE_AA)          # putText işlevini kullanarak
        #model.conf = 0.60

        if count%30==0:
            results = model(frame)
            roi_list = []
            boxes = []
            plates = []
        # Loop over all the bounding boxes
            for _, det in enumerate(results.xyxy[0]):
                # convert from tensor to numpy
                box=det.detach().cpu().numpy()[:4]
                # convert from float to integer
                box=[int(x) for x in box]
                x1,y1,x2,y2=box                # crop the license plate image part
                cropped = frame[y1:y2,x1:x2].copy()
                label = "plate"
                color=(0,255,255)
                # draw a box on original image
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
                
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]

                plate_num=""
                plate_num0=""
                plate_num1=""
                plate_num2=""


                img2 = cropped.copy()
                kernell = np.ones((5, 5), np.uint8)
                img2 = cv2.resize(cropped, (0, 0),fx=5,fy=5,interpolation = cv2.INTER_CUBIC)
                gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # togray
                dst = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)  # gaussianblur
                ret, thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Threshold
                # img_dilation2 = cv2.erode(thresh2, kernell, iterations=1)  # Dilate
                # img_dilation = cv2.dilate(thresh, None, iterations=1)  # Dilate
                img_dilation=thresh

                try:
                    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                except:
                    ret_img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
                for cnt in sorted_contours:
                    x,y,w,h = cv2.boundingRect(cnt)
                    height, width = gray.shape

                    if height / float(h) > 5:
                        continue
                    ratio = h / float(w)  #oran
                    crop_area = h * w
                    total_area=height*width

                    # Too small
                    if crop_area<100:
                        continue
                    if crop_area/total_area >0.20 or crop_area/total_area<0.01:
                        continue
                    # if width is not more than 25 pixels skip
                    if width / float(w) > 15: continue
                    pad=5
                    try:
                        if ((width / height) >= 2.5):
                            roi = thresh[y-pad:y+h+pad, x-pad:x+w+pad]
                            if roi is None:
                                continue
                            roi = cv2.bitwise_not(roi)

                            roi = cv2.medianBlur(roi, 3)
                            roi = cv2.bitwise_not(roi)
                            roi=cv2.dilate(roi,None,iterations=1)
                            roi_list.append(roi)
                            text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
                            plate_num0 += text
                            plate_num=plate_num0
                            plate_num=plate_num0.replace(" ", "").replace("\n","")


                        else:
                            if ((y >= 0) & (y <= height/2)):
                                roi = thresh[y - pad:y + h + pad, x - pad:x + w + pad]
                                roi = cv2.medianBlur(roi, 3)
                                roi_list.append(roi)
                                text1 = pytesseract.image_to_string(roi,
                                                                    config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
                                plate_num1 += text1
                                
                            elif (y>=height/2):
                                roi = thresh[y - pad:y + h + pad, x - pad:x + w + pad]
                                roi = cv2.bitwise_not(roi)
                                roi = cv2.medianBlur(roi, 3)
                                roi = cv2.bitwise_not(roi)
                                roi = cv2.dilate(roi, None, iterations=1)
                                roi_list.append(roi)
                                text2 = pytesseract.image_to_string(roi,
                                                                    config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
                                plate_num2 += text2
                                plate_num = plate_num1 + plate_num2
                            #print(plate_num)
                        roi_list.append(roi)
                    except:
                        pass
                clean_text2 = ""
                for char in plate_num:
                    if char in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ":
                        clean_text2 += char
                plate_num = clean_text2
                plates.append(plate_num)
                boxes.append(box)
                
                if len(plate_num)>=7  :
                    db=Db_Handler()
                    p=PlateInfo()
                    p.plate_num=plate_num
                    db.insert(p)
                    print(plate_num)
                    

                else:
                    pass

        #time.sleep(2.0)
        count=count+1
        drawn_frame=draw(boxes,plates,frame)

        cv2.imshow("a",drawn_frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
                break
       


    db.plate_list(p)
    cap.release()
    cv2.destroyAllWindows()