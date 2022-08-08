import torch
from PIL import Image
import cv2
import numpy as np
import pandas
import sys
import time
import os
import requests
import json
import urllib.request
from cryptography.fernet import Fernet

from openpyxl import Workbook,load_workbook
#urll="https://192.168.1.130:8080/video"
listem = []
global string3
model = torch.hub.load('ultralytics/yolov5', 'custom', path='enson.pt') #plate detection
model1 = torch.hub.load('ultralytics/yolov5', 'custom', path='best2.pt') #char detection
list = []
#"rtsp://admin:@192.168.1.24:554/mode=real&idc=1&ids=2"
cap = cv2.VideoCapture("yılmazabi.mp4")
prev_frame_time = 0  # for fps
new_frame_time = 0
dec_list=[]
dec_list2=[]
key="_Xk4ia5-y_mo3Sau663HP7Ejpq3jizdNFUYheuRyd00="
def connect(host='https://www.google.com'):
    try:
        urllib.request.urlopen(host) #Python 3.x
        return True
    except:
        return False

while (cap.isOpened()):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = frame
        gray = cv2.resize(gray, (500, 300))
        results = model(gray)

        for _, det in enumerate(results.xyxy[0]):
            # convert from tensor to numpy
            box = det.detach().cpu().numpy()[:5]
            # convert from float to integer
            box = [int(x) for x in box]
            x1, y1, x2, y2, name = box  # crop the license plate image part
            cropped = gray[y1:y2, x1:x2].copy()
            label = "plate"
            # draw a box on original image
            cv2.rectangle(gray, (x1, y1), (x2, y2), (0, 255, 255), 2)

            img2 = cropped.copy()
            img2 = cv2.resize(cropped, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
            height, width, a = img2.shape
            if ((width / height) >= 2.5):
                #print("dikdörtgen plaka")
                results1 = model1(img2)
                results1.pandas().xyxy[0].sort_values('xmin')
                plate_num = str(results1.pandas().xyxy[0].sort_values('xmin').name.values)
                clean_text2 = ""
                for char in plate_num:
                    if char in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                        clean_text2 += char
                plate_num = clean_text2
                plate_num = plate_num.replace(" ", "").replace("\n", "")
                #print(plate_num)

            else:
                #print("kare plaka")
                h, w, channels = img2.shape
                half2 = h // 2
                top = img2[:half2, :]
                bottom = img2[half2:, :]
                #cv2.imshow('Top', top)
                #cv2.imshow('Bottom', bottom)
                resultsa = model1(top)
                resultsb = model1(bottom)

                resultsa.pandas().xyxy[0].sort_values('xmin')
                plate_numa = str(resultsa.pandas().xyxy[0].sort_values('xmin').name.values)
                clean_text2 = ""
                for char in plate_numa:
                    if char in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                        clean_text2 += char
                plate_numa = clean_text2
                plate_numa = plate_numa.replace(" ", "")

                resultsb.pandas().xyxy[0].sort_values('xmin')
                plate_numb = str(resultsb.pandas().xyxy[0].sort_values('xmin').name.values)
                clean_text2 = ""
                for char in plate_numb:
                    if char in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                        clean_text2 += char
                plate_numb = clean_text2
                plate_numb = plate_numb.replace(" ", "")
                plate_num = plate_numa + plate_numb


            font = cv2.FONT_HERSHEY_SIMPLEX
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps = str(fps)
            cv2.putText(gray, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', gray)


            if (len(plate_num) < 9) and (len(plate_num) > 6):
                list.append(plate_num)
                def most_frequent(list):
                    counter = 0
                    num = list[0]
                    for i in list:
                        curr_frequency = list.count(i)
                        if (curr_frequency > counter):
                            counter = curr_frequency
                            num = i
                    return num
                if(len(list)==23):
                    b = most_frequent(list)
                    
                    list2=list[10:]
                    b = str(b)

                    txtlist = []
                    list2 = []
                    #Bağlantı varsa Api de kayıtlı plaka ile algılanan plakayı karşılaştırma ve  Excell dosyasında güncelleme yapma
                    if connect() == True:
                        print("bağlantı var.Apiden Kontrol Sağlanıyor...")

			url = "http://192.168.1.72:27019/Vehicles/GetAll"

			payload = ""
			headers = {
			  'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyaWQiOiIyIiwiZW1haWwiOiJuZWJpeWVidWxhbkBnbWFpbC5jb20iLCJ1c2Vycm9sZWlkIjoiMSIsIm5iZiI6MTY1OTEwNDE2MSwiZXhwIjoxNjc1MDAxNzYxLCJpYXQiOjE2NTkxMDQxNjF9.c86ajJsaUidN_4-BjOGrxPTImea_SE0r1LnMrC0n8Os',
			  'Content-Type': 'application/json'
			}

			response = requests.request("GET", url, headers=headers, data=payload)

			vehicles=response.text




                     
                        response = json.loads(vehicles)
                        a = response['token']
                        #print(a)
                        headers = {"Authorization": "Bearer" + " " + a}

                        xim = requests.get("http://192.168.1.72:27019/Vehicles/GetAll", headers=headers)
                        print(xim)
                        yim = xim.text
                        res = json.loads(yim)
                        # print(res)
                        aim = res['data']
                        api_platelist = []
                        datalist = []
                        for i in aim:
                            plate = i['licencePlate']
                            api_platelist.append(plate)
                            message = plate.encode()
                            ff = Fernet(key)
                            encrypted = ff.encrypt(message)
                            enc_files = open("enc.txt", "a")
                            string = str(encrypted, 'UTF-8')
                            enc_files.write(string)
                            res=bytes(string,'utf-8')
                            dec=ff.decrypt(res)
                            string3 = str(dec, 'UTF-8')
                            print(string3)
                            dec_list.append(string3)
                            for repeat in dec_list:
                                if repeat not in dec_list2:
                                    dec_list2.append(repeat)
                        print("dec_list",dec_list2)


                        if  b in api_platelist:
                            print(b," Plakası Api ye Kayıtlıdır")
                        else:
                            print(b, " Plakası Api de Mevcut Değildir.")


                    else: #İnternet yoksa excel plakası üzerine işlem
                        sys.stderr.write("Bağlantıda Bir Sorun Oluştu! Text Dosyasından Kontrol Sağlanıyor....")
                        sys.stderr.flush()
                        dec_listt=[]
			
                        enc = open("enc.txt", "r").readlines()
                        for i in enc:
                            byte=bytes(i,'utf-8')
                            fff = Fernet(key)

                            dec=fff.decrypt(byte)
                            string3 = str(dec, 'UTF-8')
                            dec_listt.append(string3)
                            for repeat in dec_list:
                                if repeat not in dec_listt:
                                    dec_listt.append(repeat)
                                else :
                                    pass
                            #print("dec_list",dec_listt)
                            """for text in enc:
                            text = text.replace("\n", "")
                            ff = Fernet(key)
                            a_list.append(text)
                            print(a_list)
                            for i in a_list:
                                one = a_list[0]
                                # print(one)
                                res = bytes(one, 'utf-8')
                                dec = ff.decrypt(res)
                                string = str(dec, 'UTF-8')
                                print(string)
                                e = string.split("\n")
                                for i in e:
                                    if i == "":
                                        e.remove(i)"""


                        if b in dec_listt:
                            print("text kayıtlı")
                        else:
                            print(b," Plakası Text Dosyasında Mevcut Değildir!")

                    list = []  # Listeye Atanan Frameler için listeyi tekrar sıfırlar.

                else:
                    continue
            else:
                continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
