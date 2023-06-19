import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_trainner.yml')
detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml');
font = cv2.FONT_HERSHEY_SIMPLEX # opencv에서 지원하는 font
# cascadePath = "haarcascades/haarcascade_frontalface_default.xml"
# if don't run, remove tag and edit cv2.CascadeClassifier(cascadePath)
#iniciate id counter
face_id = 0

names = ['dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5']
# names related to ids: example ==> cucumber: id=1,  etc
# 이런식으로 사용자의 이름을 사용자 수만큼 추가해준다.
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
if cam.isOpened() == False : 
    exit()
# Check apear Camera
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(4)
minH = 0.1*cam.get(3)

print("Have finished recognition, press 'ESC' to end it. ")
# 인식을 끝냈으면 "ESC"를 눌러 종료하라고 알림
while True:
    ret, img = cam.read() # 현재 이미지 가져오기, get it current image
    # img = cv2.flip(img, -1) # 상하좌우 반전시킬 때 사용.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 흑백으로 변환
    
    faces = detector.detectMultiScale( 
        gray,
        scaleFactor = 1.5,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       ) # 얼굴 인식

    for(x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w] # importing only the face
        face_id, confidence = recognizer.predict(roi_gray)
        #print(id, confidence)
        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 55) :
            face_id = names[face_id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            face_id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        # 얼굴 인식과 인식한 유저의 정보(이름) 일치하는 정도 출력
        cv2.putText(img, str(face_id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow('camera',img) # show realtime camera
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
cam.release()
cv2.destroyAllWindows()    
