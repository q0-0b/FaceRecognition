import cv2
import os
# 각 User들의 얼굴을 인식해서 받아 저장하는 프로그램
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
# 실행이 안 될 시, nameCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascades/사용할 하르파일.xml')
# 절대 경로로 접속 '/Users/username/opencv-python-4.5.3.56/opencv/data/haarcascades'로 .xml 파일을 불러와서 실행
# 또는 실행 코드가 저장된  디렉토리에 'haarcascades'파일(git-hub opencv에서 다운로드 가능) 저장 후 불러와서 실행
# For each person, enter one numeric face id
face_id = input('\n enter user ID end press <return> ==>   ') # 지금 인식하는 사람(User)의 ID 입력
print("\n [HEY!] Initializing face capture. Please look at the camera and wait ..  ")
# 얼굴캡쳐 초기화 안내 카메라를 보고 기다려주십시오 ..  
# Initialize individual sampling face count
count = 0

while(True):
    ret, img = cam.read()
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        # Save the captured image into the datasets folder
        # current_path = os.getcwd()
        # os.mkdir(current_path + "dataset/" + str(input) + "/")
        cv2.imwrite("dataset/" + str(face_id) + "/" + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break
# Do a bit of cleanup
print("\n We got 30 of your faces for Training! Thanks bro") # little bit a joke 약간의 조크
cam.release()
cv2.destroyAllWindows()
