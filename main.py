print('Hello World')
import cv2
import sys
import numpy as np

def with_videocam():
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        return;

    while True:
        flag = cv2.waitKey(1) % 0x100
        if flag == 27:
            break

        cv2.namedWindow("Camera Color",cv2.CV_WINDOW_AUTOSIZE)
        cv2.namedWindow("Camera Gray",cv2.CV_WINDOW_AUTOSIZE)
        ret,img = capture.read()
        image_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #faces = face_cascade.detectMultiScale(image_gray,1.3,5)
        print len(faces)

        
        cv2.imshow("Camera Color",img)
        cv2.imshow("Camera Gray",image_gray)

def main():
    face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt.xml")
    eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

    image = cv2.imread("images/foto1.jpg",cv2.CV_LOAD_IMAGE_COLOR)
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(image_gray,scaleFactor=1.3, minNeighbors=5,)

    print len(faces)
    roi_color = None

    for x,y,w,h in faces:
        #roi = image_gray[y:h, x:w]
        print x,y,w,h
        cv2.rectangle(image,(x,y),(w+x,h+y),(255,255,0),2)
        roi_gray = image_gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

    cv2.imshow("Image",image)
    cv2.imshow("Image Gray", roi_color)
    
    cv2.waitKey()
        

if __name__ == "__main__":
    print "Hello World"
    main()
    cv2.destroyAllWindows()