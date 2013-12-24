print('Hello World')
import cv2
import sys
import numpy as np
import math


def with_videocam():
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        return;

    while True:
        flag = cv2.waitKey(1) % 0x100
        if flag == 27:
            break

        cv2.namedWindow("Camera Color",cv2.CV_WINDOW_AUTOSIZE)
        ret,img = capture.read()
        image_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #faces = face_cascade.detectMultiScale(image_gray,1.3,5)
        print len(faces)

        cv2.imshow("Camera Color",img)
        cv2.imshow("Camera Gray",image_gray)

def main():
    face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt.xml")
    eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye_tree_eyeglasses.xml')

    image = cv2.imread("images/foto1.jpg",cv2.CV_LOAD_IMAGE_COLOR)
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(image_gray,scaleFactor=1.3, minNeighbors=5,)

    print len(faces)
    roi_color = None
    t_eyes = []
    for x,y,w,h in faces:
        #roi = image_gray[y:h, x:w]
        print x,y,w,h
        cv2.rectangle(image,(x,y),(w+x,h+y),(255,255,0),2)
        roi_gray = image_gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,2,minSize=(38,38),maxSize=(40,40))
        p1 = []
        for x1,y1,w1,h1 in eyes:
            print x1,y1,w1,h1
            cv2.rectangle(roi_color,(x1,y1),(x1+w1,y1+h1),(255,5,0),2)
            p1.append(x1  + (w1/2))
            p1.append(y1  + (h1/2))
            cv2.circle(roi_color,(x1 + (w1/2),y1 + (h1/2)),1,(0,250,0),2)
        t_eyes.append(p1)

    print t_eyes
    cv2.imshow("Image",image)
    cv2.waitKey()
        

def distance(eyes):
    t1 = eyes[0]
    t2 = eyes[1]

    #return math.sqrt((t2 * t2) + (t1 * t1))
    pass
if __name__ == "__main__":
    print "Hello World"
    main()
    cv2.destroyAllWindows()