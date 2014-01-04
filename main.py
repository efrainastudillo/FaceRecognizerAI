#print('Hello World')

#import cv2

import cv2
import sys
import numpy as np
import math

faces_cascade_filename = "data/haarcascade_frontalface_alt.xml"
eyes_cascade_filename = "data/haarcascade_eye_tree_eyeglasses.xml"
mouth_cascade_filename = "data/haarcascade_smile.xml"
nose_cascade_filename = "data/haarcascade_mcs_nose.xml"

face_cascade = cv2.CascadeClassifier(faces_cascade_filename)
eye_cascade = cv2.CascadeClassifier(eyes_cascade_filename)
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_filename)
nose_cascade = cv2.CascadeClassifier(nose_cascade_filename)


def with_videocam():

    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        return

    while True:
        flag = cv2.waitKey(1) % 0x100
        if flag == 27:
            break

        retval, img = capture.read()
        if retval:
            detect_faces(image=img)

    capture.release()


def detect_faces(image = None):
    
    #if not image:
        #image = cv2.imread("images/foto1.jpg",cv2.CV_LOAD_IMAGE_COLOR)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.3, minNeighbors=5,)
    print len(faces)

    roi_color = None
    t_eyes = []

    for x,y,w,h in faces:
        print x, y, w, h
        cv2.rectangle(image, (x, y), (w+x, h+y), (255, 255, 0), 2)

        roi_gray = image_gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 2, minSize=(38, 38), maxSize=(40, 40))
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.15, 1, minSize=(82, 40), maxSize=(88, 42))
        nose = nose_cascade.detectMultiScale(roi_gray, 1.15, 1, minSize=(38, 38), maxSize=(60, 50))

        for x1, y1, w1, h1 in nose:
            cv2.rectangle(roi_color, (x1, y1), (x1+w1, y1+h1), (0, 5, 255), 2)

        for x1, y1, w1, h1 in mouth:
            cv2.rectangle(roi_color, (x1, y1), (x1+w1, y1+h1), (26, 50, 0), 2)

        for x1, y1, w1, h1 in eyes:
            print x1, y1, w1, h1
            cv2.rectangle(roi_color, (x1, y1), (x1+w1, y1+h1), (255, 5, 0), 2)
            cv2.circle(roi_color, (x1 + (w1/2), y1 + (h1/2)), 1, (0, 250, 0), 2)

    cv2.namedWindow("Image")
    cv2.imshow("Image", image)

def distance(eyes):
    t1 = eyes[0]
    t2 = eyes[1]

    #return math.sqrt((t2 * t2) + (t1 * t1))
    pass
if __name__ == "__main__":
    print "Hello World"

    with_videocam()
    cv2.destroyAllWindows()