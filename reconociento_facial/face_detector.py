from asyncore import read
from sre_constants import SUCCESS
from tkinter import Frame
import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0) 

while True:

    succesful_frame_read, Frame = video.read()

    if not succesful_frame_read:

        break
    Frame_grayscale = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(Frame_grayscale)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(Frame, (x, y), (x+w, y+h), (100,200,50), 2)
    cv2.imshow("walter", Frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

print("esta funcionando!")

