import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')

video = cv2.VideoCapture("28003-1631171950.webp") 

while True:

    succesful_frame_read, Frame = video.read()

    if not succesful_frame_read:

        break
    Frame_grayscale = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(Frame_grayscale)
    
    smiles = smile_detector.detectMultiScale(Frame_grayscale, scaleFactor = 1.7, minNeighbors = 20)

    eyes = eye_detector.detectMultiScale(Frame_grayscale)

    for (x, y, w, h) in faces:
        cv2.rectangle(Frame, (x, y), (x+w, y+h), (100,200,50), 2)

    for (x, y, w, h) in eyes:
        cv2.rectangle(Frame, (x, y), (x+w, y+h), (0,0,255), 2)

    for (x, y, w, h) in smiles:
        cv2.rectangle(Frame, (x,y), (x+w, y+h), (255,255,255), 2 )
    cv2.imshow("Rosalia", Frame)

    if cv2.waitKey(0) & 0xFF == ord("1"):
        break

video.release()
cv2.destroyAllWindows()

print("esta funcionando!")
