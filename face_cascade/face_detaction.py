import cv2

import os

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the XML file
cascPath = os.path.join(current_dir, "haarcascade_frontalface_default.xml")

#cascPath = "haarcascade_eye.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

cap = cv2.VideoCapture(0)


i = 0
while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30,30)
    )

    o = 50
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        i = i + 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"X,Y{i} : {x},{y}", (50, o), font, 1, (0, 255, 0), 2)
        o = o + 30

    cv2.imshow("Face Detection - Webcam", frame)

    #  ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()