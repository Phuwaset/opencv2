import cv2
import os

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the XML file
cascPath = os.path.join(current_dir, "haarcascade_frontalface_default.xml")

faceCascade = cv2.CascadeClassifier(cascPath)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Count the number of faces detected
    face_count = len(faces)

    # Draw a rectangle around the faces
    i = 0
    for (x, y, w, h) in faces:
        i += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Put the number of the face (1, 2, 3...) above the rectangle
        cv2.putText(frame, f"Face #{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the total count of faces on the screen
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Faces detected: {face_count}", (20, 50), font, 1.2, (0, 255, 255), 3)

    cv2.imshow("Face Count Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
