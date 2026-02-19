import cv2
import sys

#imagePath = 'football.jpg'
#imagePath = 'face.jpg'
imagePath = './image/lena.png'
#imagePath = 'fullbody.jpg'

cascPath = "haarcascade_frontalface_default.xml"
#cascPath = "haarcascade_eye.xml"
#cascPath = "haarcascade_fullbody.xml"

i=0

faceCascade = cv2.CascadeClassifier(cascPath)
image = cv2.imread(imagePath)

if image is None:
    print(f"Error: Could not read image '{imagePath}'")
    sys.exit(1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Mtt colour", image)
# Adapted for headless environment: saving to file instead of showing window
cv2.imwrite("debug_color.jpg", image)
print("Saved debug_color.jpg (Mtt colour)")

cv2.imshow("Mtt gray", gray)
# Adapted for headless environment
cv2.imwrite("debug_gray.jpg", gray)
print("Saved debug_gray.jpg (Mtt gray)")

# Re-initializing classifier as per user example (though technically redundant if already done above)
faceCascade = cv2.CascadeClassifier(cascPath)
faces = faceCascade.detectMultiScale(gray)

o = 50
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    i=i+1
    print(i, x, y)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f"X,Y{i} : {x},{y}", (50, o), font, 1, (0, 255, 0), 2)
    o = o + 30


cv2.imshow("Faces found", image)
# Adapted for headless environment
cv2.imwrite("faces_found.jpg", image)
print("Saved faces_found.jpg (Faces found)")

cv2.waitKey(0) 
