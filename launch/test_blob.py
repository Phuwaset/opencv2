import cv2
import numpy as np

img = cv2.imread('/home/sphuwaset_ros/cv2/opencv2/launch/rose.jpg')
#cv2.imshow('Original Image', img)
h,w = img.shape[:2]
print("Width:", w, "Height:", h)

resized= cv2.resize(img, (w//3, h//3))
h,w = resized.shape[:2]
print("Width:", w, "Height:", h)
cv2.imshow('Resized Image', resized)
cv2.waitKey(0)

hsv_img = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV', hsv_img)
cv2.waitKey(0)


# lower range of red color in HSV
lower_range = (0, 120, 70)
# upper range of red color in HSV
upper_range = (13, 255, 255)
mask = cv2.inRange(hsv_img, lower_range, upper_range)
cv2.imshow('maskdd',mask)
cv2.waitKey(0)

color_image3 = cv2.bitwise_and(resized, resized, mask=mask)
cv2.imshow('mask', color_image3)
cv2.waitKey(0)


color_image = cv2.bitwise_and(img, img, mask=mask)

#125,95,95
lower_range = (100, 80, 80)
upper_range = (150, 120, 120)
mask2 = cv2.inRange(hsv_img, lower_range, upper_range)
color_image2 = cv2.bitwise_and(resized, resized, mask=mask2)



# Display the color of the image
cv2.imshow('Coloured Image', color_image2)
cv2.waitKey(0) 

cv2.imshow('mask',mask2)
cv2.waitKey(0)
cv2.destroyAllWindows()