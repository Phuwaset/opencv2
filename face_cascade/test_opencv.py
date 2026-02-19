import cv2
import numpy as np
import math    

# ============================================================
# อ่านรูปภาพ (ใช้ภาพ lena หรือภาพอื่น)
# ============================================================
image = cv2.imread("./image/lena.png")
if image is None:
    # ถ้าไม่มีรูป ให้สร้างรูปทดสอบขึ้นมา
    image = np.zeros((512, 512, 3), dtype="uint8")
    image[100:400, 100:400] = [180, 130, 80]
    cv2.putText(image, "Test Image", (120, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# ============================================================
# 1. Read / Show Image
# ============================================================
cv2.imshow("1 - Read Image (Original)", image)

# ============================================================
# 2. Gray Scale
# ============================================================
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("2 - Gray Scale", gray)

# ============================================================
# 3. Write Image (บันทึกไฟล์)
# ============================================================
cv2.imwrite("output_gray.jpg", gray)
print("[3] Write Image: บันทึก output_gray.jpg สำเร็จ")

# ============================================================
# 4. Get and Set Pixel
# ============================================================
img_pixel = image.copy()
[b, g, r] = img_pixel[0, 0]
print(f"[4] Get Pixel [0,0]: B={b}, G={g}, R={r}")

img_pixel[0, 0] = [0, 0, 255]  # เปลี่ยนเป็นสีแดง
[b2, g2, r2] = img_pixel[0, 0]
print(f"[4] Set Pixel [0,0] = [0,0,255] → B={b2}, G={g2}, R={r2}")

# วาดสีแดงก้อนใหญ่ให้เห็นชัด
img_pixel[50:100, 50:100] = [0, 0, 255]
cv2.imshow("4 - Get and Set Pixel (red block at top-left)", img_pixel)

# ============================================================
# 5. Crop Image
# ============================================================
h, w = image.shape[:2]
crop = image[int(h*0.05):int(h*0.6), int(w*0.25):int(w*0.75)]
cv2.imshow("5 - Crop Image", crop)

# ============================================================
# 6. Draw Line
# ============================================================
img_line = image.copy()
cv2.line(img_line, (0, 0), (w-1, h-1), (0, 255, 0), 2)       # เส้นทแยง สีเขียว
cv2.line(img_line, (0, h-1), (w-1, 0), (0, 0, 255), 2)        # เส้นทแยง สีแดง
cv2.line(img_line, (0, h//2), (w-1, h//2), (255, 255, 0), 3)  # เส้นกลาง สีเหลือง
cv2.imshow("6 - Draw Line", img_line)

# ============================================================
# 7. Draw Rectangle
# ============================================================
img_rect = image.copy()
cv2.rectangle(img_rect, (int(w*0.25), int(h*0.05)), (int(w*0.75), int(h*0.55)), (0, 255, 0), 2)
cv2.imshow("7 - Draw Rectangle", img_rect)

# ============================================================
# 8. Put Text
# ============================================================
img_text = image.copy()
cv2.rectangle(img_text, (int(w*0.25), int(h*0.05)), (int(w*0.75), int(h*0.55)), (0, 255, 0), 2)
cv2.putText(img_text, "face", (int(w*0.25), int(h*0.05) - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow("8 - Put Text", img_text)

# ============================================================
# 9. Draw Circle
# ============================================================
img_circle = image.copy()
cv2.circle(img_circle, (w//2, h//2), min(w, h)//4, (0, 255, 0), 2)
cv2.imshow("9 - Draw Circle", img_circle)

# ============================================================
# 10. Draw Polylines
# ============================================================
img_poly = image.copy()
points = np.array([(int(w*0.35), int(h*0.05)),
                   (int(w*0.65), int(h*0.05)),
                   (int(w*0.75), int(h*0.55)),
                   (int(w*0.10), int(h*0.55))], dtype=np.int32)
cv2.polylines(img_poly, [points], True, (0, 255, 0), 2)
cv2.imshow("10 - Draw Polylines", img_poly)

# ============================================================
# 11. Shift
# ============================================================
def shift_image(img, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

shifted_down  = shift_image(image, 0, 100)
shifted_right = shift_image(image, 100, 0)
cv2.imshow("11 - Shift Down", shifted_down)
cv2.imshow("11 - Shift Right", shifted_right)

# ============================================================
# 12. Rotate
# ============================================================
center = (w // 2, h // 2)
M45  = cv2.getRotationMatrix2D(center, 45, 1.0)
M180 = cv2.getRotationMatrix2D(center, 180, 1.0)
rotated45  = cv2.warpAffine(image, M45,  (w, h))
rotated180 = cv2.warpAffine(image, M180, (w, h))
cv2.imshow("12 - Rotate 45 degrees", rotated45)
cv2.imshow("12 - Rotate 180 degrees", rotated180)

# ============================================================
# 13. Resize
# ============================================================
resized_small = cv2.resize(image, (w//2, h//2), interpolation=cv2.INTER_AREA)
resized_large = cv2.resize(image, (w*2, h*2), interpolation=cv2.INTER_LINEAR)
cv2.imshow("13 - Resize Small (half)", resized_small)
cv2.imshow("13 - Resize Large (double)", resized_large)

# ============================================================
# 14. Image Arithmetic (Bright / Dark)
# ============================================================
ones         = np.ones(image.shape, dtype="uint8")
img_bright   = cv2.add(image, ones * 80)
img_dark     = cv2.subtract(image, ones * 60)
cv2.imshow("14 - Image Bright", img_bright)
cv2.imshow("14 - Image Dark",   img_dark)

# ============================================================
# 15. Image Bitwise Operation
# ============================================================
rect_mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(rect_mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
circ_mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.circle(circ_mask, (w//2, h//2), min(w, h)//3, 255, -1)

bw_and = cv2.bitwise_and(rect_mask, circ_mask)
bw_or  = cv2.bitwise_or (rect_mask, circ_mask)
bw_xor = cv2.bitwise_xor(rect_mask, circ_mask)
bw_not = cv2.bitwise_not(rect_mask)
cv2.imshow("15 - Bitwise AND",  bw_and)
cv2.imshow("15 - Bitwise OR",   bw_or)
cv2.imshow("15 - Bitwise XOR",  bw_xor)
cv2.imshow("15 - Bitwise NOT",  bw_not)

# ============================================================
# 16. Mask
# ============================================================
mask_circle = np.zeros(image.shape[:2], dtype="uint8")
cv2.circle(mask_circle, (w//2, h//2), min(w, h)//3, 255, -1)
masked = cv2.bitwise_and(image, image, mask=mask_circle)
cv2.imshow("16 - Mask (circle)", masked)

# ============================================================
# 17. Split and Merge
# ============================================================
B, G, R = cv2.split(image)
zeros   = np.zeros((image.shape[0], image.shape[1]), dtype="uint8")
img_R   = cv2.merge([zeros, zeros, R])
img_G   = cv2.merge([zeros, G, zeros])
img_B   = cv2.merge([B, zeros, zeros])
merged  = cv2.merge([B, G, R])
cv2.imshow("17 - Channel Red",   img_R)
cv2.imshow("17 - Channel Green", img_G)
cv2.imshow("17 - Channel Blue",  img_B)
cv2.imshow("17 - Merged",        merged)

# ============================================================
# 18. Color Space
# ============================================================
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
img_lab  = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("18 - Color Space GRAY", img_gray)
cv2.imshow("18 - Color Space HSV",  img_hsv)
cv2.imshow("18 - Color Space LAB",  img_lab)

# ============================================================
# 19. Histogram Equalization
# ============================================================
gray_eq = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
eq      = cv2.equalizeHist(gray_eq)
cv2.imshow("19 - Original Gray",          gray_eq)
cv2.imshow("19 - Histogram Equalization", eq)

# ============================================================
# 20. Blurring - Averaging
# ============================================================
blur_avg3 = cv2.blur(image, (3, 3))
blur_avg5 = cv2.blur(image, (5, 5))
blur_avg7 = cv2.blur(image, (7, 7))
cv2.imshow("20 - Blur Average 3x3", blur_avg3)
cv2.imshow("20 - Blur Average 5x5", blur_avg5)
cv2.imshow("20 - Blur Average 7x7", blur_avg7)

# ============================================================
# 21. Blurring - Gaussian
# ============================================================
blur_gau3 = cv2.GaussianBlur(image, (3, 3), 0)
blur_gau5 = cv2.GaussianBlur(image, (5, 5), 0)
blur_gau7 = cv2.GaussianBlur(image, (7, 7), 0)
cv2.imshow("21 - Gaussian Blur 3x3", blur_gau3)
cv2.imshow("21 - Gaussian Blur 5x5", blur_gau5)
cv2.imshow("21 - Gaussian Blur 7x7", blur_gau7)

# ============================================================
# 22. Thresholding (Black and White)
# ============================================================
gray_th   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred   = cv2.GaussianBlur(gray_th, (5, 5), 0)
(_, thresh_bin) = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
(_, thresh_inv) = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("22 - Threshold Binary",         thresh_bin)
cv2.imshow("22 - Threshold Binary Inverse", thresh_inv)

# ============================================================
# 23. Remove Background by Thresholding
# ============================================================
masked_bg = cv2.bitwise_and(image, image, mask=thresh_inv)
cv2.imshow("23 - Remove Background (mask with thresh_inv)", masked_bg)

# ============================================================
# 24. Image Gradient Detection (Laplacian + Sobel)
# ============================================================
gray_grad = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
lap       = cv2.Laplacian(gray_grad, cv2.CV_64F)
lap       = np.uint8(np.absolute(lap))

sobelX = cv2.Sobel(gray_grad, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(gray_grad, cv2.CV_64F, 0, 1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobelCombined = cv2.bitwise_or(sobelX, sobelY)

cv2.imshow("24 - Laplacian",       lap)
cv2.imshow("24 - Sobel X",         sobelX)
cv2.imshow("24 - Sobel Y",         sobelY)
cv2.imshow("24 - Sobel Combined",  sobelCombined)

# ============================================================
# 25. Canny Edge Detection
# ============================================================
gray_canny  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur_canny  = cv2.GaussianBlur(gray_canny, (5, 5), 0)
canny       = cv2.Canny(blur_canny, 50, 200)
cv2.imshow("25 - Canny Edge Detection", canny)

# ============================================================
# 26. Image Contour
# ============================================================
gray_cnt  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur_cnt  = cv2.GaussianBlur(gray_cnt, (5, 5), 0)
canny_cnt = cv2.Canny(blur_cnt, 50, 200)

(cnts, _) = cv2.findContours(canny_cnt.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"[26] จำนวน Contour ที่พบ: {len(cnts)}")

img_contour = image.copy()
cv2.drawContours(img_contour, cnts, -1, (0, 255, 0), 2)
cv2.imshow("26 - Image Contour", img_contour)

# ============================================================
# 27. Combined Drawing (Mechatronic Style)
# ============================================================
img_mecha = image.copy()
h27, w27 = img_mecha.shape[:2]

# --- เส้นสีแดง 2 เส้น ทแยงมุมเป็น X ---
cv2.line(img_mecha, (0, 0),    (w27, h27), (0, 0, 255), 3)   # บนซ้าย → ล่างขวา
cv2.line(img_mecha, (w27, 0),  (0, h27),   (0, 0, 255), 3)   # บนขวา  → ล่างซ้าย

# --- เส้นสีเขียว: สามเหลี่ยมมุมล่างขวา ---
# กำหนดมุมก่อน
angle1 = 60    # มุม A (ล่างซ้าย)
angle2 = 60    # มุม B (ล่างขวา)
angle3 = 60    # มุม C (ยอด)
print(f"มุมรวม = {angle1 + angle2 + angle3}")  # ต้องได้ 180

# กำหนดความยาวด้าน
side = 200

# คำนวณจุด 3 มุม
pt_A = (w27 - side, h27)      # มุม A (ล่างซ้าย)
pt_B = (w27,        h27)      # มุม B (ล่างขวา)

# คำนวณความสูงและตำแหน่งยอด จากมุมที่กำหนด
height   = int(side * math.tan(math.radians(angle1)) * math.tan(math.radians(angle2))
               / (math.tan(math.radians(angle1)) + math.tan(math.radians(angle2))))
offset_x = int(height / math.tan(math.radians(angle1)))
pt_C     = (pt_A[0] + offset_x, h27 - height)   # มุม C (ยอด)

print(f"pt_A = {pt_A}")
print(f"pt_B = {pt_B}")
print(f"pt_C = {pt_C}")

triangle = np.array([pt_A, pt_B, pt_C], dtype=np.int32)
cv2.polylines(img_mecha, [triangle], True, (0, 255, 0), 3)

# --- วงกลมสีเหลือง มุมล่างซ้าย ---
radius = min(w27, h27) // 7
cv2.circle(img_mecha,
           (radius + 10, h27 - radius - 10),
           radius, (0, 255, 255), -1)   # BGR → (0,255,255) = สีเหลือง

# --- ข้อความ "Mechatronic" สีเขียว มุมบนขวา ---
cv2.putText(img_mecha, "Mechatronic",
            (w27 - 260, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

cv2.imshow("27 - Mechatronic Style Drawing", img_mecha)

# ============================================================
cv2.waitKey(0)
cv2.destroyAllWindows()