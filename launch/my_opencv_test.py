import cv2 as cv
import numpy as np

# 1. เริ่มต้นกล้อง (ลองเปลี่ยนเลขเป็น 0, 2, 4 หากกล้องไม่ติด)
cap = cv.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ปรับขนาดให้เล็กลงเพื่อให้แสดงผลรวมกันได้ง่าย
    frame = cv.resize(frame, (480, 360))
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # 2. กำหนดช่วงสี (ตามที่คุณใช้ในสไลด์และวิดีโอ)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_green = np.array([35, 70, 60])
    upper_green = np.array([85, 255, 255])

    # สร้าง Mask สำหรับแต่ละสี
    mask_red = cv.inRange(hsv, lower_red, upper_red)
    mask_green = cv.inRange(hsv, lower_green, upper_green)
    
    # รวม Mask เข้าด้วยกัน (แดง OR เขียว)
    combined_mask = cv.bitwise_or(mask_red, mask_green)

    # 3. กำจัดจุดรบกวน (Denoising) ด้วย Morphology
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel)

    # 4. ค้นหาขอบวัตถุ (Contours) และวาดเส้นล้อมรอบ
    contours, _ = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 500: # กรองเฉพาะวัตถุที่มีขนาดใหญ่พอ
            x, y, w, h = cv.boundingRect(cnt)
            # วาดสี่เหลี่ยมล้อมรอบวัตถุที่เจอ
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, "Target Detected", (x, y - 10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 5. ตัดภาพแสดงผลสีเฉพาะส่วนที่เลือก
    result = cv.bitwise_and(frame, frame, mask=combined_mask)

    # 6. รวมภาพแสดงผล (Stacking)
    # แปลง Mask 1 แชนแนลเป็น 3 แชนแนลเพื่อรวมกับภาพสี
    mask_3ch = cv.cvtColor(combined_mask, cv.COLOR_GRAY2BGR)
    final_output = np.hstack((frame, mask_3ch, result))

    # แสดงผล
    cv.imshow('OpenCV Ultimate Analysis', final_output)

    # กด 'q' เพื่อออก
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()