import cv2
import numpy as np

color_ranges = {
    "Green":  {"min": (35, 50, 50),   "max": (85, 255, 255),  "color": (0, 255, 0)},
    "Yellow": {"min": (20, 100, 100), "max": (30, 255, 255),  "color": (0, 255, 255)},
    "Red":    {"min": (0, 100, 100),  "max": (10, 255, 255),  "color": (0, 0, 255)},
    "Blue":   {"min": (100, 150, 0),  "max": (140, 255, 255), "color": (255, 0, 0)}
}

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 500
params.maxArea = 50000
detector = cv2.SimpleBlobDetector_create(params)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # วนลูปเช็คทีละสีตามเงื่อนไขข้อ 2
    for color_name, range_val in color_ranges.items():
        mask = cv2.inRange(hsv, range_val["min"], range_val["max"])
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # กลับสี Mask เพราะ Detector หาจุดมืดบนพื้นสว่าง
        reversemask = 255 - mask
        keypoints = detector.detect(reversemask)

        # ถ้าเจอสี ให้วาดวงกลมและแสดงข้อความตามเงื่อนไขข้อ 3 และ 4
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size)
            
            # วาดวงกลม (ใช้สีเดียวกับที่ตรวจจับได้)
            cv2.circle(frame, (x, y), int(size/2), range_val["color"], 3)
            
            # แสดงชื่อสีและพิกัด
            text = f"Found: {color_name} ({x},{y})"
            cv2.putText(frame, text, (x - 50, y - int(size/2) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, range_val["color"], 2)

    cv2.imshow("Multi-Color Detection", frame)
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()
