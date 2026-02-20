import os
from ultralytics import YOLO
import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "best.pt")

# โหลดโมเดล (โหลดครั้งแรกจะดาวน์โหลดอัตโนมัติ)
model = YOLO(model_path)   # n = nano (เร็วมาก)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Detect
    results = model(frame, device="cpu")

    # วาดกรอบอัตโนมัติ
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

