from ultralytics import YOLO
import cv2
import time

# โหลดโมเดล yolov8n.pt (COCO 80 classes - detect หมาและแมวได้)
model = YOLO("/home/sphuwaset_ros/cv2/opencv2/face_cascade/yolov8n.pt")
model.to("cpu")  # GTX 1050 ไม่รองรับ PyTorch เวอร์ชันนี้ ใช้ CPU แทน



cap = cv2.VideoCapture(2)

total_dog_count = 0
total_cat_count = 0
last_count_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect ทุกเฟรม (เพื่อแสดงกรอบตลอด)
    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    current_time = time.time()

    # ===== นับทุก 5 วินาที =====
    if current_time - last_count_time >= 5:

        dog_count = 0
        cat_count = 0
        boxes = results[0].boxes

        if boxes is not None:
            for cls in boxes.cls:
                cls_id = int(cls)
                if cls_id == 16:
                    dog_count += 1
                elif cls_id == 15:
                    cat_count += 1

        total_dog_count += dog_count
        total_cat_count += cat_count
        last_count_time = current_time

        print(f"[นับ] หมา: {dog_count}  แมว: {cat_count}")
        print(f"[รวม] หมา: {total_dog_count}  แมว: {total_cat_count}")

    # แสดงผลบนหน้าจอ
    cv2.putText(annotated_frame, f"Dogs:  {total_dog_count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
    cv2.putText(annotated_frame, f"Cats:  {total_cat_count}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 150, 0), 2)

    cv2.imshow("YOLO - Dog & Cat Detection", annotated_frame)

    if cv2.waitKey(1) == 27:  # กด ESC เพื่อออก
        break

cap.release()
cv2.destroyAllWindows()