import cv2
import numpy as np
import os

# ============================================================
# อ่านรูปภาพ lena (ใช้ path อ้างอิงจากตำแหน่งของไฟล์นี้)
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
lena_path = os.path.join(script_dir, "image", "lena.png")
image = cv2.imread(lena_path)

if image is None:
    # ถ้าหาภาพไม่เจอ สร้างภาพทดสอบขึ้นมา
    image = np.ones((512, 512, 3), dtype="uint8") * 80

h, w = image.shape[:2]
print(f"Width = {w}, Height = {h}")

# ============================================================
# สร้างภาพ copy สำหรับวาด
# ============================================================
img_draw = image.copy()

# ============================================================
# 1. พิมพ์ขนาด w+h ที่กลางภาพ
# ============================================================
size_text = f"w={w} h={h}"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.9
thickness = 2

# คำนวณขนาดข้อความเพื่อวางตรงกลาง
(text_w, text_h), baseline = cv2.getTextSize(size_text, font, font_scale, thickness)
text_x = (w - text_w) // 2
text_y = (h + text_h) // 2

cv2.putText(img_draw, size_text,
            (text_x, text_y),
            font, font_scale, (0, 255, 255), thickness)

name_text = "Sphuwaset"

# ============================================================
# 3. วาด วงกลม สามเหลี่ยม สี่เหลี่ยม ให้มีสี
# ============================================================

# --- วงกลม (circle) สีเขียว มุมขวาบน ---
radius = min(w, h) // 7
circle_cx = w - radius - 15           # ชิดขวา
circle_cy = radius + 15               # ชิดบน
cv2.circle(img_draw, (circle_cx, circle_cy), radius,
           (0, 255, 0), 3)  # สีเขียว (BGR)

# --- สี่เหลี่ยม (rectangle) สีน้ำเงิน มุมล่างซ้าย พร้อมชื่อ ---
(name_w, name_h), _ = cv2.getTextSize(name_text, font, 0.8, 2)
rect_pad = 10
rect_x1 = 10
rect_y1 = h - name_h - rect_pad * 2 - 10
rect_x2 = rect_x1 + name_w + rect_pad * 2
rect_y2 = h - 10
cv2.rectangle(img_draw, (rect_x1, rect_y1), (rect_x2, rect_y2),
              (255, 0, 0), 2)  # สีน้ำเงิน (BGR)
cv2.putText(img_draw, name_text,
            (rect_x1 + rect_pad, rect_y2 - rect_pad),
            font, 0.8, (255, 255, 255), 2)

# --- สามเหลี่ยม (triangle) สีแดง มุมบนซ้าย ---
tri_size = min(w, h) // 5
tri_offset = 15
pt1 = (tri_offset + tri_size // 2, tri_offset)             # ยอดบน
pt2 = (tri_offset,                 tri_offset + tri_size)  # มุมล่างซ้าย
pt3 = (tri_offset + tri_size,      tri_offset + tri_size)  # มุมล่างขวา
triangle_pts = np.array([pt1, pt2, pt3], dtype=np.int32)
cv2.polylines(img_draw, [triangle_pts], True,
              (0, 0, 255), 3)  # สีแดง (BGR)

# ============================================================
# แสดงผลหน้าต่างหลัก (ภาพสี)
# ============================================================
cv2.imshow("OpenCV Image - Color", img_draw)

# ============================================================
# 4. เปิดหน้าต่างใหม่เปลี่ยนเป็น Gray color
# ============================================================
img_gray = cv2.cvtColor(img_draw, cv2.COLOR_BGR2GRAY)
cv2.imshow("OpenCV Image - Gray", img_gray)

# ============================================================
cv2.waitKey(0)
cv2.destroyAllWindows()
