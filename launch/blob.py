import cv2
import numpy as np;
import time
## Right
font = cv2.FONT_HERSHEY_SIMPLEX
# H(0-179), S(0-255), V(0-255)
#hsvMin = (20,120,120)
#hsvMax = (49,255,255)

#hsvMin = (110,50,50)
#hsvMax = (130,255,255)
#hsvMin = (10,150,150)    #real Yellow HSV :  [[[ 16 168 204]]], BRG Format:  [204 142 70] 
#hsvMax = (20,255,255)

##hsvMin = (20,50,10)    #real Yellow HSV :  [[[ 15 180 185]]], BRG Format:  [204 142 70] 
##hsvMax = (100,255,250)

hsvMin = (10,100,10)    #real Yellow HSV :  [[[ 15 180 185]]], BRG Format:  [204 142 70] 
hsvMax = (80,255,250)
#hsvMin = (10,150,150)    #real Yellow HSV :  [[[ 16 168 204]]], BRG Format:  [204 142 70] 
#hsvMax = (20,190,190)

params = cv2.SimpleBlobDetector_Params()
 
params.minThreshold = 0;
params.maxThreshold = 100;
 
params.filterByArea = True
params.minArea = 1000
params.maxArea = 20000
 
 
 
params.filterByConvexity = False
params.minConvexity = 0.5

params.filterByInertia = False
params.minInertiaRatio = 0.5

cap = cv2.VideoCapture(1)
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

duty = 1500

def main():
        while True:
                _, frame = cap.read()
##                print("Shape of the image", frame.shape)      480*640
                #crop = frame[1:320, 1:360]
                #frame = crop
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                mask = cv2.inRange(hsv, hsvMin, hsvMax)
                
                mask = cv2.erode(mask, None, iterations=3)
                mask = cv2.dilate(mask, None, iterations=3)
                
                detector = cv2.SimpleBlobDetector_create(params)
                
                reversemask = 255-mask
                
                keypoints = detector.detect(reversemask)

                blobCount = len(keypoints)
                
                text = "Count7=" + str(blobCount) 
                cv2.putText(frame, text, (5,25), font, 1, (0, 255, 0), 2)
                

                if blobCount > 0:
                        blob_x = keypoints[0].pt[0]
                        text2 = "X=" + "{:.2f}".format(blob_x )
                        cv2.putText(frame, text2, (5,50), font, 1, (0, 255, 0), 2)
                
                        blob_y = keypoints[0].pt[1]
                        text3 = "Y=" + "{:.2f}".format(blob_y)
                        cv2.putText(frame, text3, (5,75), font, 1, (0, 255, 0), 2)        
                
                        blob_size = keypoints[0].size
                        text4 = "S=" + "{:.2f}".format(blob_size)
                        cv2.putText(frame, text4, (5,100), font, 1, (0, 255, 0), 2)    
                
                        cv2.circle(frame, (int(blob_x),int(blob_y)), int(blob_size / 2), (0, 255, 0), 2)

                        x = int(blob_x)
                        y = int(blob_y)
                        #colorsB = frame[blob_y,blob_x,0]
                        colorsB = frame[y,x,0]
                        colorsG = frame[y,x,1]
                        colorsR = frame[y,x,2]
                        colors = frame[y,x]
                        hsv_value= np.uint8([[[colorsB ,colorsG,colorsR ]]])
                        hsv = cv2.cvtColor(hsv_value,cv2.COLOR_BGR2HSV)
                        text5 = "HSV=" + str(hsv[0, 0, 0])+','+ str(hsv[0, 0, 1])+','+ str(hsv[0, 0, 2])
                        cv2.putText(frame, text5, (5,125), font, 1, (0, 0, 255), 2)
                        text6 = "RGB=" + str(colorsR)+','+ str(colorsG)+','+ str(colorsB)
                        cv2.putText(frame, text6, (5,150), font, 1, (0, 0, 255), 2)
                        
                        height, width, channels = frame.shape
                        text7 = "Img:W=" + str(width)+',H='+ str(height)+',Ch='+ str(channels)
                        cv2.putText(frame, text7, (5,175), font, 1, (0, 0, 255), 2)
                        FOV_x = 12.5 # from Measurement
                        FOV_y = 10.0 # from Measurement
                        text8 = "FOV(mm.)" + str(FOV_x) + ',' + str(FOV_y)
                        cv2.putText(frame, text8, (5,200), font, 1, (0, 0, 255), 2)
                        pos_x = (blob_x/width)*FOV_x;
                        pos_y = (blob_y/height)*FOV_y;
                        text9 = "Pos(mm)=" + "{:.2f}".format(pos_x) + ',' + "{:.2f}".format(pos_y)
                        cv2.putText(frame, text9, (5,225), font, 1, (0, 0, 255), 2)

                        blob_Ang = keypoints[0].angle
                        text10 = "Ang=" + "{:.2f}".format(blob_Ang)
                        cv2.putText(frame, text10, (5,250), font, 1, (0, 0, 255), 2)

                        blob_Res = keypoints[0].response
                        text11 = "Res=" + "{:.2f}".format(blob_Res)
                        cv2.putText(frame, text11, (5,270), font, 1, (0, 0, 255), 2)

                        blob_Oct = keypoints[0].octave
                        text11 = "Octave=" + "{:.2f}".format(blob_Oct)
                        cv2.putText(frame, text11, (5,290), font, 1, (0, 0, 255), 2)
                        
        #print ("HSV : " ,hsv[0, 0, 0],',',hsv[0, 0, 1], ',', hsv[0, 0, 2])
        #print("Red: ",colorsR)
        #print("Green: ",colorsG)
        #print("Blue: ",colorsB)
        #print("BRG Format: ",colors)
        #print("Coordinates of pixel: X: ",x,"Y: ",y)
        
                cv2.imshow("Blob detection", frame)

                print(text)

                key = cv2.waitKey(1)
                if key == 27:   #ESC Key
                        cap.release()
                        cv2.destroyAllWindows()
                        break



if __name__ == "__main__":
    main()



import cv2
import numpy as np

# 1. ตั้งค่าพารามิเตอร์สำหรับ SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

# กำหนดช่วง Threshold สำหรับการแปลงภาพเป็นขาว-ดำ (Binarization)
params.minThreshold = 10
params.maxThreshold = 200

# การกรองตามขนาดพื้นที่ (Area): ช่วยลด Noise จากกล้อง
params.filterByArea = True
params.minArea = 150 # ปรับเพิ่มขึ้นหากมีจุดรบกวนเล็กๆ มากเกินไป

# การกรองตามรูปทรง (Circularity, Convexity, Inertia)
params.filterByCircularity = True
params.minCircularity = 0.1 # ตรวจจับวัตถุที่มีความกลมในระดับหนึ่ง

params.filterByConvexity = True
params.minConvexity = 0.87 # ตรวจจับวัตถุที่มีความนูน

params.filterByInertia = True
params.minInertiaRatio = 0.01 # ตรวจจับวัตถุตามความเรียวยาว

# 2. สร้าง Detector จากพารามิเตอร์ที่ตั้งไว้
detector = cv2.SimpleBlobDetector_create(params)

# 3. เปิดกล้อง (ลองเปลี่ยนเลขเป็น 0, 1 หรือ 2 ตามพอร์ตที่ใช้งาน)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: ไม่สามารถเปิดกล้องได้!")
    exit()

print("เริ่มการตรวจจับ... กด 'q' เพื่อปิดโปรแกรม")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4. แปลงภาพเป็น Gray Scale เพราะ Detector ต้องการภาพแชนแนลเดียว
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 5. ตรวจจับ Blobs
    keypoints = detector.detect(gray)

    # 6. วนลูปเพื่อดึงพิกัด x, y และแสดงผล
    for kp in keypoints:
        # พิกัด x และ y (ค่ากลางของ Blob)
        x = int(kp.pt[0])
        y = int(kp.pt[1])
        # ขนาดพิกเซลของ Blob
        size = kp.size
        
        # แสดงพิกัดใน Terminal
        print(f"พบวัตถุที่: X={x}, Y={y}, Size={size:.2f}")

        # วาดพิกัดและวงกลมกำกับบนหน้าจอ
        cv2.putText(frame, f"({x}, {y})", (x + 10, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # 7. วาดวงกลมสีแดงตามขนาด Blob จริง
    output = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 8. แสดงผลลัพธ์
    cv2.imshow("Real-time Blob Detection & Coordinates", output)

    # กด 'q' เพื่อหยุดการทำงาน
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# คืนค่าทรัพยากร
cap.release()
cv2.destroyAllWindows()