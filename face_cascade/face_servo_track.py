import cv2
import os
import RPi.GPIO as GPIO
import time

# --- Setup GPIO ---
servo_pin = 3
GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)

pwm = GPIO.PWM(servo_pin, 50)  # 50Hz
pwm.start(0)  # Initial Duty Cycle

def ServoDegree(degree):
    """
    Convert Servo degree to PWM duty cycle based on the formula provided:
    duty = (degree / 18) + 2.5
    """
    if degree < 0:
        degree = 0
    elif degree > 180:
        degree = 180
    
    duty = (degree / 18) + 2.5
    pwm.ChangeDutyCycle(duty)

# --- Setup Face Detection ---
# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the XML file
cascPath = os.path.join(current_dir, "haarcascade_frontalface_default.xml")

# Check if XML exists
if not os.path.exists(cascPath):
    print(f"ERROR: XML file not found at: {cascPath}")
    print("Please copy 'haarcascade_frontalface_default.xml' to this directory.")
    # Fallback to local filename just in case
    cascPath = "haarcascade_frontalface_default.xml" 

faceCascade = cv2.CascadeClassifier(cascPath)
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(-1) # Try this if 0 doesn't work

if not cap.isOpened():
    print("ERROR: Could not open camera (VideoCapture).")
    print("Common fixes on Raspberry Pi:")
    print("1. Enable Legacy Camera in raspi-config")
    print("2. Try VideoCapture(-1)")
    print("3. Check connection")
    exit()

print("Face Tracking Servo Started...")
print("Press ESC to exit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame")
            print("Troubleshooting:")
            print("1. Try changing 'cap = cv2.VideoCapture(0)' to -1 or 1")
            print("2. Run 'libcamera-hello' in terminal to check if camera works")
            print("3. Check 'sudo raspi-config' > Interface > Legacy Camera")
            break

        # Mirror the frame (optional, makes it easier to track movement)
        # frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Get screen width for mapping
        height, width, channels = frame.shape

        if len(faces) > 0:
            # Track the first face found
            (x, y, w, h) = faces[0]

            # Calculate Center X of the face
            centerX = x + (w // 2)

            # Map centerX (0 to width) to Degree (0 to 180)
            # Formula: degree = (centerX / width) * 180
            # Note: You might need to invert this (180 - degree) depending on servo mounting
            target_degree = (centerX / width) * 180
            
            # Move Servo
            ServoDegree(target_degree)

            # Visuals
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Angle: {int(target_degree)}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # If no face, maybe stop servo or return to center?
            # pwm.ChangeDutyCycle(0) # Stop sending signal
            pass

        cv2.imshow("Face Tracking Servo", frame)

        if cv2.waitKey(1) == 27:
            break

except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"Error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pwm.stop()
    GPIO.cleanup()
