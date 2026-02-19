import RPi.GPIO as GPIO
import time

servo_pin = 3

GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)

pwm = GPIO.PWM(servo_pin, 50)  # 50Hz
pwm.start(0)

try:
    while True:
        # ?? 90 ????
        pwm.ChangeDutyCycle(7.5)
        time.sleep(2)

        # ???? 0 ????
        pwm.ChangeDutyCycle(2.5)
        time.sleep(2)

except KeyboardInterrupt:
    pass

pwm.stop()
GPIO.cleanup()
