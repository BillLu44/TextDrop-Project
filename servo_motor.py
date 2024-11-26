import RPi.GPIO as GPIO
import time

# Function to set the servo's angle
def SetAngle(angle):
    # Convert angle to duty cycle value
    duty = angle / 18 + 2
    # Set the PWM duty cycle
    servo1.ChangeDutyCycle(duty)
    time.sleep(1)  # Allow servo to move to the position
    servo1.ChangeDutyCycle(0)  # Stop sending pulses

# Stop PWM when finished turning
def Cleanup():
    servo1.stop()
    GPIO.cleanup()

if __name__ == '__main__':
    # Set GPIO numbering mode
    GPIO.setmode(GPIO.BOARD)

    # PIN NUMBERING
    # 2 - Red wire for power (Vservo, battery positive terminal)
    # 3 - Yellow wire for control signalling
    # 6 - Brown wire for ground (GND, battery negative terminal)

    # Set output pin as pin 11 (GPIO 17)
    GPIO.setup(11, GPIO.OUT)

    # Set PWM pin to 50 Hz (standard for servos)
    servo1 = GPIO.PWM(11, 50)

    # Start PWM with a duty cycle of 0 (no initial movement)
    servo1.start(0)