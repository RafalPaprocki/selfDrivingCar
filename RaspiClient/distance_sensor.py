# Libraries
import RPi.GPIO as GPIO
import time



GPIO.setmode(GPIO.BCM)


class DistanceSensor:
    def __init__(self):
        self.stopped = False
        GPIO.setup(18, GPIO.OUT)
        GPIO.setup(24, GPIO.IN)

    def distance(self):
        GPIO.output(18, True)

        time.sleep(0.00001)
        GPIO.output(18, False)

        start_time = time.time()
        stop_time = time.time()

        while GPIO.input(24) == 0:
            start_time = time.time()

        while GPIO.input(24) == 1:
            stop_time = time.time()

        time_elapsed = stop_time - start_time
        distance = (time_elapsed * 34300) / 2

        return distance

