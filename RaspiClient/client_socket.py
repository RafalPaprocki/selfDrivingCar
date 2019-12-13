from camera_pi import Camera
import io
import socket
import struct
import time
import picamera
import RPi.GPIO as GPIO
from gpiozero import Robot
from distance_sensor import DistanceSensor
LEFT_FRONT_MOTOR = 5
LEFT_REAR_MOTOR = 26
RIGHT_FRONT_MOTOR = 13
RIGHT_REAR_MOTOR = 6
robby = Robot(left=(13,6), right=(5,26))
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(LEFT_FRONT_MOTOR, GPIO.OUT)
GPIO.setup(LEFT_REAR_MOTOR, GPIO.OUT)
GPIO.setup(RIGHT_FRONT_MOTOR, GPIO.OUT)
GPIO.setup(RIGHT_REAR_MOTOR, GPIO.OUT)
camera = Camera()
camera.initialize()
frame = camera.take_frame()
distance_sensor = DistanceSensor()
client_socket = socket.socket()
client_socket.connect(('192.168.137.1', 8000))
connection = client_socket.makefile('rwb')

try:
    
    
    while True:
        frame = camera.take_frame()
        size = len(frame)
        distance = distance_sensor.distance()
        connection.write(struct.pack('<L', int(distance)))
        connection.write(struct.pack('<L', size))
        connection.write(frame)
        connection.flush()
        length = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        action = connection.read(length).decode()
        print(action)
        if action == "stop":
            robby.stop()
        elif action == 'Normal_GO':
            robby.backward(0.75)
        elif action == 'Limit40_GO':
            robby.backward(0.60)
        elif action == 'left':
            robby.left(1)
            time.sleep(0.2)
        elif action == 'right':
            robby.right(1)
            time.sleep(0.2)
        elif action == 'Withdraw':
            robby.forward(1)
            time.sleep(0.8)
            robby.left(1)
            time.sleep(0.5)

finally:
    connection.close()
    client_socket.close()
