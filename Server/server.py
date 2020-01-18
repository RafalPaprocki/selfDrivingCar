import enum
import io
import socket
import struct
import cv2
import numpy as np


class Server:
    def __init__(self):
        self.connection = None
        self.server_socket = None

    def start_server(self, ip, port):
        self.server_socket = socket.socket()
        self.server_socket.bind((ip, port))
        self.server_socket.listen(0)
        self.connection = self.server_socket.accept()[0].makefile('rwb')

    def send_action(self, name):
        self.connection.write(struct.pack('<L', len(name)))
        self.connection.write(bytes(name.encode("utf-8")))
        self.connection.flush()

    def receive_img_and_distance(self):
        distance = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
        image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
        print(distance)
        image_stream = io.BytesIO()
        image_stream.write(self.connection.read(image_len))

        image_stream.seek(0)
        img = cv2.imdecode(np.fromstring(image_stream.getvalue(), dtype=np.uint8), 1)
        return img, distance

    def close(self):
        self.connection.close()
        self.server_socket.close()

