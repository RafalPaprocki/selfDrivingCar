import enum
import io
import socket
import struct
import cv2
import numpy as np
from preprocessing import SignDetector
import time

from traffic_objects import Traffic
signDetector = SignDetector('object-detection3.pbtxt',
                            'sign_graph_rpi')


from commands import Commands


def send_action(name):
    connection.write(struct.pack('<L', len(name)))
    connection.write(bytes(name.encode("utf-8")))
    connection.flush()


server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)

connection = server_socket.accept()[0].makefile('rwb')
traffic = Traffic(connection)
try:
    start = time.time()
    fps = 0
    while True:
        distance = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        print(distance)
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))

        image_stream.seek(0)
        img = cv2.imdecode(np.fromstring(image_stream.getvalue(), dtype=np.uint8), 1)
        predicted = signDetector.detect_objects(img)
        if cv2.waitKey(1) & 0xFF == ord('r'):
            cv2.destroyAllWindows()
            break

        if time.time() - start >= 1:
            print(fps)
            fps = 0
            start = time.time()
        fps += 1

        if predicted is not None:
            command = eval("traffic." + predicted["label"] + "_action")()
        else:
            command = traffic.none_traffic_object_action(distance)
        send_action(command.name)
        connection.flush()
finally:
    connection.close()
    server_socket.close()

#enum z komendami jakie można wysłać - gotowe
#wtraffic_objects determining which command should be perform on client gotowe
#server handle this command from traffic object id and send it using send_action function
