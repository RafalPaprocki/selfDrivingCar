import io
import socket
import struct
import cv2
import numpy as np
import enum
from preprocessing import SignDetector
import time

from traffic_objects import Traffic


class State(enum.Enum):
   Go = 1
   Stop = 2
   Red_Light_stop = 3
   Limit_40 = 4


signDetector = SignDetector('C:\\Users\\rafpa\Downloads\\models\\research\object_detection\\training/object-detection3.pbtxt',
                            'C:\\Users\\rafpa\Downloads\\models\\research\object_detection\\sign_graph_rpi')

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
        # output_dict = show_inference(detection_model, img)
        if cv2.waitKey(1) & 0xFF == ord('r'):
            cv2.destroyAllWindows()
            break

        if time.time() - start >= 1:
            print(fps)
            fps = 0
            start = time.time()
        fps += 1

        if predicted is not None:
            eval("traffic." + predicted["label"] + "_action")()
        elif distance < 10:
            if traffic.state["red_light"]:
                connection.write(struct.pack('<L', 4))
                connection.write(b"stop")
                connection.flush()
            else:
                connection.write(struct.pack('<L', 8))
                connection.write(b"Withdraw")
                connection.flush()
        else:
            if traffic.state["red_light"]:
                connection.write(struct.pack('<L', 4))
                connection.write(b"stop")
                connection.flush()
            else:
                connection.write(struct.pack('<L', 9))
                connection.write(b"Normal_GO")
                connection.flush()

        connection.flush()
finally:
    connection.close()
    server_socket.close()

