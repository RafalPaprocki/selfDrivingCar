from server import Server
from preprocessing import SignDetector
from traffic_objects import Traffic
import time
import cv2
signDetector = SignDetector('object-detection3.pbtxt',
                            'sign_graph_rpi')
server = Server()
server.start_server("0.0.0.0", 8000)
traffic = Traffic(server.connection)

try:
    start = time.time()
    fps = 0
    while True:
        img, distance = server.receive_img_and_distance()
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
        server.send_action(command.name)
        server.connection.flush()
finally:
    server.close()