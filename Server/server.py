import io
import socket
import struct
import cv2
import numpy as np
import imutils
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
import enum
from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import time

from traffic_objects import Traffic


class State(enum.Enum):
   Go = 1
   Stop = 2
   Red_Light_stop = 3
   Limit_40 = 4


state = None


def load_model(model_name):
    model_file = model_name

    model_dir = pathlib.Path(model_file) / "saved_model"

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

def show_inference(model, image):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = image
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    # image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # image_np = cv2.resize(image_np, (640,320))
    cv2.imshow("win", image_np)
    return output_dict
    # cv2.waitKey(0)
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

PATH_TO_LABELS = 'C:\\Users\\rafpa\Downloads\\models\\research\object_detection\\training/object-detection3.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

model_name = 'C:\\Users\\rafpa\Downloads\\models\\research\object_detection\\sign_graph_rpi'
detection_model = load_model(model_name)



# net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ree.pbtxt')

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
        output_dict = show_inference(detection_model, img)
        if cv2.waitKey(1) & 0xFF == ord('r'):
            cv2.destroyAllWindows()
            break
        # frame = imutils.resize(img, width=400)
        #
        # (h, w) = frame.shape[:2]
        # blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
        #
        #
        # net.setInput(blob)
        # detections = net.forward()
        #
        # for i in np.arange(0, detections.shape[2]):
        #     confidence = detections[0, 0, i, 2]
        #     if confidence > 0.3:
        #         idx = int(detections[0, 0, i, 1])
        #         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        #         (startX, startY, endX, endY) = box.astype("int")
        #         print("idx: " + str(idx))
        #         print("class: " + CLASSES[idx])
        #         if idx < len(CLASSES):
        #             label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        #             y = startY - 15 if startY - 15 > 15 else startY + 15
        #             cv2.putText(frame, label, (startX, y),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        #             cv2.rectangle(frame, (startX, startY), (endX, endY),
        #                                  (0, 0, 255), 2)
        #

        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF
        if time.time() - start >= 1:
            print(fps)
            fps = 0
            start = time.time()
        fps += 1

        detection_scores = [s for s in output_dict['detection_scores'] if s > 0.5]
        detected_labels = [category_index[l]['name'] for l in output_dict['detection_classes'][0:len(detection_scores)]]
        boxes = output_dict['detection_boxes'][0:len(detection_scores)]
        for box in boxes:
            box[0] = box[0] * 380
            box[1] = box[1] * 640
            box[2] = box[2] * 380
            box[3] = box[3] * 640

        predicted = None

        for label, score, box in zip(detected_labels, detection_scores, boxes):
            if box[2] > 100:
                predicted = {"detection_scores": score, "box": box, "label": label}
                break

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


        # connection.write(struct.pack('<s', b"s"))
        # if "red_light" in predicted["labels"]:
        #     state = State.Red_Light_stop
        #     connection.write(struct.pack('<s', b"s"))
        # elif "green_light" in predicted["labels"]:
        #     if state is State.Limit_40:
        #         state = State.Go
        #         connection.write(struct.pack('<s', b"g"))
        # elif "limit40" in predicted["labels"]:
        #     if state is not State.Red_Light_stop:
        #         state = State.Limit_40
        #         connection.write(struct.pack('<s', b"o"))
        # elif "left" in predicted["labels"]:
        #     if state is not State.Red_Light_stop:
        #         state = State.Go
        #         connection.write(struct.pack('<s', b"l"))
        # elif "stop_sign" in predicted["labels"]:
        #     if state is not State.Red_Light_stop:
        #         state = State.Stop
        #         connection.write(struct.pack('<s', b"s"))
        # elif "person" in predicted["labels"]:
        #     if state is not State.Red_Light_stop:
        #         state = State.Stop
        #         connection.write(struct.pack('<s', b"s"))
        # elif "car" in predicted["labels"]:
        #     if state is not State.Red_Light_stop:
        #         state = State.Stop
        #         connection.write(struct.pack('<s', b"s"))
        # elif "end40limit" in predicted["labels"]:
        #     if state is not State.Red_Light_stop:
        #         state = State.Go
        #         connection.write(struct.pack('<s', b"s"))

        connection.flush()
finally:
    connection.close()
    server_socket.close()

