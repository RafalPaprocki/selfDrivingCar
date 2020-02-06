import cv2
import numpy as np
import tensorflow as tf
import pathlib
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile


class SignDetector:
    def __init__(self, path_to_labels, model_name ):
        self.path_to_labels = path_to_labels
        self.category_index = label_map_util.create_category_index_from_labelmap(self.path_to_labels, use_display_name=True)
        self.model_name = model_name
        self.model = self.load_model(self.model_name)

    def load_model(self, model_name):
        model_file = model_name

        model_dir = pathlib.Path(model_file) / "saved_model"

        model = tf.saved_model.load(str(model_dir))
        model = model.signatures['serving_default']

        return model

    def run_inference_for_single_image(self, model, image):
        image = np.asarray(image)
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]

        output_dict = model(input_tensor)
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key: value[0, :num_detections].numpy()
                       for key, value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        if 'detection_masks' in output_dict:
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                               tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

        return output_dict

    def show_inference(self, model, image):
        image_np = image
        output_dict = self.run_inference_for_single_image(model, image_np)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        cv2.imshow("win", image_np)
        return output_dict

    def detect_objects(self, image):
        output_dict = self.show_inference(self.model, image)
        detection_scores = [s for s in output_dict['detection_scores'] if s > 0.5]
        detected_labels = [self.category_index[l]['name'] for l in
                           output_dict['detection_classes'][0:len(detection_scores)]]
        boxes = output_dict['detection_boxes'][0:len(detection_scores)]
        for box in boxes:
            box[0] = box[0] * 380
            box[1] = box[1] * 640
            box[2] = box[2] * 380
            box[3] = box[3] * 640

        predicted = None

        for label, score, box in zip(detected_labels, detection_scores, boxes):
            if box[2] > 175: #było 100
                predicted = {"detection_scores": score, "box": box, "label": label}
                break

        return predicted

