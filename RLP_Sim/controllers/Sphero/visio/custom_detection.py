# Based on https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb
import tensorflow as tf
import numpy as np
import cv2
import pathlib
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops


class ObjectDetection:

    def __init__(self):
        utils_ops.tf = tf.compat.v1
        tf.gfile = tf.io.gfile

        self.category_index = label_map_util.create_category_index_from_labelmap(
            './mscoco_label_map.pbtxt', use_display_name=True)

        self.model = self.load_model()

    def load_model(self):
        model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
        model_file = model_name + '.tar.gz'
        model_dir = tf.keras.utils.get_file(
            fname=model_name,
            origin='./' + model_file,
            untar=True
        )
        model_dir = pathlib.Path(model_dir)/"saved_model"
        model = tf.saved_model.load(str(model_dir))
        return model

    def run_inference_for_single_image(self, image):
        image = np.asarray(image)
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        model_fn = self.model.signatures['serving_default']
        output_dict = model_fn(input_tensor)
        
        # All outputs are batches tensors.
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

        # We only want to detect cats.
        box = np.asarray(output_dict['detection_boxes'])
        cls = np.asarray(output_dict['detection_classes']).astype(np.int64)
        scr = np.asarray(output_dict['detection_scores'])
        boxes = [] 
        for i in range(1, num_detections):    
            if cls[i] == 17:
                boxes.append(box[i])
        output_dict['detection_boxes'] = np.array(boxes)
        bl = (cls == 17)
        output_dict['detection_classes'] = np.extract(bl, cls)
        output_dict['detection_scores'] = np.extract(bl, scr)

        return output_dict

    def show_inference(self, image_np):
        output_dict = self.run_inference_for_single_image(image_np)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        cv2.imshow('object detection', image_np)
        cv2.waitKey(1)
        return output_dict


if __name__ == "__main__":
    detector = ObjectDetection()
    cap = cv2.VideoCapture('example.mp4')

    while True:
        ret, image_np = cap.read()
        image_np_expanded = np.expand_dims(image_np, axis=0)

        if ret:
            detector.show_inference(image_np)

        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            cap.release()
            break
