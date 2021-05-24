# https://github.com/ultralytics/yolov5
import numpy as np
import cv2
import torch
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size,  non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

class Yolo:
    def __init__(self, weights='yolov5s.pt', view_img=True, imgsz=256,device = 'cpu') -> None:
        self.device = select_device(device)
        self.view_img = view_img
        # Load model
        self.model = attempt_load(
            weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names  # get class names
        return

    def show_inference_and_return(self, im0s):
        # Initialize
        print("search for a cat")
        img = letterbox(im0s)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred)

        # Process detections
        cat = False
        for det in pred:  # detections per image
            s, im0 = '', im0s.copy()
            s += '%gx%g ' % img.shape[2:]  # print string
            # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        if self.names[c] == 'cat':
                            cat = True
                            label = f'{self.names[c]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label)

            # Stream results
            if self.view_img:
                cv2.imshow("", im0)
                cv2.waitKey(1)

        if cat:
            return ['there\'s a cat!!']
        else:
            return []


if __name__ == '__main__':
    im0s = cv2.imread('./yolo/cat.jpg')
    im0s = cv2.resize(im0s, (1920//2, 1024//2))
    y = Yolo()
    from time import time
    t = time()
    y.show_inference_and_return(im0s)
    print(time() - t)
    t = time()
    y.show_inference_and_return(im0s)
    print(time() - t)
