# Reference: https://github.com/rrichards7/Obstacle-Avoidance-CT

import numpy as np
import cv2


class ObstacleDetection:
    def __init__(self, motion_threshold: float = 50) -> None:
        self.motion_threshold = motion_threshold
        self.prev_gray = None
        self.mask = None

    def detect(self, image: np.array, draw: bool = False) -> np.array:
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is not None:
            # Calculates dense optical flow by Farneback method
            # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # Computes the magnitude and angle of the 2D vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            mask = np.zeros_like(magnitude, dtype='uint8')
            mask[magnitude > 1] = 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.int8((9, 9)))
            self.mask = mask
            # Opens a new window and displays the output frame
            if draw:
                im = image.copy()
                im[np.bool8(mask)] = 0
                cv2.imshow("dense optical flow", im)
                cv2.waitKey(1)
        else:
            mask = np.zeros_like(gray)

        # Updates previous frame
        self.prev_gray = gray

        return mask
