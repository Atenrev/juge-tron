# Reference: https://github.com/rrichards7/Obstacle-Avoidance-CT

import numpy as np
from skimage import filters
from scipy import ndimage
from matplotlib import pyplot as plt
import cv2


class ObstacleDetection:
    def __init__(self, lower_RGB=np.array([0, 0, 0]), upper_RGB=np.array([50, 50, 50]), contour_threshold=300):
        self.lower_RGB = lower_RGB
        self.upper_RGB = upper_RGB
        self.contour_threshold = contour_threshold

    def detect(self, image):
        # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # l, a, b = cv2.split(lab)
        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # cl = clahe.apply(l)
        # limg = cv2.merge((cl, a, b))
        # final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        # upper_RGB = list(self.upper_RGB)
        # upper_RGB[0] = filters.threshold_otsu(image[:, :, 0])
        # upper_RGB[1] = filters.threshold_otsu(image[:, :, 1])
        # upper_RGB[2] = filters.threshold_otsu(image[:, :, 2])
        # upper_RGB = np.array(upper_RGB)
        # img_threshold = cv2.inRange(
        #     final,
        #     self.lower_RGB,
        #     upper_RGB
        # )
        # img_threshold = cv2.inRange(final, self.lower_RGB, self.upper_RGB)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detecció de l'horitzó
        # Detecció d'obstacles
        kernel = np.ones((8, 8), np.float32)/32
        gray = cv2.filter2D(gray, -1, kernel)
        equalized = cv2.equalizeHist(gray)
        equalized = cv2.morphologyEx(
            (equalized / 16).astype('uint8') * 16,
            cv2.MORPH_CLOSE,
            kernel=np.ones((3, 3), np.uint8)
        )
        img_threshold = ((equalized < filters.threshold_otsu(equalized))
                         * 255).astype('uint8')
        # img_threshold = cv2.morphologyEx(
        #     img_threshold,
        #     cv2.MORPH_CLOSE,
        #     kernel=np.ones((3, 3), np.uint8)
        # )
        contours, hierarchy = cv2.findContours(
            img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Based on https://github.com/sallamander/horizon-detection/blob/c1a555d78a03573e98bcb3f9da1f31e874616399/utils.py#L8
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = ((gray > filters.threshold_otsu(gray))
                * 255).astype('uint8')
        gray = cv2.morphologyEx(
            gray,
            cv2.MORPH_CLOSE,
            kernel=np.ones((9, 9), np.uint8)
        )

        horizon_x1 = 0
        horizon_x2 = gray.shape[1] - 1
        
        try:
            horizon_y1 = max(np.where(gray[:, horizon_x1] == 0)[0])
            horizon_y2 = max(np.where(gray[:, horizon_x2] == 0)[0])
        except:
            horizon_y1 = gray.shape[0] // 2
            horizon_y2 = gray.shape[0] // 2
    
        # Només ens interessen els rectangles dels contorns amb una àrea
        # més gran que un threshold
        return [contour for contour in contours
                if cv2.contourArea(contour) > self.contour_threshold], ((horizon_x1, horizon_y1), (horizon_x2, horizon_y2))

    def draw_obstacles(self, image, contours, line):
        img_contours = np.array(image, dtype='uint8')

        cv2.line(img_contours, line[0], line[1], (255, 0, 0), 2)
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(img_contours, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("Image", img_contours)


if __name__ == "__main__":
    import time

    image_np = cv2.imread('simul_env_ex.PNG')
    detector = ObstacleDetection()
    # contours, _ = detector.detect(image_np)
    # detector.draw_obstacles(image_np, contours)
    # cv2.waitKey(0)

    cap = cv2.VideoCapture('example_obstacles.mp4')

    while True:
        ret, image_np = cap.read()

        if ret:
            image_np = image_np[:, 160:-160]
            contours, _ = detector.detect(image_np)
            detector.draw_obstacles(image_np, contours)

        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            cap.release()
            break
