# Reference: https://nanonets.com/blog/optical-flow/

import numpy as np
import cv2


def draw_flow(img, flow, step=16):
    # https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py#L25
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step /
                    2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = img
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


class ObstacleDetection:
    def __init__(self, motion_threshold: float = 50) -> None:
        self.motion_threshold = motion_threshold
        self.prev_gray = None
        self.prev_mask = None

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
            mask[magnitude > 0.01] = 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.int8((9, 9)))

            if self.prev_mask is not None:
                final_mask = mask + self.prev_mask
            else:
                final_mask = mask

            if not np.all((mask == 0)):
                self.prev_mask = mask

            if draw:
                mask = final_mask.astype('bool')
                c, h = cv2.findContours(final_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
                obstacles = image.copy()
                cv2.drawContours(obstacles, c, -1, (0, 255, 0), 1)
                opflow = draw_flow(image.copy(), flow)
                img = cv2.hconcat((obstacles, opflow))
                cv2.imshow("Obstacles - Flow", img)
                cv2.waitKey(1)
        else:
            final_mask = np.zeros_like(gray)

        # Updates previous frame
        self.prev_gray = gray

        return final_mask


if __name__ == "__main__":
    cap = cv2.VideoCapture('example_obstacles.mp4')
    detector = ObstacleDetection()

    while True:
        ret, image_np = cap.read()

        if ret:
            image_np = image_np[:, 160:-160]
            mask = detector.detect(image_np, draw=True)

        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            cap.release()
            break
