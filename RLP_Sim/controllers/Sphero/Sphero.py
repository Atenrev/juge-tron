from controller import Robot
from cmath import rect, polar
import math
import numpy as np
import struct
import cv2
from visio.obstacle_detection import ObstacleDetection
from pathplanner import PathPlanner, Movement
SPHERE_R = 0.5
WHEEL_R = 0.03


class Sphero(Robot):
    VELOCITY_MSG = 0
    velocity = [0, 0, 0]
    polarVel = [0, 0]
    cameraPosition = 0
    speed = 5
    rotationSpeed = 1 * np.pi / 180
    maxVelocity = 0
    lastMessage = None
    imageShape = None

    def __init__(self):
        super(Sphero, self).__init__()

        # Get devices and enable
        self.timeStep = int(self.getBasicTimeStep())
        self.camera = self.getDevice('camera')
        h, w = self.camera.getHeight(), self.camera.getWidth()
        self.imageShape = h, w
        self.emitter = self.getDevice('emitter')
        self.receiver = self.getDevice('receiver')
        self.accelerometer = self.getDevice('accelerometer')
        self.pen = self.getDevice('pen')
        self.gyro = self.getDevice('gyro')
        self.cameraMotor = self.getDevice('cameraMotor')

        self.gyro.enable(self.timeStep)
        self.accelerometer.enable(self.timeStep)
        self.camera.enable(self.timeStep)
        self.receiver.enable(self.timeStep)
        self.keyboard.enable(32)
        self.keyboard = self.getKeyboard()

        self.motors = []
        for i in range(3):
            m = self.getDevice(f"wheel{i}")
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
            self.motors.append(m)
        self.maxVelocity = int(self.motors[0].getMaxVelocity())

        self.obstacle_detector = ObstacleDetection(
            lower_RGB=np.array([0, 0, 0, 0]),
            upper_RGB=np.array([60, 60, 60, 255])
        )
        self.pathplanner = PathPlanner(
            self.camera.getHeight(),
            self.camera.getWidth(),
            debug=True
        )

    def sleep(self, ms):
        """
        Async sleep

        Parameter
        ---------
        ms: int
            Time to sleep in miliseconds
        """
        a = int(ms / 1000 * 1024)
        self.step(a)

    def setCamera(self, rad=None):
        """
        Set the direccion of the camera.
        If not given any ang, it will point towards the forward direcction.

        Parameter
        ---------
        rad: float, optional
            Angle in radians.
        """
        if rad is not None:
            self.cameraMotor.setPosition(rad)
            return
        v1 = rect(self.velocity[0], 0)
        v2 = rect(self.velocity[1], -2/3 * np.pi)
        v3 = rect(self.velocity[2], 2/3 * np.pi)
        s = np.array([v1, v2, v3]).sum(axis=0)
        s = polar(s)
        if s[0] != 0:
            self.cameraMotor.setPosition(s[1])

    def setVelocity(self):
        """
        Set the velocity of the motors based on `velocity`

        Notes
        -----
        Caps the velocity to `maxVelocity`
        """
        self.velocity = [self.maxVelocity if v >
                         self.maxVelocity else v for v in self.velocity]
        self.velocity = [-self.maxVelocity if v < -
                         self.maxVelocity else v for v in self.velocity]

        self.sendData()
        for i, j in zip(self.velocity, self.motors):
            j.setVelocity(i)

    def findMotors(self, rad):
        """
        Find which two motors needs to be turned on so that robot
        moves in the direccion of `rad`

        Parameter
        ---------
        rad: float
            Angle in radians

        Returns
        -------
        out: ndarray
            array of shape (1, 2) containing the index of the motors
        """
        # p1 is the direction of velocity.
        # p2 contains the directions of the motors and their inverse
        # first 3 are for positive velocity and the other for negative velocity.
        # Computes the distance from p1 to each point in p2 and
        # return the index in mod 3 of the two smallest distances.

        p1 = np.array([(1, rad)], dtype=np.float32).reshape(-1, 2)
        p2 = np.array([(1, 0),
                       (1, -2/3 * np.pi),
                       (1, 2/3 * np.pi),
                       (1, np.pi),
                       (1, 1/3 * np.pi),
                       (1, -1/3 * np.pi)], dtype=np.float32).reshape(-1, 2)

        ang1, ang2 = p1[:, 1], p2[:, 1]
        d = np.sqrt(2 * (1 - np.cos(ang2 - ang1)))
        m = np.argpartition(d, axis=0, kth=2)[:2]
        m[m > 2] -= 6
        return m

    def movePolar(self):
        """
        Convert the velocity `r` and direction `rad` to motor velocities
        """
        r, rad = self.polarVel
        if r > self.maxVelocity:
            r = self.maxVelocity

        if r < -self.maxVelocity:
            r = -self.maxVelocity

        self.polarVel[0] = r

        v = rect(r, rad)
        v = np.abs([v.real, v.imag])
        v1, v2 = v.max(), v.min()
        m1, m2 = self.findMotors(rad)

        v1 = -v1 if m1 < 0 else v1
        v2 = -v2 if m2 < 0 else v2

        v1 = -v1 if r < 0 else v1
        v2 = -v2 if r < 0 else v2

        vel = [0, 0, 0]
        vel[int(m1)] = v1
        vel[int(m2)] = v2

        self.velocity = vel

        self.setVelocity()
        # print([m1, m2], v, rad * 180 / np.pi)
        # print(vel)

    def substract_polar(self, v1, v2):
        v1 = rect(*v1)
        v2 = rect(*v2)
        v3 = v1 - v2
        print(v1, v2, v3, polar(v3))
        return polar(v3)

    def stabilize(self):
        # x, _, z = self.gyro.getValues()
        # gyro = polar(complex(x, z))
        # print(gyro[0], gyro[1]*180/np.pi)
        # r, phi = self.substract_polar(self.polarVel, gyro)
        # self.polarVel[0] = r
        # self.polarVel[1] = phi
        # print("gyro", gyro)
        # self.movePolar()
        pass

    def forward(self, speed=None, cameraLock=False):
        '''
        Move the robot in the forward direction.
        If `cameraLock` is set then the camera will be fixed in place.

        Parameter
        ---------
        dSpeed: int, optional
            Defualt is `self.speed`.

        cameraLock: bool, optional
            Camera-free movement.
        '''
        self.polarVel[0] += self.speed if speed is None else speed
        self.movePolar()
        if not cameraLock:
            self.setCamera(self.cameraPosition)

    def backward(self, speed=None, cameraLock=False):
        '''
        Move the robot in the backward direction.
        If `cameraLock` is set then the camera will be fixed in place.

        Parameter
        ---------
        dSpeed: int, optional
            Defualt is `self.speed`.

        cameraLock: bool, optional
            Camera-free movement.
        '''
        self.polarVel[0] -= self.speed if speed is None else speed
        self.movePolar()
        if not cameraLock:
            self.setCamera(self.cameraPosition)

    def rotateLeft(self, rotationSpeed=None, cameraLock=False):
        '''
        Rotate the robot to the left by `rotationSpeed` radians.
        If `cameraLock` is set then the camera will be fixed in place.

        Parameter
        ---------
        rotationSpeed: int, optional
            Defualt is `self.rotationSpeed`.

        cameraLock: bool, optional
            Camera-free rotation.
        '''
        self.polarVel[1] += self.rotationSpeed if rotationSpeed is None else rotationSpeed
        if cameraLock:
            self.movePolar()
            return
        self.cameraPosition += self.rotationSpeed if rotationSpeed is None else rotationSpeed
        self.movePolar()
        self.setCamera(self.cameraPosition)

    def rotateRight(self, rotationSpeed=None, cameraLock=False):
        '''
        Rotate the robot to the right by `rotationSpeed` radians.
        If `cameraLock` is set then the camera will be fixed in place.

        Parameter
        ---------
        rotationSpeed: int, optional
            Defualt is `self.rotationSpeed`.

        cameraLock: bool, optional
            Camera-free rotation.
        '''
        self.polarVel[1] -= self.rotationSpeed if rotationSpeed is None else rotationSpeed
        if cameraLock:
            self.movePolar()
            return
        self.cameraPosition -= self.rotationSpeed if rotationSpeed is None else rotationSpeed
        self.movePolar()
        self.setCamera(self.cameraPosition)

    def changeDirection(self, cameraLock=False):
        '''
        Reverse the direcction of movement.
        If `cameraLock` is set then the camera will be fixed in place.

        Parameter
        ---------
        cameraLock: bool, optional
            Camera-free rotation.
        '''
        if not cameraLock:
            self.polarVel[1] -= np.pi
            self.setCamera(self.cameraPosition)
        self.polarVel[0] = -self.polarVel[0]
        self.movePolar()

    def stop(self):
        '''
        Stop the robot
        '''
        self.polarVel[0] = 0
        self.cameraPosition = self.polarVel[1]
        self.movePolar()
        self.setCamera(self.cameraPosition)

    def sendData(self):
        '''
        Send data to the plugin
        '''

        s = struct.pack('ifff', self.VELOCITY_MSG, *self.velocity)
        if s != self.lastMessage:
            # print(s)
            self.lastMessage = s
            self.emitter.send(s)

    def getGyroValues(self):
        '''
        Get raw gyroscope values.

        Returns
        -------
        out: list
            angular velocity along the x, y, z axes in rad/s
        '''
        return self.gyro.getValues()

    def getAccValues(self):
        '''
        Get raw accelerometer values.

        Returns
        -------
        out: list
            acceleration along the x, y, z axes in m/s
        '''
        return self.accelerometer.getValues()

    def penWrite(self, write):
        self.pen.write(write)

    def controlPolar(self):
        '''
        Control the robot using the keyboard.

        W, S, A, D

        To move the robot without moving the camera use.

        I, K, J, L

        Stop the robot by pressing 

        SPACE
        '''
        key = self.keyboard.getKey()

        if key == ord('W'):
            self.forward()

        if key == ord('S'):
            self.backward()

        if key == ord('A'):
            self.rotateLeft()

        if key == ord('D'):
            self.rotateRight()

        if key == ord('Q'):
            self.changeDirection()

        if key == ord('I'):
            self.forward(cameraLock=True)

        if key == ord('K'):
            self.backward(cameraLock=True)

        if key == ord('J'):
            self.rotateLeft(cameraLock=True)

        if key == ord('L'):
            self.rotateRight(cameraLock=True)

        if key == ord('U'):
            self.changeDirection(cameraLock=True)

        if key == ord(' '):
            self.stop()

    def getImage(self):
        buffer = self.camera.getImage()
        h, w = self.imageShape
        img = np.frombuffer(buffer, np.uint8).reshape(
            (h, w, 4))[:, :, :3]
        return img

    def next_move(self, next_direction):
        if next_direction == Movement.STABILIZE:
            self.stabilize()

        if next_direction == Movement.STOP:
            self.stop()

        if next_direction == Movement.FORWARD:
            self.forward()

        if next_direction == Movement.BACKWARDS:
            self.backward()

        if next_direction == Movement.ROTATE_LEFT:
            self.rotateLeft()

        if next_direction == Movement.ROTATE_RIGHT:
            self.rotateRight()

    def detectObstacle(self):
        # https://github.com/BigFace83/BFRMR1/blob/167b9a152960246f5dd431af4718bff8b2a81a87/BFRMR1OpenCV.py#L230
        img = self.getImage()
        # convert img to grayscale and store result in imgGray
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blur the image slightly to remove noise
        imgGray = cv2.bilateralFilter(imgGray, 9, 30, 30)
        imgEdge = cv2.Canny(imgGray, 50, 100)  # edge detection

        imagewidth = imgEdge.shape[1] - 1
        imageheight = imgEdge.shape[0] - 1
        img2 = img.copy()
        for j in range(0, imagewidth, 8):  # for the width of image array
            # step through every pixel in height of array from bottom to top
            for i in range(imageheight, 0, -1):
                # Ignore first couple of pixels as may trigger due to undistort
                # check to see if the pixel is white which indicates an edge has been found
                if imgEdge.item(i, j) == 255:
                    # if it is, add x,y coordinates to ObstacleArray
                    img2[:i, :j] = 0
                    break  # if white pixel is found, skip rest of pixels in column
        # cv2.imshow("", imgEdge)
        self._prepare_subimage_for_detection(img)
        # cv2.imshow("", imgEdge)
        cv2.imshow("", img2)
        cv2.waitKey(1)
        return img2

    def asd(self, lower_RGB=np.array([0, 0, 0]), upper_RGB=np.array([255, 255, 255]), contour_threshold=300):
        self.lower_RGB = lower_RGB.astype('uint8')
        self.upper_RGB = upper_RGB.astype('uint8')
        self.contour_threshold = contour_threshold
        self.images = []
        self.prev_gray = None
        self.mask = np.zeros((256, 256, 3))
        # Sets image saturation to maximum
        self.mask[..., 1] = 255

    def _prepare_subimage_for_detection(self, image):
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        # ret, frame = cap.read()
        # Opens a new window and displays the input frame
        # cv.imshow("input", frame)
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        def sigmoid(x):
            z = np.exp(-x)
            return 1 / (1 + z)

        if self.prev_gray is not None:
            # Calculates dense optical flow by Farneback method
            # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # Computes the magnitude and angle of the 2D vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # Sets image hue according to the optical flow direction
            # self.mask[..., 0] = angle * 180 / np.pi
            # Sets image value according to the optical flow magnitude (normalized)
            np.save("n.npy", magnitude)
            self.mask[..., 2] = cv2.normalize(sigmoid(magnitude),
                                              None, 0, 255, cv2.NORM_MINMAX)
            # Converts HSV to RGB (BGR) color representation
            rgb = cv2.cvtColor(self.mask.astype(np.float32), cv2.COLOR_HSV2BGR)
            # rgb[:,:,2] = np.where(rgb[:,:,2] < 50, 0, rgb[:,:,2])
            # Opens a new window and displays the output frame
            # for i in range(0, rgb.shape[0], 8):
            #     for j in range(0, rgb.shape[1], 8):

            # cv2.arrowedLine(rgb, flow[i, j, 0],
            #                 flow[i, j, 1], (255, 0, 0), 1)
            cv2.imshow("dense optical flow", rgb)

        # Updates previous frame
        self.prev_gray = gray
        return gray

    def run(self):
        self.i = 0
        self.asd()
        # out = cv2.VideoWriter('1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 24, self.imageShape)
        between_time = 0.064
        last_frame_time = 0
        import time
        while self.step(self.timeStep) != -1:
            frame_time = time.time()
            self.controlPolar()
            if last_frame_time + between_time < frame_time:
                self._prepare_subimage_for_detection(self.getImage())
                last_frame_time = frame_time
                cv2.waitKey(1)
            # self.detectObstacle()
            # out.write(self.getImage())


controller = Sphero()
controller.run()

# out.release()
