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

    def __init__(self):
        super(Sphero, self).__init__()

        # Get devices and enable
        self.timeStep = int(self.getBasicTimeStep())
        self.camera = self.getDevice('camera')
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
        if r > math.sqrt(2) * self.maxVelocity:
            r = math.sqrt(2) * self.maxVelocity

        if r < -math.sqrt(2) * self.maxVelocity:
            r = -math.sqrt(2) * self.maxVelocity

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
        print([m1, m2], v, rad * 180 / np.pi)
        print(vel)

    def stabilize(self):
        x, y, z = self.gyro.getValues()
        self.polarVel[0] = 0
        self.movePolar()

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
            print(s)
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

    def run(self):
        while self.step(self.timeStep) != -1:
            # key = self.keyboard.getKey()
            # self.controlPolar(key)
            # gyro = self.gyro.getValues()
            # print(gyro)
            cameraData = self.camera.getImage()
            image = np.frombuffer(cameraData, np.uint8).reshape(
                (self.camera.getHeight(), self.camera.getWidth(), 4)
            )
            # image = cv2.GaussianBlur(image,)
            contours, horizon = self.obstacle_detector.detect(image)
            self.obstacle_detector.draw_obstacles(image, contours, horizon)
            cv2.waitKey(16)

            gyro = self.gyro.getValues()
            next_direction = self.pathplanner.next_direction(
                gyro, contours, horizon)
            self.next_move(next_direction)


controller = Sphero()
controller.run()
