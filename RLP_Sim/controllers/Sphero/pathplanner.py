from enum import Enum
import time
import cv2
import numpy as np


class State(Enum):
    FLEE = 0
    NEW_ORIENTATION = 1
    STABILIZE = 2


class Movement(Enum):
    BACKWARDS = -2
    STOP = -1
    STABILIZE = 0
    FORWARD = 1
    ROTATE_LEFT = 2
    ROTATE_RIGHT = 3


class StateMachine:
    def __init__(self, state):
        self.state = state

        if state == State.STABILIZE:
            self.counter = -1
            self.timelimit = 3
        elif state == State.FLEE:
            self.set_counter()
            self.timelimit = np.random.randint(10, 16)
        elif state == State.NEW_ORIENTATION:
            self.set_counter()
            self.timelimit = -1

    def set_counter(self):
        self.counter = time.time()

    def update(self, gyro_magnitude):
        if self.state == State.STABILIZE:
            if gyro_magnitude < 1 and self.counter == -1:
                self.set_counter()

    def next_state(self, gyro_magnitude):
        if self.state == State.STABILIZE:
            if self.counter != -1 and self.counter + self.timelimit < time.time():
                return StateMachine(State.FLEE)
            else:
                return self

        elif self.state == State.FLEE:
            if gyro_magnitude > 1.5:
                return StateMachine(State.STABILIZE)
            else:
                return self

        elif self.state == State.NEW_ORIENTATION:
            # TODO
            pass


class PathPlanner:
    def __init__(self, camera_width, camera_heigth, debug=False, options=None):
        self.debug = debug
        self.camera_width = camera_width
        self.camera_width_half = camera_width / 2
        self.camera_heigth = camera_heigth
        self.screen_area = camera_heigth * camera_width
        self._init_options(options)
        self.last_obstacle_values = []
        self.last_floor_values = []
        self.current_state = StateMachine(State.FLEE)

    def _init_options(self, options=None):
        if options is None:
            options = dict()
        if 'dir_threshold' not in options:
            options['dir_threshold'] = 1.5
        if 'obstacle_weight' not in options:
            options['obstacle_weight'] = 5
        if 'max_obstacle_values' not in options:
            options['max_obstacle_values'] = 2
        if 'floor_weight' not in options:
            options['floor_weight'] = 10
        if 'max_floor_values' not in options:
            options['max_floor_values'] = 100

        self.options = options

    def _new_orientation(self):
        pass

    def _flee(self, contours, horizon):
        """
        Calcula la direcció del proper moviment basant-se en la posició
        dels obstacles.
        """
        direction = 0
        total_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            total_area += area
            # Calculem el centre del contorn de l'obstacle
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            # cY = int(M["m01"] / M["m00"])
            # print(M["m01"])
            (x, y, w, h) = cv2.boundingRect(contour)
            ypos = y+h

            if ypos > self.camera_heigth / 2:
                new_direction = ((self.camera_width_half - cX) / self.camera_width) * \
                    (ypos / self.camera_heigth)  # * ((area) / self.screen_area)
                direction += new_direction * self.options['obstacle_weight']

        floor = self.camera_heigth - max(horizon[0][1], horizon[1][1])
        self.last_floor_values.append(floor)
        if len(self.last_floor_values) > self.options['max_floor_values']:
            self.last_floor_values.pop(0)

        if self.debug:
            print(
                "Direction threshold:", self.options['dir_threshold'],
                "Total area:", total_area,
                "Direction:", direction
            )

        if np.mean(self.last_floor_values) < self.camera_heigth / 3:
            if horizon[1][1] > horizon[0][1]:
                # Si la paret la tenim a la nostra dreta, volem girar a l'esquerra
                direction -= horizon[1][1] / \
                    self.camera_heigth * self.options['floor_weight']
            else:
                # Si no a la dreta
                direction += horizon[1][1] / \
                    self.camera_heigth * self.options['floor_weight']

        self.last_obstacle_values.append(direction)
        if len(self.last_obstacle_values) > self.options['max_obstacle_values']:
            self.last_obstacle_values.pop(0)
        direction = np.mean(self.last_obstacle_values)

        if (direction < self.options['dir_threshold']
                and direction > -self.options['dir_threshold']):
                # and total_area < self.screen_area / 4):
            if self.debug:
                print("GOING FORWARD")
            return Movement.FORWARD
        elif direction > self.options['dir_threshold']:
            if self.debug:
                print("ROTATING RIGHT")
            return Movement.ROTATE_RIGHT
        else:
            if self.debug:
                print("ROTATING LEFT")
            return Movement.ROTATE_LEFT

    def _stabilize(self):
        return Movement.STABILIZE

    def _next_state(self, gyro):
        gyro_magnitude = np.linalg.norm(gyro)
        self.current_state.update(gyro_magnitude)
        self.current_state = self.current_state.next_state(gyro_magnitude)
        print("gyro magnitude:", gyro_magnitude, "state:", self.current_state.state)

    def next_direction(self, gyro, contours, horizon):
        self._next_state(gyro)

        if self.current_state.state == State.FLEE:
            return self._flee(contours, horizon)
        else:
            return self._stabilize()
