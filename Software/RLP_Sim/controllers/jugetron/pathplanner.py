from __future__ import annotations
from enum import Enum
import time
import numpy as np


class State(Enum):
    FLEE = 0
    SEARCH_CAT = 1
    SPRINT = 2
    STABILIZE = 3


class StateMachine:
    def __init__(self, state: State) -> None:
        self.state = state
        self.ended = False

        if state == State.STABILIZE:
            self.counter = -1
            self.timelimit = 3
        elif state == State.FLEE:
            self.set_counter()
            self.timelimit = 30 #np.random.randint(45, 55)
        elif state == State.SEARCH_CAT:
            self.set_counter()
            self.timelimit = 25
        elif state == State.SPRINT:
            self.set_counter()
            self.timelimit = 20

    def _end_state_by_time(self):
        return self.counter + self.timelimit < time.time()

    def set_counter(self) -> None:
        self.counter = time.time()

    def end_state(self):
        self.ended = True

    def update(self, gyro_magnitude: float) -> None:
        if self.state == State.STABILIZE:
            if gyro_magnitude < 1.5 and self.counter == -1:
                self.set_counter()

    def next_state(self, gyro_magnitude: float) -> StateMachine:
        if self.state == State.STABILIZE:
            if self.counter != -1 and self._end_state_by_time():
                return StateMachine(State.FLEE)
            else:
                return self

        elif self.state == State.FLEE:
            if gyro_magnitude > 3:
                return StateMachine(State.STABILIZE)
            elif self._end_state_by_time():
                return StateMachine(State.SEARCH_CAT)
            else:
                return self

        elif self.state == State.SEARCH_CAT:
            if self.ended:
                return StateMachine(State.SPRINT)
            elif self._end_state_by_time():
                return StateMachine(State.FLEE)
            else:
                return self

        elif self.state == State.SPRINT:
            if self._end_state_by_time():
                return StateMachine(State.FLEE)
            else:
                return self


class PathPlanner:
    def __init__(self, camera_width: float, camera_heigth: float,
                 debug: bool = False, options: dict = None) -> None:
        self.debug = debug
        self.camera_width = camera_width
        self.camera_width_half = camera_width / 2
        self.camera_heigth = camera_heigth
        self.screen_area = camera_heigth * camera_width
        self._init_options(options)
        self.last_obstacle_values = []
        self.last_floor_values = []
        self.current_state = StateMachine(State.FLEE)

    def _init_options(self, options: dict = None) -> None:
        if options is None:
            options = dict()
        if 'max_obstacle_values' not in options:
            options['max_obstacle_values'] = 1
        if 'velocity_weight' not in options:
            options['velocity_weight'] = 400
        if 'direction_weight' not in options:
            options['direction_weight'] = 1.2
        if 'direction_bonus_weight' not in options:
            options['direction_bonus_weight'] = 0.85
        if 'direction_bonus_threshold' not in options:
            options['direction_bonus_threshold'] = 40

        self.options = options

    def _search_cat(self, bounding_boxes: np.array) -> list:
        if len(bounding_boxes) > 0:
            self.current_state.end_state()
        
        return [0, 0.015]

    def _sprint(self, motion_image: np.array) -> list:
        output = self._flee(motion_image)
        return [output[0] * 10, output[1] * 1.05]

    def _flee(self, motion_image: np.array) -> list:
        """
        Calcula les velocitats objectiu basant-se en la posició
        dels obstacles i l'horitzó.
        """
        def last_nonzero(arr, axis, invalid_val=-1):
            # https://stackoverflow.com/questions/47269390/numpy-how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array
            mask = arr != 0
            val = arr.shape[axis] - \
                np.flip(mask, axis=axis).argmax(axis=axis) - 1
            return np.where(mask.any(axis=axis), val, invalid_val)

        horizon = last_nonzero(motion_image, axis=0, invalid_val=0)
        horizon_norm = horizon / motion_image.shape[1]
        horizon_average = horizon.mean()
        left_limit = horizon[0]
        right_limit = horizon[-1]
        horizon_min = min(left_limit, right_limit)

        avg = motion_image.shape[1] / (2 * motion_image.shape[1])
        c = np.linspace(0, 1, horizon_norm.shape[0]) - avg
        direction = (c * horizon_norm ** 8).sum()

        self.last_obstacle_values.append(direction)

        if len(self.last_obstacle_values) > self.options['max_obstacle_values']:
            self.last_obstacle_values.pop(0)

        direction = np.mean(self.last_obstacle_values)

        if (motion_image.shape[0] - self.options['direction_bonus_threshold']) < horizon_min:
            direction *= self.options['direction_bonus_weight']

        vertical_velocity = (
            motion_image.shape[0] - horizon_average - 60) / motion_image.shape[0]

        return [self.options['velocity_weight'] * vertical_velocity,  # expit(horizon_average),
                self.options['direction_weight'] * direction * np.pi / 180]

    def _stabilize(self) -> list:
        return [0,  0]

    def _next_state(self, gyro: list) -> None:
        gyro_magnitude = np.linalg.norm(gyro)
        self.current_state.update(gyro_magnitude)
        self.current_state = self.current_state.next_state(gyro_magnitude)

    def next_direction(self, gyro: list, vision_image: np.array) -> list:
        if self.current_state.state == State.FLEE:
            output = self._flee(vision_image)
        elif self.current_state.state == State.SEARCH_CAT:
            output = self._search_cat(vision_image)
        elif self.current_state.state == State.SPRINT:
            output = self._sprint(vision_image)
        else:
            output = self._stabilize()

        self._next_state(gyro)
        return output
