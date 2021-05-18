from __future__ import annotations
from enum import Enum
import time
import cv2
import numpy as np
from scipy.special import expit


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
    def __init__(self, state: State) -> None:
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

    def set_counter(self) -> None:
        self.counter = time.time()

    def update(self, gyro_magnitude: float) -> None:
        if self.state == State.STABILIZE:
            if gyro_magnitude < 3 and self.counter == -1:
                self.set_counter()

    def next_state(self, gyro_magnitude: float) -> StateMachine:
        if self.state == State.STABILIZE:
            if self.counter != -1 and self.counter + self.timelimit < time.time():
                return StateMachine(State.FLEE)
            else:
                return self

        elif self.state == State.FLEE:
            if gyro_magnitude > 3:
                return StateMachine(State.STABILIZE)
            else:
                return self

        elif self.state == State.NEW_ORIENTATION:
            # TODO
            pass


class Node:
    def __init__(self, value: int, point: list) -> None:
        self.value = value
        self.point = point
        self.parent = None
        self.H = 0
        self.G = 0

    def move_cost(self, other: Node) -> int:
        return other.value

    def __eq__(self, other) -> bool:
        if not isinstance(other, Node):
            return NotImplemented

        return self.point[0] == other.point[0] and self.point[1] == other.point[1]

    def __hash__(self):
        return hash((self.point[0], self.point[1]))

    def __str__(self) -> str:
        return f'Point: {self.point[0]}, {self.point[1]}\tValue: {self.value}\tCost: {self.G}'


class AStar:
    def neighbours(self, point: list, image: np.array) -> list:
        x, y = point.point
        return [Node(image[d[0], d[1]], d)
                for d in [(x-1, y), (x, y - 1), (x, y + 1), (x+1, y)]
                if (d[0] >= 0 and d[0] < image.shape[0]
                    and d[1] >= 0 and d[1] < image.shape[1]
                    and image[d[0], d[1]] < 2)
                ]

    def manhattan(self, point: list, point2: list) -> int:
        # + abs(point.point[1]-point2.point[1])
        return abs(point.point[0] - point2.point[0])

    def search(self, image: np.array) -> np.array:
        grid = np.zeros_like(image)
        contours, _ = cv2.findContours(
            image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            M = cv2.moments(c)
            cX = int(M["m10"] / (M["m00"]+1))
            cY = int(M["m01"] / (M["m00"]+1))
            grid[cY, cX] = 1

        import scipy.ndimage.morphology as morph
        grid = morph.distance_transform_edt(1 - grid)
        grid = 1 - grid/np.max(grid)
        grid = grid * (1-(image/255)) + image/255
        grid = grid*2
        openset = set()
        closedset = set()
        current = Node(0, [grid.shape[0]-1, grid.shape[1] // 2])
        goal = Node(0, [0, grid.shape[1] // 2])
        openset.add(current)

        while openset:
            current = min(openset, key=lambda o: o.G + o.H)

            if current.point[0] == 0 and current.value < 2:
                path = []

                while current.parent:
                    path.append(current.point)
                    current = current.parent

                path.append(current.point)

                for point in path:
                    image[point[0], point[1]] = 128

                return image

            openset.remove(current)
            closedset.add(current)

            for node in self.neighbours(current, grid):
                if node in closedset:
                    continue

                if node in openset:
                    new_g = current.G + current.move_cost(node)

                    if node.G > new_g:
                        node.G = new_g
                        node.parent = current
                else:
                    node.G = current.G + current.move_cost(node)
                    node.H = self.manhattan(node, goal)
                    # Set the parent to our current item
                    node.parent = current
                    # Add it to the set
                    openset.add(node)
        # Throw an exception if there is no path
        raise ValueError('No Path Found')


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
        if 'dir_threshold' not in options:
            options['dir_threshold'] = 1.5
        if 'obstacle_weight' not in options:
            options['obstacle_weight'] = 120
        if 'max_obstacle_values' not in options:
            options['max_obstacle_values'] = 2
        if 'floor_weight' not in options:
            options['floor_weight'] = 10
        if 'max_floor_values' not in options:
            options['max_floor_values'] = 64

        self.options = options

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
        horizon = horizon / motion_image.shape[1]
        horizon_average = horizon.mean()
        left_limit = horizon[0]
        right_limit = horizon[-1]
        horizon_min = min(left_limit, right_limit)

        avg = motion_image.shape[1] / (2 * motion_image.shape[1])
        c = np.linspace(0, 1, horizon.shape[0]) - avg
        direction = (c * horizon ** 8).sum()

        self.last_obstacle_values.append(direction)

        if len(self.last_obstacle_values) > self.options['max_obstacle_values']:
            self.last_obstacle_values.pop(0)

        direction = np.mean(self.last_obstacle_values)

        if (motion_image.shape[0] - 20) < horizon_min:
            direction *= 2

        return [200 * expit(horizon_average), direction * np.pi / 180]

    def next_direction(self, motion_image: np.array) -> list:
        return self._flee(motion_image)
