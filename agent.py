# -*- coding: UTF-8 -*-
"""
@Project : w10 
@File    : agent.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 26/03/2022 15:21 
@Brief   : 
"""
from collections import deque

import numpy as np

from target import Position, Target


class PID:

    def __init__(self, p: float, i: float, d: float):
        """PID controller."""
        self.kp = p
        self.ki = i
        self.kd = d

    def output(self, error_p: float, error_i: float, error_d: float):
        return self.kp * error_p + self.ki * error_i + self.kd * error_d


class Agent:
    EPSILON = 1e-6  # A small value to truncate errors
    FOV = 25  # The field of view

    def __init__(self, speed: float, x: float, y: float, target: Target):
        """The agent class.

        Args:
            speed (int): The unit is [cm/s].
            x (float): The x position of the target relative to the agent.
            y (float): The y position of the target relative to the agent.
        """
        self.speed = speed
        self.position = Position(x, y)
        self._original_points = (x, y)
        self._target = target

        self.los = np.asarray([[target.position.x - self.position.x], [target.position.y - self.position.y]])
        self.head_vector = np.asarray([[0], [1]])
        self.initial_delay = 0.18
        self.delay = self.initial_delay

        # Errors, the errors used for constant bearing
        self.errors_p = deque()
        self.errors_i = deque()
        self.errors_d = deque()
        self.errors = deque()
        self.error_integrate = 0
        self.pre_error = 0

    def cal_error(self, target_position: Position) -> float:
        """Calculate the error between target and agent head.
        Args:
            target_position (Position): The position information of the target.

        Returns:
            theta error, i.e., the angle between the agent head vector and the LOS (line of sight) with unit of radian.
        """
        # Line of sight
        los = np.asarray([[target_position.x - self.position.x], [target_position.y - self.position.y]])
        los /= np.linalg.norm(los)
        theta_e = np.arccos(np.dot(self.head_vector.T, los))

        # Check the theta_error position
        if np.cross(self.head_vector.T, los.T) < 0:
            theta_e = -theta_e

        # Truncate the small value for theta_error
        if abs(theta_e.item()) < self.EPSILON:
            theta_e = 0

        return float(theta_e)

    def rotation_matrix(self, theta: float):
        """Calculate the rotation matrix i.e., [cos(beta), -sin(beta); sin(beta), cos(beta)]. The rotation matrix can be
        used to rotate the head vector to the corrected direction.

        Args:
            theta (float): The angle between the head vector and LOS with unit of radians.

        Returns:
            A rotation matrix
        """
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    def const_bearing(self, bearing_angle: float, target_position: Position, dt: float,
                      pid_controller: PID = PID(5, 0.5, 0.5)) -> Position:
        """Constant bearing. Using PID framework to construct a controller that would achieve geometric relationship
        of a simple pursuit and constant bearing in the scenario shown in the schematics, which has been achieved in
        the target class.

        Args:
            bearing_angle (float): Bearing angle, unit - [degree]
            target_position (Position): Position class of the target.
            dt (float): Time step.
            pid_controller (PID): PID controller.

        Returns:
            Position class of the agent.
        """
        bearing = np.radians(bearing_angle)
        # todo: try to specify this step.
        self.delay -= 0.01

        error = self.cal_error(target_position) - bearing

        # If - else statement is to fix the issue when the bot cannot see the target
        if abs(np.degrees(error)) <= self.FOV:
            self.error_integrate += error * dt
            error_derivative = (error - self.pre_error) / dt
            self.pre_error = error

            # Save errors
            self.errors_p.append(error)
            self.errors_i.append(self.error_integrate)
            self.errors_d.append(error_derivative)
            self.errors.append(error + bearing)  # pure error

            if self.delay < 0.0:
                theta = pid_controller.output(self.errors_p[0], self.errors_i[0], self.errors_d[0]) * dt
                # theta = pid_controller.output(self.errors_p[-1], self.errors_i[-1], self.errors_d[-1]) * dt
                rotation_matrix = self.rotation_matrix(theta.item())
                self.head_vector = np.dot(rotation_matrix, self.head_vector)

                # update position
                distance = self.speed * dt
                self.position.x += distance * self.head_vector[0, 0]
                self.position.y += distance * self.head_vector[1, 0]

                # clear cache
                self.errors_p.popleft()
                self.errors_i.popleft()
                self.errors_d.popleft()

        else:
            # print(np.degrees(error))
            rotation_matrix = self.rotation_matrix(- np.pi / 6)
            self.head_vector = np.dot(rotation_matrix, self.head_vector)
            # self.delay = self.initial_delay

        return self.position

    def proportional_navigation(self, target_position: Position, dt: float, pid_controller: PID) -> Position:
        self.delay -= 0.01
        pre_los = self.los
        error = self.cal_error(target_position)

        # If - else statement is to fix the issue when the bot cannot see the target
        if abs(error) <= self.FOV:
            self.errors.append(error)
            self.los = np.asarray([[target_position.x - self.position.x], [target_position.y - self.position.y]])
            los_diff = np.arccos(np.dot(pre_los.T, self.los) / (np.linalg.norm(pre_los) * np.linalg.norm(self.los)))

            if np.cross(pre_los.T, self.los.T) < 0:
                los_diff = -los_diff
            if float(abs(los_diff)) < self.EPSILON:
                los_diff = 0

            los_diff_vel = los_diff / dt

            if self.delay < 0.0:
                theta = pid_controller.output(0, 0, los_diff_vel) * dt
                rotation_matrix = self.rotation_matrix(float(theta))
                self.head_vector = np.dot(rotation_matrix, self.head_vector)

                distance = self.speed * dt
                self.position.x += distance * self.head_vector[0, 0]
                self.position.y += distance * self.head_vector[1, 0]
        else:
            # print(np.degrees(error))
            rotation_matrix = self.rotation_matrix(- np.pi / 6)
            self.head_vector = np.dot(rotation_matrix, self.head_vector)
            # self.delay = self.initial_delay

        return self.position

    def clear_cache(self):
        self.position.clear_cache(self._original_points)
        self.errors_p = deque()
        self.errors_i = deque()
        self.errors_d = deque()
        self.errors = deque()
        self.error_integrate = 0
        self.pre_error = 0
        self.delay = self.initial_delay
        self.los = np.asarray([[self._target.position.x - self.position.x], [self._target.position.y -
                                                                             self.position.y]])
        self.head_vector = np.asarray([[0], [1]])
