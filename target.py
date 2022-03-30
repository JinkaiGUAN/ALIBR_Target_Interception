# -*- coding: UTF-8 -*-
"""
@Project : w10 
@File    : target.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 26/03/2022 14:28 
@Brief   : 
"""
import numpy as np
import typing as t


class Position:
    def __init__(self, x: float, y: float):
        """Cartesian position information."""
        self._x = x
        self._y = y

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, val: float):
        self._x = val

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, val: float):
        self._y = val

    def clear_cache(self, original_points: t.Tuple):
        """Set the x and y point to original points"""
        self._x = original_points[0]
        self._y = original_points[1]


class Target:
    def __init__(self, speed: int, x: float = 0, y: float = 120):
        """This is the class to build the movement of the target!

        Args:
            speed (int): The unit is [cm/s].
            x (float): The x position of the target relative to the agent.
            y (float): The y position of the target relative to the agent.
        """

        self.speed = speed
        self.position = Position(x, y)
        self._original_points = (x, y)

    def move_line(self, dt: float) -> Position:
        """Simulate the situation when the target moves along the line.

        Args:
           dt (float): Time step.

        Returns:
             x and y position information.

        """
        self.position.x += self.speed * dt
        return self.position

    def move_sine(self, dt: float) -> Position:
        """Simulate the situation when the target moves according to a sine function.

        Args:
           dt (float): Time step.

        Notes:
            1. The sine wave mode follows the equation y = A sin(w x). Thus, the speed y' = Aw cos(wx).
            2. The moving distance along y-axis is dy = Aw cos(wx) dx.

        Returns:
             x and y position information.
        """
        # Basic parameters
        A = 40  # [cm]
        T = 60
        omega = 2 * np.pi / T

        vel_y = A * omega * np.cos(omega * self.position.x)
        step_size = self.speed * dt  # the distance that the target can move
        # split the distance along x abd y axes
        dx = np.sqrt(step_size ** 2 / (1 + vel_y ** 2))
        dy = vel_y * dx

        # Update position information
        self.position.x += dx
        self.position.y += dy

        return self.position

    def clear_cache(self):
        self.position.clear_cache(self._original_points)





