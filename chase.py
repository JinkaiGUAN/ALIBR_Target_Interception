# -*- coding: UTF-8 -*-
"""
@Project : w10 
@File    : chase.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 26/03/2022 19:30 
@Brief   : 
"""
import copy
import typing as t
from math import floor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
import scipy.io as scio

from agent import Agent, PID
from target import Position, Target

config = {
    "font.family": 'Times New Roman',
    "font.size": 24,
    "mathtext.fontset": 'stix',
    "axes.titlesize": 26
}
rcParams.update(config)


class PositionContainer:
    def __init__(self):
        self._x = []
        self._y = []

    def push(self, position: Position):
        self._x.append(copy.copy(position.x))
        self._y.append(copy.copy(position.y))

    def get(self, idx: int) -> t.Tuple:
        return self._x[idx], self._y[idx]

    @property
    def x(self) -> t.List:
        return self._x

    @property
    def y(self) -> t.List:
        return self._y


class Chase:
    """Main class to achieve the chase game."""
    X_TERMINAL = 330
    MEET_DISTANCE = 6
    ANGLE_TERMINAL = 70

    def __init__(self, target: Target, agent: Agent, time_step: float = 0.01):
        self.target = target
        self.agent = agent
        self.time_step = time_step

        self.is_first_meet = False
        self.first_meet_time = 0

        self.Kps = np.linspace(start=0, stop=30, num=30, endpoint=False)
        self.Kis = np.linspace(0, 30, num=30, endpoint=False)
        self.Kds = np.linspace(0, 1, num=20, endpoint=False)

    def plot_pursuit(self, isSine: bool = True, isConstantBearing: bool = True):

        # if isConstantBearing:
        PID_controller_bearing = PID(5, 0.4, 0.4)
        # PID_controller_bearing = PID(5, 17, 0.25)
        # else:
        PID_controller_prop = PID(0, 0, 1.2)

        fig_trajectory = plt.figure(figsize=(15, 10))  # used to plot the trajectory
        ax_trajectory = fig_trajectory.add_subplot(1, 1, 1)
        fig_distance = plt.figure(figsize=(15, 10))
        ax_distance = fig_distance.add_subplot(1, 1, 1)
        fig_error = plt.figure(figsize=(20, 10))  # used for error histogram
        fig_error.subplots_adjust(wspace=0.25, hspace=0.35)

        is_once = True  # only plot once
        bearing_angles = [0, 15, 30, 45, 60, 90]
        for i, angle in enumerate(bearing_angles):
            print("{:-^40}".format("Angle " + str(angle)))

            # recode the position of the agent and target
            target_positions = PositionContainer()
            agent_positions = PositionContainer()
            distances = []

            # Clear cache for target and agent object, including the x and y position information
            self.target.clear_cache()
            self.agent.clear_cache()
            self.is_first_meet = False
            self.first_meet_time = 0

            time_count = 0
            legend_label = ""  # Gain the legend label
            error_stop = 0  # How many steps the agent meet the target
            while self.target.position.x <= self.X_TERMINAL:
                # save the agent and target position information
                target_positions.push(self.target.position)
                agent_positions.push(self.agent.position)

                # Calculate the distance between target and agent
                distance = np.sqrt((self.target.position.x - self.agent.position.x) ** 2 + (self.target.position.y -
                                                                                            self.agent.position.y) ** 2)
                distances.append(distance)
                if distance <= self.MEET_DISTANCE and not self.is_first_meet:
                    self.is_first_meet = True
                    self.first_meet_time = (time_count + 1) * self.time_step
                    error_stop = time_count + 1
                    # label the position that first time meet
                    # ax_trajectory.scatter(self.target.position.x, self.target.position.y, marker="*", color="black")

                # update the position of target and agent
                if isSine:
                    # There is no necessary to explicitly return the position information since we can use the position
                    # class to gain them
                    self.target.move_sine(self.time_step)
                else:
                    self.target.move_line(self.time_step)

                if isConstantBearing:
                    if angle <= self.ANGLE_TERMINAL:
                        self.agent.const_bearing(angle, self.target.position, self.time_step, PID_controller_bearing)
                        legend_label = "Bearing Angle - {}$\degree$".format(angle)
                    else:
                        self.agent.proportional_navigation(self.target.position, self.time_step, PID_controller_prop)
                        legend_label = "Proportional Navigation"
                else:
                    self.agent.proportional_navigation(self.target.position, self.time_step, PID_controller_prop)
                    legend_label = "Proportional Navigation"

                # Update time
                time_count += 1

            # Log the information
            if not self.is_first_meet:
                print("Did not meet the target!")
            if self.first_meet_time <= self.agent.delay:
                self.first_meet_time = self.time_step * time_count
            print(f"Mean distance is {np.round(np.mean(distances), 2)} cm! The first time is {self.first_meet_time}.")

            # plot the target and agent position
            if is_once:
                ax_trajectory.plot(target_positions.x, target_positions.y, marker=".", markevery=20,
                                   markersize=20, linewidth=3, alpha=0.7, label="Target Position")
                is_once = False

            # Check whether both objects meet or not
            error_stop = -1 if error_stop == 0 else error_stop
            # using the handle to gain the color
            line = ax_trajectory.plot(agent_positions.x[:error_stop], agent_positions.y[:error_stop],
                                      marker=".", markevery=20, markersize=20, linewidth=3, alpha=0.7, label=legend_label)
            # Plot the meeting position
            if self.is_first_meet:
                ax_trajectory.scatter(target_positions.get(error_stop)[0], target_positions.get(error_stop)[1],
                                      marker="*",
                                      color=line[0].get_color(), s=300)

            ax_distance.plot(distances[:error_stop], label=legend_label, linewidth=3, alpha=0.7)
            # Plot the error bar
            ax_error_hist = fig_error.add_subplot(2, 3, i + 1)
            ax_error_hist.hist(np.degrees(list(self.agent.errors)[:error_stop]), bins=40, facecolor=line[0].get_color(),
                               edgecolor="black")
            ax_error_hist.set_title(legend_label)
            ax_error_hist.set_xlabel("Angle [$^\circ$]")
            ax_error_hist.set_ylabel("Occurring times [-]")

        ax_trajectory.set_xlabel("X distance [cm]")
        ax_trajectory.set_ylabel("Y distance [cm]")
        trajectory_title = "Trajectory of the target and agent for various controllers under sinusoidal target " \
                           "trajectory"
        trajectory_title = trajectory_title.replace("sinusoidal", "linear") if not isSine else trajectory_title
        ax_trajectory.set_title(trajectory_title)
        ax_trajectory.legend()
        ax_distance.set_xticklabels([floor(i) for i in ax_distance.get_xticks() * 10])
        ax_distance.set_xlabel("Iteration time [ms]")
        ax_distance.set_ylabel("Value of LoS [cm]")
        distance_title = "LoS between the target and agent for sinusoidal target route"
        distance_title = distance_title.replace("sinusoidal", "linear") if not isSine else distance_title
        ax_distance.set_title(distance_title)
        ax_distance.legend()

        trajectory_fig_name = "trajectory_sine.png" if isSine else "trajectory_linear.png"
        distance_fig_name = "distance_sine.png" if isSine else "distance_linear.png"
        error_hist_name = "error_sine.png" if isSine else "error_linear.png"
        fig_trajectory.savefig(f"./figures/{trajectory_fig_name}", format="png", bbox_inches="tight", dpi=600)
        fig_distance.savefig(f"./figures/{distance_fig_name}", format="png", bbox_inches="tight", dpi=600)
        fig_error.savefig(f"./figures/{error_hist_name}", format="png", bbox_inches="tight", dpi=600)
        plt.show()

    def plot_pid_tuning(self, isSine: bool = True, isConstantBearing: bool = True):
        """Doing the PID tuning to gain better results.

        Returns:

        """
        angle = 15  # bearing angle

        # tuning_map_data = np.zeros((self.Kps.size, self.Kis.size, self.Kds.size))
        tuning_map_data = np.zeros((self.Kps.size, self.Kis.size))
        # tuning_map_df = pd.DataFrame(columns=['P', 'I', 'D', 'Time', 'Distance'])

        for p_idx, kp in enumerate(self.Kps):
            for i_idx, ki in enumerate(self.Kis):
                # for d_idx, kd in enumerate(self.Kds):
                PID_controller_bearing = PID(kp, ki, 0)
                PID_controller_prop = PID(0, 0, 1.2)

                # Clear cache for target and agent object, including the x and y position information
                self.target.clear_cache()
                self.agent.clear_cache()
                self.is_first_meet = False
                self.first_meet_time = 0

                distances = []  # Stores the distance for one simulation
                time_count = 0
                error_stop = 0  # How many steps the agent meet the target, which is used to calculate the time
                while self.target.position.x <= self.X_TERMINAL:
                    # Calculate the distance between target and agent
                    distance = np.sqrt(
                        (self.target.position.x - self.agent.position.x) ** 2 + (self.target.position.y -
                                                                                 self.agent.position.y) ** 2)
                    distances.append(distance)
                    if distance <= self.MEET_DISTANCE and not self.is_first_meet:
                        self.is_first_meet = True
                        self.first_meet_time = (time_count + 1) * self.time_step
                        error_stop = time_count + 1

                    # update the position of target and agent
                    if isSine:
                        # There is no necessary to explicitly return the position information since we can use the
                        # position class to gain them
                        self.target.move_sine(self.time_step)
                    else:
                        self.target.move_line(self.time_step)

                    if isConstantBearing:
                        if angle <= self.ANGLE_TERMINAL:
                            self.agent.const_bearing(angle, self.target.position, self.time_step,
                                                     PID_controller_bearing)
                        else:
                            self.agent.proportional_navigation(self.target.position, self.time_step,
                                                               PID_controller_prop)
                    else:
                        self.agent.proportional_navigation(self.target.position, self.time_step,
                                                           PID_controller_prop)

                    time_count += 1

                if self.first_meet_time <= self.agent.delay:
                    self.first_meet_time = self.time_step * time_count

                # print(f"{error_stop}, {np.mean(distances)}")
                tuning_map_data[p_idx, i_idx] = np.mean(distances) + self.first_meet_time
                # tuning_map_df.append(
                #     {'P': kp, 'I': ki, 'D': kd, 'Time': self.first_meet_time, 'Distance': np.mean(distances)},
                #     ignore_index=True)

                if tuning_map_data[p_idx, i_idx] < 0.01:
                    print("Get")

        plt.imshow(tuning_map_data)
        tuning_map = tuning_map_data
        idxes = np.where(tuning_map == np.min(tuning_map))
        print(f"Kp: "
              f"{self.Kis[idxes[0].item()]},  Ki: {self.Kis[idxes[0].item()]}, minDis: {np.min(tuning_map)}")
        plt.colorbar()
        plt.title(f"Constant Kp: {self.Kis[idxes[0].item()]},  Ki: {np.round(self.Kis[idxes[0].item()], 2)},"
                  f" minDis: {np.min(tuning_map)}")
        # plt.xticks(labels=self.Kds[::5])
        # plt.yticks(labels=self.Kis[::5])

        plt.show()
        # break

        # tuning_map_df.to_csv("tuning_map_1.csv")
        np.save("tuning_map_data_1.npy", tuning_map_data)

    def plot_tuning_map(self):
        tuning_map = np.load("tuning_map_data_1.npy")

        # reverse the turning map
        i = 0
        while i < self.Kps.size / 2:
            temp = tuning_map[i, :]
            tuning_map[i, :] = tuning_map[self.Kps.size - 1 - i, :]
            tuning_map[self.Kps.size - 1 - i, :] = temp

            i += 1

        fig = plt.figure(figsize=(10, 10))
        data = tuning_map
        extent = np.min(self.Kds), np.max(self.Kds), np.min(self.Kis), np.max(self.Kis)
        plt.imshow(data)
        idxes = np.where(data == np.min(data))
        plt.colorbar()
        plt.title("PI Tuning Map", fontsize=32)
        for idx0, idx1 in zip(idxes[0], idxes[1]):
            print(f"Constant Kp: {self.Kps[idx0]},  Ki: {np.round(self.Kis[idx1], 2)}, "
                  f" minDis: {np.round(np.min(tuning_map), 2)}")
        plt.xlabel("$K_i$", fontsize=28)
        plt.ylabel("$K_p$", fontsize=28)
        # plt.axis('equal')

        # plt.show()

        ax = plt.gca()
        ax.invert_yaxis()
        ax.set_xticks(np.arange(-.5, 29.5, 1))
        ax.set_yticks(np.arange(-.5, 29.5, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks(np.arange(0, 30, 3), minor=True)
        ax.set_xticklabels([np.round(i, 2) for i in self.Kps[::3]], minor=True)
        ax.set_yticks(np.arange(0, 30, 3), minor=True)
        ax.set_yticklabels(self.Kis[::3], minor=True)
        ax.grid(color='k', linestyle='-', linewidth=2)

        plt.savefig("./figures/pi_tuning_map.png", format="png", bbox_inches="tight", dpi=600)
        plt.show()

    def plot_delay(self, isSine: bool = True, isConstantBearing: bool = True, is_once: bool = True):

        angle = 15

        PID_controller_bearing = PID(5, 0.4, 0.4)
        PID_controller_prop = PID(0, 0, 1.2)

        fig_trajectory = plt.figure(figsize=(15, 10))  # used to plot the trajectory
        ax_trajectory = fig_trajectory.add_subplot(1, 1, 1)

        delays = [0, 0.25, 0, 8.3]
        prop_index = 2  # index where the prop navi should be used
        for i, delay in enumerate(delays):
            print("{:-^40}".format(" Delay " + str(delay)) + " ")

            # recode the position of the agent and target
            target_positions = PositionContainer()
            agent_positions = PositionContainer()
            distances = []

            # Clear cache for target and agent object, including the x and y position information
            self.target.clear_cache()
            self.agent.clear_cache()
            self.is_first_meet = False
            self.first_meet_time = 0
            # Add delay
            self.agent.initial_delay = delay
            self.agent.delay = delay

            time_count = 0
            legend_label = ""  # Gain the legend label
            error_stop = 0  # How many steps the agent meet the target
            while self.target.position.x <= self.X_TERMINAL:
                # save the agent and target position information
                target_positions.push(self.target.position)
                agent_positions.push(self.agent.position)

                # Calculate the distance between target and agent
                distance = np.sqrt((self.target.position.x - self.agent.position.x) ** 2 + (self.target.position.y -
                                                                                            self.agent.position.y) ** 2)
                distances.append(distance)
                if distance <= self.MEET_DISTANCE and not self.is_first_meet:
                    self.is_first_meet = True
                    self.first_meet_time = (time_count + 1) * self.time_step
                    error_stop = time_count + 1
                    # label the position that first time meet
                    # ax_trajectory.scatter(self.target.position.x, self.target.position.y, marker="*", color="black")

                # update the position of target and agent
                if isSine:
                    # There is no necessary to explicitly return the position information since we can use the position
                    # class to gain them
                    self.target.move_sine(self.time_step)
                else:
                    self.target.move_line(self.time_step)

                if i < prop_index:
                    self.agent.const_bearing(angle, self.target.position, self.time_step, PID_controller_bearing)
                    legend_label = "Constant Bearing Delay - {}$s$".format(delay)
                else:
                    self.agent.proportional_navigation(self.target.position, self.time_step, PID_controller_prop)
                    legend_label = "Proportional Navigation Delay - {}$s$".format(delay)

                # Update time
                time_count += 1

            # Log the information
            if not self.is_first_meet:
                print("Did not meet the target!")
            if self.first_meet_time <= self.agent.delay:
                self.first_meet_time = self.time_step * time_count
            print(
                f"Mean distance is {np.round(np.mean(distances), 2)} cm! The first time is {self.first_meet_time}.")

            # plot the target and agent position
            if is_once:
                ax_trajectory.plot(target_positions.x, target_positions.y, marker=".", markevery=20,
                                   markersize=20, label="Target Position", linewidth=3)
                is_once = False

            # Check whether both objects meet or not
            error_stop = -1 if error_stop == 0 else error_stop
            # using the handle to gain the color
            line = ax_trajectory.plot(agent_positions.x, agent_positions.y,
                                      marker=".", markevery=20, markersize=20, label=legend_label, linewidth=3,
                                      alpha=0.7)
            # Plot the meeting position
            if self.is_first_meet:
                ax_trajectory.scatter(target_positions.get(error_stop)[0], target_positions.get(error_stop)[1],
                                      marker="*", color=line[0].get_color(), s=300)

        ax_trajectory.set_xlabel("X distance [cm]")
        ax_trajectory.set_ylabel("Y distance [cm]")
        trajectory_title = "Trajectory of the target and agent for various controllers under sinusoidal target " \
                           "trajectory"
        trajectory_title = trajectory_title.replace("sinusoidal", "linear") if not isSine else trajectory_title
        ax_trajectory.set_title(trajectory_title)
        ax_trajectory.legend()

        fig_trajectory.savefig(f"./figures/delay_1.png", format="png", bbox_inches="tight", dpi=600)
        plt.show()


if __name__ == "__main__":
    # Kp = 5, Ki = 17 Kd = 0.25

    target = Target(75, 0, 120)
    agent = Agent(85, 80, 0, target)
    chase = Chase(target, agent)
    chase.plot_pursuit(isSine=False, isConstantBearing=True)
    # chase.plot_pid_tuning(isSine=True, isConstantBearing=True)
    # chase.plot_tuning_map()
    # chase.plot_delay(isSine=True, isConstantBearing=True)

