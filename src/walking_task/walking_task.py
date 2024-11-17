import math
import time
import numpy as np

from src.utils.helper import load_yaml
from src.utils.orientation_cost_function import (
    orientation_cost,
)

## TBD: Check feasibility of using angular_velocity_cost in cost function
from src.utils.angular_velocity_cost_function import angular_velocity_cost


class WalkingTelosTask:
    def __init__(self, agent, sim_engine):
        _config = load_yaml("src/pybullet_config.yaml")
        self.agent = agent
        self.sim = sim_engine
        self.fall_reward = _config["task"]["fall_reward"]
        self.fall_threshold = _config["task"]["fall_threshold"]

        self.up_threshold = _config["walking_task"]["up_threshold"]
        self.max_pitch_angle = _config["walking_task"]["max_pitch_angle"]
        self.max_roll_angle = _config["walking_task"]["max_roll_angle"]
        self.max_yaw_angle = _config["walking_task"]["max_yaw_angle"]
        self.forward_motion_reward = _config["walking_task"]["forward_motion_reward"]
        self.torque_penalty = _config["walking_task"]["torque_penalty"]

        self.start_pitch, self.start_roll, self.start_yaw = [
            *_config["standing_task"]["initial_angles"]
        ]
        self.max_robot_angular_velocity = _config["pybullet"]["robot"][
            "max_robot_angular_velocity"
        ]
        self.start_time = time.time()

    def reset(self, seed=None):
        self.start_time = time.time()

    def get_obs(self):
        return self.agent.get_obs()

    def get_episode_time(self) -> float:
        return time.time() - self.start_time

    def is_terminated(self, agent_pos: np.ndarray) -> bool:
        roll_angle = abs(
            self.sim.get_roll_angle(self.agent.robot_agent) - self.start_roll
        )
        pitch_angle = abs(
            self.sim.get_pitch_angle(self.agent.robot_agent) - self.start_pitch
        )
        yaw_angle = abs(
            abs(self.sim.get_yaw_angle(self.agent.robot_agent)) - self.start_yaw
        )
        is_terminated = (
            # Tipping over
            pitch_angle > self.max_pitch_angle
            or self.max_roll_angle < roll_angle
            or self.max_yaw_angle < yaw_angle
            # Falling or Jumping
            or self.agent.get_obs()[2] < self.fall_threshold
            or self.up_threshold < self.agent.get_obs()[2]
            # Cap on angular velocity
            or max(abs(self.agent.get_joints_velocities()))
            > self.max_robot_angular_velocity
        )

        return is_terminated

    def compute_reward(
        self,
        achieved_goal,
        info={},
    ) -> float:

        # Reward the frontal motion
        forward_velocity = self.sim.get_body_velocity(self.agent.robot_agent, type=0)[
            0
        ]  # x velocity
        forward_reward = self.forward_motion_reward * max(0, forward_velocity)

        # Penalization of too high torque on the joints
        torques = self.sim.get_moving_joints_torques(self.agent.robot_agent)
        torque_penalty = (
            -1 * np.sum(torque**2 for torque in torques) * self.torque_penalty
        )

        # Orientation Penalty
        orientation = -1 * orientation_cost(
            current_orientation=achieved_goal[:2],
            desired_orientation=np.array([self.start_pitch, self.start_roll]),
        )

        # Penalize jumping and falling
        if (
            achieved_goal[2] < self.fall_threshold
            or self.up_threshold < achieved_goal[2]
        ):
            fall_reward = self.fall_reward

        reward = forward_reward + torque_penalty + orientation + fall_reward

        return reward
