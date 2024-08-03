import math
import numpy as np

from utils.helper import load_yaml


class TelosTask:
    def __init__(self, agent, sim_engine):
        _config = load_yaml("pybullet_config.yaml")
        self.agent = agent
        self.sim = sim_engine
        self.pitch_bias = _config["task"]["pitch_bias"]
        self.goal_radius = _config["task"]["goal_radius"]
        self.fall_reward = _config["task"]["fall_reward"]
        self.max_angle_dip = _config["task"]["max_angle_dip"]
        self.fall_threshold = _config["task"]["fall_threshold"]
        self.dist_threshold = _config["task"]["distance_threshold"]
        self.smoothing_factor = _config["task"]["smoothing_factor"]
        self.forward_velocity_bias = _config["task"]["forward_velocity_bias"]
        self.angle_bounds = np.deg2rad([*_config["task"]["goal_angle_bounds"]])
        theta = np.random.uniform(*self.angle_bounds)
        self.goal = np.array(
            [self.goal_radius * np.cos(theta), self.goal_radius * np.sin(theta), 0],
            dtype=np.float32,
        )

    def reset(self, seed=None):
        theta = np.random.uniform(*self.angle_bounds)
        self.goal = np.array(
            [self.goal_radius * np.cos(theta), self.goal_radius * np.sin(theta), 0],
            dtype=np.float32,
        )

    def get_obs(self):
        return self.agent.get_obs()

    def get_achieved_goal(self):
        return self.agent.get_obs()

    def is_success(self, achieved_goal, desired_goal, info={}) -> bool:
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return np.array(distance < self.dist_threshold, dtype=bool)

    def is_terminated(self) -> bool:
        is_terminated = False
        is_terminated = (
            abs(self.sim.get_pitch_angle(self.agent.robot_agent)) > self.max_angle_dip
        )
        is_terminated = (
            abs(self.sim.get_roll_angle(self.agent.robot_agent)) > self.max_angle_dip
        )
        # For now, we are not considering the yaw angle because the goal might be in any direction
        is_terminated = self.agent.get_obs()[2] < self.fall_threshold
        return is_terminated

    def compute_reward(self, achieved_goal, desired_goal, info={}) -> float:
        healthy_reward = (
            0 if achieved_goal[2] > self.fall_threshold else self.fall_reward
        )
        distance_reward = -np.linalg.norm(achieved_goal - desired_goal)
        forward_velocity_reward = (
            self.sim.get_body_velocity(self.agent.robot_agent, 0)[0]
            * self.forward_velocity_bias
        )
        smoothing_reward = -self.smoothing_factor * np.sum(
            self.sim.get_velocity_from_rotary(self.agent.robot_agent)
        )
        pitch_reward = -self.pitch_bias * math.pow(
            self.sim.get_pitch_angle(self.agent.robot_agent), 2
        )
        return (
            healthy_reward
            + distance_reward
            + smoothing_reward
            + forward_velocity_reward
            + pitch_reward
        )
