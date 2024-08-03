import math
import time
import numpy as np

from utils.helper import load_yaml


class StandingTelosTask:
    def __init__(self, agent, sim_engine):
        _config = load_yaml("pybullet_config.yaml")
        self.agent = agent
        self.sim = sim_engine
        self.pitch_bias = _config["task"]["pitch_bias"]
        self.fall_reward = _config["task"]["fall_reward"]
        self.fall_threshold = _config["task"]["fall_threshold"]
        self.up_threshold = _config["standing_task"]["up_threshold"]
        self.max_angle_dip = _config["standing_task"]["max_angle_dip"]
        self.time_emphasis = _config["standing_task"]["time_emphasis"]
        self.max_angle_dev = _config["standing_task"]["max_angle_dev"]
        self.time_threshold = _config["standing_task"]["time_threshold"]
        self.angle_dip_bias = _config["standing_task"]["angle_dip_bias"]
        self.smoothing_factor = _config["standing_task"]["smoothing_factor"]
        self.dist_threshold = _config["standing_task"]["distance_threshold"]
        self.angle_bounds = np.deg2rad([*_config["task"]["goal_angle_bounds"]])
        self.bad_position_penalty = _config["standing_task"]["bad_position_penalty"]
        self.start_pitch, self.start_roll, self.start_yaw = [
            *_config["standing_task"]["initial_angles"]
        ]
        self.max_robot_angular_velocity = _config["pybullet"]["robot"][
            "max_robot_angular_velocity"
        ]
        self.good_position_reward = _config["standing_task"]["good_position_reward"]
        self.action_smoothing_factor = _config["standing_task"][
            "action_smoothing_factor"
        ]
        self.agent_start_pos = np.array(
            [*_config["pybullet"]["robot"]["start_orientation"]]
        )
        self.goal = np.array(
            [*_config["standing_task"]["desired_position"]], dtype=np.float32
        )
        self.start_time = time.time()

    def reset(self, seed=None):
        self.start_time = time.time()

    def get_obs(self):
        return self.agent.get_obs()

    def get_episode_time(self) -> float:
        return time.time() - self.start_time

    def is_terminated(self) -> bool:
        is_terminated = (
            abs(self.sim.get_pitch_angle(self.agent.robot_agent)) > self.max_angle_dip
            or abs(self.sim.get_roll_angle(self.agent.robot_agent)) > self.max_angle_dip
            or abs(self.sim.get_yaw_angle(self.agent.robot_agent)) > self.max_angle_dip
            or self.get_episode_time() > self.time_threshold
            or self.agent.get_obs()[2] < self.fall_threshold
            or self.up_threshold < self.agent.get_obs()[2]
            or max(abs(self.agent.get_joints_velocities()))
            > self.max_robot_angular_velocity
        )

        return is_terminated

    def compute_reward(
        self,
        achieved_goal,
        info={},
    ) -> float:

        smoothing_reward = -self.smoothing_factor * np.sum(
            self.sim.get_velocity_from_rotary(self.agent.robot_agent)
        )

        pitch_reward = -self.angle_dip_bias * math.pow(
            self.sim.get_pitch_angle(self.agent.robot_agent) - self.start_pitch, 2
        )
        roll_reward = -self.angle_dip_bias * math.pow(
            self.sim.get_roll_angle(self.agent.robot_agent) - self.start_roll, 2
        )
        yaw_reward = -self.angle_dip_bias * math.pow(
            self.sim.get_yaw_angle(self.agent.robot_agent) - self.start_yaw, 2
        )

        time_reward = self.time_emphasis * self.get_episode_time()
        distance_reward = (
            self.good_position_reward
            if np.linalg.norm(achieved_goal - self.goal) < self.dist_threshold
            else -np.linalg.norm(achieved_goal - self.goal) * self.bad_position_penalty
        )

        return (
            smoothing_reward
            + pitch_reward
            + time_reward
            + distance_reward
            + yaw_reward
            + roll_reward
        )
