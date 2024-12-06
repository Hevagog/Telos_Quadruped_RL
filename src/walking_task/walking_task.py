import math
import time
import numpy as np

from src.utils.helper import load_yaml
from src.utils.orientation_cost_function import (
    orientation_cost,
    unbounded_orientation_cost,
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
        self.optimal_z_height = _config["walking_task"]["optimal_z_height"]
        self.x_position_reward = _config["walking_task"]["x_position_reward"]
        self.max_robot_torque = _config["pybullet"]["robot"]["max_robot_torque"]
        self.x_backward_threshold = _config["walking_task"]["x_backward_threshold"]
        self.x_successful_threshold = _config["walking_task"]["max_x_position"]
        self.start_pitch, self.start_roll, self.start_yaw = [
            *_config["standing_task"]["initial_angles"]
        ]
        self.max_robot_angular_velocity = _config["pybullet"]["robot"][
            "max_robot_angular_velocity"
        ]
        self.start_time = time.time()

        # Measure x increment for the reward
        self.odl_x = 0

        # Curriculum learning
        self.current_difficulty = 0.005

    def reset(self, seed=None):
        self.start_time = time.time()

    def get_obs(self):
        return self.agent.get_obs()

    def set_difficulty(self, difficulty: float):
        self.current_difficulty = difficulty

    def get_episode_time(self) -> float:
        return time.time() - self.start_time

    def is_terminated(self, agent_pos: np.ndarray) -> bool:
        roll_angle = abs(
            self.sim.get_roll_angle(self.agent.robot_agent) - self.start_roll
        )
        # pitch_angle = abs(
        #     self.sim.get_pitch_angle(self.agent.robot_agent) - self.start_pitch
        # )
        yaw_angle = abs(
            abs(self.sim.get_yaw_angle(self.agent.robot_agent)) - abs(self.start_yaw)
        )
        is_terminated = (
            # Tipping over
            # pitch_angle > self.max_pitch_angle or
            self.max_roll_angle < roll_angle
            or self.x_backward_threshold > agent_pos[0]
            or self.max_yaw_angle < yaw_angle
            # Falling or Jumping
            or self.agent.get_obs()[2] < self.fall_threshold
            or self.up_threshold < self.agent.get_obs()[2]
            # Cap on max torque
            or max(abs(self.agent.get_moving_joints_torques())) > self.max_robot_torque
            or agent_pos[0] >= self.x_successful_threshold
        )

        return is_terminated

    def compute_reward(
        self,
        achieved_goal,
        info={},
    ) -> float:
        agent_velocity = self.sim.get_body_velocity(self.agent.robot_agent, type=0)

        # Reward the frontal motion
        forward_reward = agent_velocity[0] if agent_velocity[0] > 0 else 0
        if forward_reward > 0.5:
            forward_reward = forward_reward * self.forward_motion_reward
        else:
            forward_reward = forward_reward * 0.5

        # # Penalization of too high torque on the joints
        torques = self.sim.get_moving_joints_torques(self.agent.robot_agent)
        torque_penalty = (
            -1
            * np.sum(torque**2 for torque in torques)
            * self.torque_penalty
            * (self.current_difficulty * 100.0)
        )

        # # Orientation Penalty
        orientation = (
            2.0
            * self.current_difficulty
            * unbounded_orientation_cost(
                current_orientation=[
                    self.sim.get_pitch_angle(self.agent.robot_agent),
                    self.sim.get_roll_angle(self.agent.robot_agent),
                ],
                desired_orientation=np.array([self.start_pitch, self.start_roll]),
                lower_threshold_deg=5,
                weight_per_axis=np.array([2.0, 1.0]),
            )
        )

        # Reward the x position
        x_position_reward = (
            self.x_position_reward * (achieved_goal[0] - self.odl_x)
            if achieved_goal[0] > self.odl_x
            else 0
        )
        self.odl_x = achieved_goal[0]

        if np.sum(achieved_goal[-4:] < 0) > 2:
            forward_reward = forward_reward * (
                1 - 0.23 * np.sum(achieved_goal[-4:] < 0)
            )
            x_position_reward = x_position_reward * (
                1 - 0.23 * np.sum(achieved_goal[-4:] < 0)
            )

        # Penalize jumping and falling
        if info.get("is_terminated"):
            fall_reward = self.fall_reward
        else:
            fall_reward = 0

        if self.x_successful_threshold < achieved_goal[0]:
            success = 25
        else:
            success = 0

        reward = (
            forward_reward
            + torque_penalty
            + orientation
            + fall_reward
            + success
            + x_position_reward
        )

        return reward
