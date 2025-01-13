import math
import time
import numpy as np
from typing import Tuple

from src.utils.helper import load_yaml
from src.utils.orientation_cost_function import (
    orientation_cost,
    unbounded_orientation_cost,
)
from src.utils.angular_velocity_cost_function import angular_velocity_cost
from src.utils.telos_joints import low_angles, high_angles
from src.RMP.RMPControllers.RMP_standing import rmp_standing
from src.RMP.RMPControllers.RMP_constraint import RMP_joint_limits
from src.RMP.RMP import combine_rmp


class StandingTelosTask:
    def __init__(self, agent, sim_engine):
        _config = load_yaml("src/pybullet_config.yaml")
        self.agent = agent
        self.sim = sim_engine
        self.pitch_bias = _config["task"]["pitch_bias"]
        self.fall_reward = _config["task"]["fall_reward"]
        self.fall_threshold = _config["task"]["fall_threshold"]
        self.up_threshold = _config["standing_task"]["up_threshold"]
        self.max_angle_dip = _config["standing_task"]["max_angle_dip"]
        self.time_emphasis = _config["standing_task"]["time_emphasis"]
        self.time_threshold = _config["standing_task"]["time_threshold"]
        self.angle_dip_bias = _config["standing_task"]["angle_dip_bias"]
        self.torque_penalty = _config["standing_task"]["torque_penalty"]
        self.jump_penalty = _config["standing_task"]["jump_penalty"]
        self.max_robot_torque = _config["pybullet"]["robot"]["max_robot_torque"]
        self.angular_velocity_penalty = _config["standing_task"][
            "angular_velocity_penalty"
        ]
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
        self._agent_starting_orientation = self.agent.start_orientation
        self.alpha = _config["rmps"]["alpha"]
        self.beta = _config["rmps"]["beta"]
        self.force_vector_emphasis = _config["rmps"]["force_vector_empahsis"]
        self.start_time = time.time()
        self._in_goal_pos_start_time = None

    def reset(self, seed=None):
        self.start_time = time.time()
        self._in_goal_pos_start_time = None

    def get_obs(self):
        return self.agent.get_obs()

    def get_episode_time(self) -> float:
        return time.time() - self.start_time

    def get_in_goal_pos_time(self) -> float:
        return time.time() - self._in_goal_pos_start_time

    def is_terminated(self, agent_pos: np.ndarray) -> bool:
        roll_angle = abs(
            self.sim.get_roll_angle(self.agent.robot_agent) - self.start_roll
        )
        pitch_angle = abs(
            self.sim.get_pitch_angle(self.agent.robot_agent) - self.start_pitch
        )
        yaw_angle = abs(
            abs(self.sim.get_yaw_angle(self.agent.robot_agent)) - abs(self.start_yaw)
        )
        is_terminated = (
            # Tipping over
            pitch_angle > self.max_angle_dip
            or self.max_angle_dip < roll_angle
            or self.max_angle_dip < yaw_angle
            # Falling or Jumping
            or self.agent.get_obs()[2] < self.fall_threshold
            or self.up_threshold < self.agent.get_obs()[2]
            # Cap on max torque
            or max(abs(self.agent.get_moving_joints_torques())) > self.max_robot_torque
        )

        return is_terminated

    def compute_reward(
        self,
        achieved_goal,
        info={},
    ) -> float:

        position_penalty = np.abs(achieved_goal[:3] - self.goal)
        position_penalty = np.sum(position_penalty)

        num_legs_touching_ground = np.sum((achieved_goal[-4:] < -10))

        if (
            np.abs(achieved_goal[:3] - self.goal) < self.dist_threshold
        ).all() and num_legs_touching_ground >= 3:
            position_reward = self.good_position_reward - position_penalty
        else:
            position_reward = -1 * (abs(position_penalty * 10))
            # position_reward = (
            #     self.get_in_goal_pos_time()
            #     * (self.good_position_reward - position_penalty)
            #     if self.get_in_goal_pos_time() > 1
            #     else self.good_position_reward - position_penalty
            # )

        torques = self.sim.get_moving_joints_torques(self.agent.robot_agent)
        torque_penalty = (
            -1 * np.sum(torque**2 for torque in torques) * self.torque_penalty
        )

        orientation = unbounded_orientation_cost(
            current_orientation=[
                self.sim.get_pitch_angle(self.agent.robot_agent),
                self.sim.get_roll_angle(self.agent.robot_agent),
            ],
            desired_orientation=np.array([self.start_pitch, self.start_roll]),
            lower_threshold_deg=10,
            weight_per_axis=np.array([0.75, 0.5]),
        )

        if info.get("is_terminated"):
            fall_reward = self.fall_reward
        else:
            fall_reward = 0.1

        # Sum all components to get the final reward
        reward = position_reward + torque_penalty + orientation + fall_reward
        return reward

        # standing_f, _ = rmp_standing(
        #     q=achieved_goal[0:3],
        #     q_dot=self.agent.get_joints_velocities(),
        #     xg=self.goal,
        #     center_of_mass=self.sim.get_center_of_mass(self.agent.robot_agent),
        # )

        # standing_f = -math.pow(np.dot(self.force_vector_emphasis, standing_f[:3]), 2)
        # angles_rmp = achieved_goal[3:-4]
        # joint_limits_f, _ = RMP_joint_limits(
        #     q=angles_rmp,
        #     dq=self.agent.get_joints_velocities(),
        #     alpha=self.alpha,
        #     beta=self.beta,
        #     low_angles=low_angles,
        #     high_angles=high_angles,
        # )
        # joint_limits_f = -math.pow(np.linalg.norm(joint_limits_f), 2)

        # end_leg_touching_ground_reward = 0.0
        # for z in achieved_goal[-4:]:
        #     if abs(z) < 0.01:
        #         end_leg_touching_ground_reward += 1.5

        # rmp_reward = standing_f + joint_limits_f

        # if self._in_goal_pos_start_time is not None:
        #     time_reward = self.time_emphasis * self.get_in_goal_pos_time()
        #     distance_coeff_reward = 2.0
        # else:
        #     time_reward = -self.time_emphasis * self.get_episode_time()
        #     distance_coeff_reward = 0.5

        # distance_reward = (
        #     self.good_position_reward
        #     * end_leg_touching_ground_reward
        #     * distance_coeff_reward
        #     if np.linalg.norm(achieved_goal[:3] - self.goal) < self.dist_threshold
        #     else 0
        # )

        # if is_terminated:
        #     fall_reward = self.fall_reward
        # else:
        #     fall_reward = 0.0

        # return (
        #     rmp_reward
        #     + end_leg_touching_ground_reward
        #     + orientation_cost_reward
        #     + time_reward
        #     + distance_reward
        #     + smoothing_reward
        #     + fall_reward
        # )
