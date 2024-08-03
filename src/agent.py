import os
import numpy as np

import pybullet as p
from utils.telos_joints import (
    DEFAULT_ANGLES,
    MOVING_JOINTS,
)
from utils.helper import load_yaml
from utils.PyBullet import PyBullet


class TelosAgent:
    def __init__(
        self,
        sim_engine: PyBullet,
    ) -> None:
        self.sim = sim_engine
        _config = load_yaml("pybullet_config.yaml")
        _current_dir = os.path.dirname(os.path.realpath(__file__))
        _urdf_root_path = _current_dir + _config["pybullet"]["robot"]["urdf_path"]

        self.default_angles = DEFAULT_ANGLES
        self.cube_start_orientation = self.sim.get_quaternion_from_euler(
            [*_config["pybullet"]["robot"]["start_orientation"]]
        )

        self.start_pos = [*_config["pybullet"]["robot"]["start_position"]]
        self.robot_agent = self.sim.load_agent(
            _urdf_root_path, self.start_pos, self.cube_start_orientation, False
        )

        self.reset_angles()

    def reset_angles(self):
        default_angles = self.default_angles.copy()
        for joint in range(16):
            self.sim.reset_joint_state(
                self.robot_agent,
                joint,
                default_angles.pop(0),
            )

    def reset(self):
        self.sim.reset_base_pos(
            self.robot_agent, self.start_pos, self.cube_start_orientation
        )
        self.reset_angles()

    def get_joints_velocities(self):
        return np.array(self.sim.get_velocity_from_rotary(self.robot_agent))

    def set_action(self, action):
        self.sim.control_joints(self.robot_agent, MOVING_JOINTS, action, np.zeros(12))

    def get_obs(self):
        """
        Gets the observation for the quadruped robot.
        :return: Observation for the quadruped robot as a list of shape (34,).
        """
        observation = []
        position, orientation = self.sim.get_all_info_from_agent(self.robot_agent)
        observation.extend(position)  # x, y, z coordinates
        observation.extend(orientation)  # x, y, z, w orientation

        for joint in MOVING_JOINTS:
            joint_state = self.sim.get_joint_state(self.robot_agent, joint)
            observation.extend(joint_state[:2])  # Joint angle and velocity

        base_velocity = self.sim.get_body_velocity(self.robot_agent, type=0)
        for vel in base_velocity:
            observation.append(vel)

        return np.array(observation, dtype=np.float32)
