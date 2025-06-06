import os
import numpy as np
from typing import List, Tuple

import pybullet as p
from src.utils.telos_joints import (
    DEFAULT_ANGLES,
    MOVING_JOINTS,
    DEFAULT_MOVING_ANGLES,
    HIP_IDX,
    TIP_IDX,
)
from src.utils.helper import load_yaml
from src.utils.PyBullet import PyBullet


class TelosAgent:
    def __init__(
        self,
        sim_engine: PyBullet,
    ) -> None:
        self.sim = sim_engine
        _config = load_yaml("src/pybullet_config.yaml")
        _current_dir = os.path.dirname(os.path.realpath(__file__))
        _urdf_root_path = _current_dir + _config["pybullet"]["robot"]["urdf_path"]

        self.default_angles = DEFAULT_ANGLES
        self.default_moving_angles = DEFAULT_MOVING_ANGLES

        self.liftoff_height = _config["pybullet"]["robot"]["liftoff_height"]
        self.start_orientation = _config["pybullet"]["robot"]["start_orientation"]
        self.cube_start_orientation = self.sim.get_quaternion_from_euler(
            [*self.start_orientation]
        )

        self.start_pos = [*_config["pybullet"]["robot"]["start_position"]]
        self.robot_agent = self.sim.load_agent(
            _urdf_root_path, self.start_pos, self.cube_start_orientation, False
        )

        self.reset_angles()

        ## leg sensing also static for now
        self._leg_tip_indices = TIP_IDX

        ## For now static values to test
        self.max_ray_length = 2.0  # m
        self.ray_angle_rad_x = np.radians(15)  # rad
        self.ray_angle_rad_z = np.radians(5)

        self.ray_origins = [
            [0.2, 0.0, 0.05],  # front right
            [0.2, 0.0, -0.05],  # front left
            [-0.2, 0.0, 0.05],  # rear right
            [-0.2, 0.0, -0.05],  # rear left
        ]

    def reset_angles(self):
        default_angles = self.default_angles.copy()
        for joint in range(len(DEFAULT_ANGLES)):
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
        self.sim.reset_joints_force(self.robot_agent, range(len(DEFAULT_ANGLES)))

    def get_moving_joints_torques(self):
        return np.array(self.sim.get_moving_joints_torques(self.robot_agent))

    def set_action(self, action):
        self.sim.control_joints_with_Kp(
            self.robot_agent,
            MOVING_JOINTS,
            self.default_moving_angles
            + action,  # ACTION IS DEVIATION FROM DEFAULT ANGLE
        )
        # Forgive me for the following code
        self.sim.control_joints_with_Kp(
            self.robot_agent,
            HIP_IDX,
            [self.default_angles[i] for i in HIP_IDX],
        )

    def get_obs(self, plane_id=None):
        """
        Gets the observation for the quadruped robot.
        :param plane_id: The plane id of the environment.
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

        # Add ray distances
        observation.extend(self.get_ray_distances(position, orientation))
        if plane_id is not None:
            end_effector_pos = []
            # Get leg tip positions (this time correctly :))
            for end_leg_sensing_link in self._leg_tip_indices:
                points = self.sim.get_closest_point_to_plane(
                    self.robot_agent,
                    plane_id,
                    end_leg_sensing_link,
                    self.liftoff_height,
                )
                if points:
                    end_effector_pos.extend(points[0][5])
                else:
                    end_effector_pos.extend(-100 * np.ones(3))
            observation.extend(end_effector_pos)

        return np.array(observation, dtype=np.float32)

    def get_ray_endpoints(
        self, base_pos, base_orient
    ) -> List[Tuple[List[float], List[float]]]:
        """Calculate ray start and end points in world coordinates."""
        rot_matrix = self.sim.get_matrix_from_quaternion(base_orient)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        rays = []
        for origin in self.ray_origins:
            # Transform ray origin from local to world coordinates
            world_origin = np.array(base_pos) + rot_matrix.dot(origin)

            sign_x = 1 if origin[0] > 0 else -1
            sign_z = 1 if origin[2] > 0 else -1
            local_dir = np.array(
                [
                    sign_x * np.sin(self.ray_angle_rad_x),  # x component (forward tilt)
                    -np.cos(self.ray_angle_rad_x),  # y component (downward)
                    sign_z * np.sin(self.ray_angle_rad_z),  # z component (right tilt)
                ]
            )
            world_dir = rot_matrix.dot(local_dir)

            # Calculate ray end point
            ray_end = world_origin + world_dir * self.max_ray_length

            rays.append((world_origin.tolist(), ray_end.tolist()))

        return rays

    def get_ray_distances(self, position, orientation) -> List[float]:
        """
        Get distances from ray intersections.

        Returns:
            List of distances for each ray. Returns max_ray_length if no intersection.
        """
        rays = self.get_ray_endpoints(position, orientation)
        distances = []

        for ray_from, ray_to in rays:
            # Perform raytest
            result = self.sim.ray_test(ray_from, ray_to)[0]
            hit_fraction = result[2]

            # Calculate distance
            if hit_fraction < 1.0:  # If ray hits something
                distance = hit_fraction * self.max_ray_length
            else:
                distance = self.max_ray_length

            distances.append(distance)

        return distances

    def visualize_rays(self):
        """Visualize rays for debugging purposes."""
        rays = self.get_ray_endpoints()
        debug_lines = []

        for ray_from, ray_to in rays:
            line_id = p.addUserDebugLine(
                ray_from,
                ray_to,
                lineColorRGB=[1, 0, 0],  # Red color
                lineWidth=1,
                lifeTime=5,  # Short lifetime for dynamic updating
            )
            debug_lines.append(line_id)

        return debug_lines
