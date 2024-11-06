import numpy as np
import pybullet as p
import pybullet_data
from typing import List, Tuple, Optional

from .telos_joints import (
    MOVING_JOINTS,
)
from .helper import load_yaml


class PyBullet:
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: str = "rgb_array",
        renderer: str = "Tiny",
    ) -> None:
        _config = load_yaml("src/pybullet_config.yaml")
        self.n_substeps = _config["pybullet"]["simulation"]["num_substeps"]

        self.motor_power = _config["pybullet"]["robot"]["power"]
        self.max_velocity = _config["pybullet"]["robot"]["max_velocity"]
        self.transmission_ratio = _config["pybullet"]["robot"]["transmission_ratio"]

        self.max_motor_power = self.motor_power

        self.render_mode = render_mode
        self.renderer = renderer

        if self.render_mode == "human":
            self.connection_mode = p.GUI
        elif self.render_mode == "rgb_array":
            if self.renderer == "OpenGL":
                self.connection_mode = p.GUI
            elif self.renderer == "Tiny":
                self.connection_mode = p.DIRECT

        self.physics_client = p.connect(self.connection_mode)
        p.setGravity(*_config["pybullet"]["simulation"]["gravity"])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def get_body_velocity(self, agent, type: int) -> List[float]:
        """
        Gets the body acceleration of the agent.
        :param type: type of velocity. 0 for linear, 1 for angular.
        :return: Body acceleration of the agent.
        """
        return p.getBaseVelocity(agent)[type]

    def get_orientation(self, agent) -> np.ndarray:
        """
        Gets the orientation of the agent.
        :return: Orientation of the agent.
        """
        return np.array(
            p.getEulerFromQuaternion(p.getBasePositionAndOrientation(agent)[1])[:3]
        )

    def get_pitch_angle(self, agent) -> float:
        """
        Gets the pitch angle of the agent.
        :return: Pitch angle of the agent.
        """
        return p.getEulerFromQuaternion(p.getBasePositionAndOrientation(agent)[1])[1]

    def get_roll_angle(self, agent) -> float:
        """
        Gets the roll angle of the agent.
        :return: Roll angle of the agent.
        """
        return p.getEulerFromQuaternion(p.getBasePositionAndOrientation(agent)[1])[0]

    def get_yaw_angle(self, agent) -> float:
        """
        Gets the yaw angle of the agent.
        :return: Yaw angle of the agent.
        """
        return p.getEulerFromQuaternion(p.getBasePositionAndOrientation(agent)[1])[2]

    def get_joint_state(
        self, agent, joint_id
    ) -> Tuple[float, float, List[float], float]:
        """
        Gets the joint state for the specified joint.
        :param joint_id: ID of the joint.
        :return: Joint state.
        """
        return p.getJointState(agent, joint_id)

    def get_joint_velocity(self, agent, joint_id) -> float:
        """
        Gets the joint velocity for the specified joint.
        :param joint_id: ID of the joint.
        :return: Joint velocity.
        """
        return p.getJointState(agent, joint_id)[3]

    def get_velocity_from_rotary(self, agent) -> List[float]:
        """
        Gets the velocity from rotary joints.
        :return: velocity from rotary joints.
        """
        velocity = [
            self.get_joint_velocity(agent, joint_id=joint) for joint in MOVING_JOINTS
        ]
        return velocity

    def get_all_info_from_agent(self, agent) -> Tuple[List[float], List[float]]:
        return p.getBasePositionAndOrientation(agent)

    def reset_joint_state(self, agent, joint_id, joint_angle) -> None:
        """
        Resets the joint state of the agent.
        :param joint_id: ID of the joint.
        :param joint_angle: Angle of the joint.
        """
        p.resetJointState(agent, joint_id, joint_angle)

    def get_center_of_mass(self, agent) -> List[float]:
        """
        Gets the center of mass of the agent.
        :return: Center of mass of the agent.
        """
        return p.getBasePositionAndOrientation(agent)[0]

    def control_joints(
        self,
        agent,
        joint_ids,
        target_positions,
        target_velocities,
        control_mode=p.POSITION_CONTROL,
    ) -> None:
        """
        Controls the joints of the agent.
        :param joint_ids: IDs of the joints.
        :param target_positions: Target positions.
        :param target_velocities: Target velocities.
        :param control_mode: Control mode.
        """
        p.setJointMotorControlArray(
            agent,
            joint_ids,
            controlMode=control_mode,
            targetPositions=target_positions,
            targetVelocities=target_velocities,
        )

    def control_joints_with_Kp(
        self,
        agent,
        joint_ids,
        target_positions,
        control_mode=p.POSITION_CONTROL,
        Kp=2.0,
    ) -> None:
        """
        Controls the joints of the agent with target velocities based on proportional control.
        :param joint_ids: IDs of the joints.
        :param target_positions: Target positions.
        :param control_mode: Control mode.
        :param Kp: Proportional gain for velocity control.
        :param max_velocity: Maximum allowable velocity for stability.
        """

        current_positions = [self.get_joint_state(agent, jid)[0] for jid in joint_ids]

        target_velocities = np.clip(
            Kp * (np.array(target_positions) - np.array(current_positions)),
            -self.max_velocity,
            self.max_velocity,
        ).tolist()

        p.setJointMotorControlArray(
            agent,
            joint_ids,
            controlMode=control_mode,
            targetPositions=target_positions,
            targetVelocities=target_velocities,
            forces=[self.max_motor_power] * len(joint_ids),
        )

    def reset_joints_force(
        self, agent, joint_ids, control_mode=p.POSITION_CONTROL
    ) -> None:
        p.setJointMotorControlArray(
            agent,
            joint_ids,
            controlMode=control_mode,
            forces=np.zeros(len(joint_ids)),
        )

    def step_simulation(self) -> None:
        """
        Steps the simulation forward by one time step.
        """
        for _ in range(self.n_substeps):
            p.stepSimulation()

    def disconnect(self) -> None:
        """
        Disconnects from PyBullet.
        """
        p.disconnect(self.physics_client)

    def get_contact_points_with_ground(
        self, agent, ground_plane, zero_z=True
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Gets the contact points of the agent with the ground.
        :return: Boolean info and contact points of the agent with the ground.
        """
        contact_points = p.getContactPoints(agent, ground_plane)
        euc_contact_points = []
        for contact_point in contact_points:
            euc_contact_points.append(contact_point[5])
        if not euc_contact_points:
            return False, None
        euc_contact_points = np.array(euc_contact_points)
        if zero_z:
            euc_contact_points[:, 2] = 0
        return True, euc_contact_points

    def load_plane(self, plane_path: str = "plane.urdf") -> int:
        """
        Loads the ground plane.
        :param plane_path: Path to the plane URDF file. By default, it is "plane.urdf".
        """
        return p.loadURDF(plane_path)

    def reset_base_pos(self, agent, start_pos, cube_start_orientation) -> None:
        p.resetBasePositionAndOrientation(agent, start_pos, cube_start_orientation)

    def get_quaternion_from_euler(self, euler_angles) -> List[float]:
        """
        Gets the quaternion from Euler angles.
        :param euler_angles: Euler angles.
        :return: Quaternion from Euler angles.
        """
        return p.getQuaternionFromEuler(euler_angles)

    def load_agent(
        self,
        agent_path: str,
        start_pos: np.ndarray,
        start_orientation: np.ndarray,
        orientation_from_euler: bool = True,
    ) -> int:
        """
        Loads the agent.
        :param agent_path: Path to the agent URDF file.
        :param start_pos: Start position of the agent.
        :param start_orientation: Start orientation of the agent.
        :param orientation_from_euler: If True, the orientation is from Euler angles. Otherwise, it is from a quaternion.
        """
        if orientation_from_euler:
            start_orientation = p.getQuaternionFromEuler(start_orientation)
        return p.loadURDF(agent_path, start_pos, start_orientation)

    def close(self) -> None:
        """
        Closes the PyBullet environment.
        """
        p.disconnect()

    def draw_debug_plane(self, height) -> None:
        plane_visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[5, 5, 0.001],  # Large plane with a very small height
            rgbaColor=[1, 0, 0, 0.5],  # Set the color to red with some transparency
        )
        plane_position = [0, 0, 0.15]

        # Create the plane without collision
        p.createMultiBody(
            baseVisualShapeIndex=plane_visual_shape_id,
            baseCollisionShapeIndex=-1,  # No collision shape
            basePosition=plane_position,
        )

    def draw_debug_sphere(self, sphere_position) -> None:
        sphere_radius = 0.05
        sphere_visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=[0, 1, 0, 1]
        )

        p.createMultiBody(
            baseVisualShapeIndex=sphere_visual_shape_id,
            baseCollisionShapeIndex=-1,  # No collision shape
            basePosition=sphere_position,
        )

    def get_matrix_from_quaternion(self, quaternion) -> np.ndarray:
        """
        Gets the rotation matrix from a quaternion.
        :param quaternion: Quaternion.
        :return: Rotation matrix.
        """
        return p.getMatrixFromQuaternion(quaternion)

    def ray_test(self, ray_from, ray_to) -> Optional[List[float]]:
        """
        Ray test.
        :param ray_from: Ray from.
        :param ray_to: Ray to.
        :return: Boolean info and hit position.
        """
        return p.rayTest(ray_from, ray_to)

    def add_debug_line(
        self,
        ray_from,
        ray_to,
        lineColorRGB=[1, 0, 0],
        lineWidth=1,
        lifeTime=0.1,
    ) -> None:
        """
        Adds a debug line.
        :param ray_from: Ray from.
        :param ray_to: Ray to.
        :param lineColorRGB: Line color.
        :param lineWidth: Line width.
        :param lifeTime: Lifetime.
        """
        p.addUserDebugLine(
            ray_from,
            ray_to,
            lineColorRGB=lineColorRGB,
            lineWidth=lineWidth,
            lifeTime=lifeTime,
        )

    def get_num_joints(self, agent) -> int:
        """
        Gets the number of joints in the agent.
        :return: Number of joints in the agent.
        """
        return p.getNumJoints(agent)

    def get_link_state(self, agent, link_id) -> Tuple[List[float], List[float]]:
        """
        Gets the link state.
        :param link_id: ID of the link.
        :return: Link state.
        """
        return p.getLinkState(agent, link_id)

    def get_aabb(self, agent, link_id) -> Tuple[List[float], List[float]]:
        """
        Gets the axis-aligned bounding box (AABB).
        :param link_id: ID of the link.
        :return: AABB.
        """
        return p.getAABB(agent, link_id)
