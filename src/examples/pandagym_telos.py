import os
import numpy as np
import math
from gymnasium import spaces
from panda_gym.pybullet import PyBullet
import time

from panda_gym.envs.core import PyBulletRobot

from utils.telos_joints import TelosJoints as tj, KNEE_ANGLE, THIGH_HIP_ANGLE, HIP_ANGLE


class TelosEnv(PyBulletRobot):
    _current_dir = os.path.dirname(os.path.realpath(__file__))
    _urdf_root_path = _current_dir + "/urdf"
    _urdf_robot_path = _urdf_root_path + "/tt.urdf"

    def __init__(self, sim):
        action_dim = 12
        action_space = spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        self.default_angles = [0, HIP_ANGLE, THIGH_HIP_ANGLE, KNEE_ANGLE] * 4
        base_orientation = (0.7071067811865475, 0.0, 0.0, 0.7071067811865476)
        super().__init__(
            sim,
            body_name="telos",
            file_name=self._urdf_robot_path,
            base_position=[0, 0, 0.5],
            action_space=action_space,
            base_orientation=base_orientation,
            joint_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            joint_forces=[0] * 16,  # in Nm
        )

    def set_action(self, action):
        self.control_joints(action)

    def get_obs(self):
        angles = np.empty(12)
        for i in range(12):
            angles[i] = self.get_joint_angle(i)
        return np.array(angles)

    def reset(self) -> None:
        neutral_angles = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.set_joint_angles(neutral_angles)


##################################

if __name__ == "__main__":
    sim = PyBullet(
        render_mode="human", background_color=[0, 0, 0], gravity=[0, 0, -9.8]
    )
    robot = TelosEnv(sim)

    for _ in range(100):
        for angle in range(-45, 46):
            angles = np.array([math.radians(angle)] * 16)
            angles = robot.default_angles + angles
            # indices = [0, 4, 8, 12]
            # angles[indices] = 0
            robot.set_action(angles)
            # quadruped_agent.step_simulation()
            sim.step()
            time.sleep(0.01)
        for angle in range(45, -46, -1):
            angles = np.array([math.radians(angle)] * 16)
            angles = robot.default_angles + angles
            # indices = [0, 4, 8, 12]
            # angles[indices] = 0
            robot.set_action(angles)
            sim.step()
            time.sleep(0.01)

    # for _ in range(50):
    #     robot.set_action(np.array([math.radians(90)] * 16))
    #     sleep(0.2)
    #     sim.step()
