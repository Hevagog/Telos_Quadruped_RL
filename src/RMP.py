import numpy as np
import gymnasium as gym

from utils.stability_functions import target_task_map
from utils.telos_joints import MOVING_JOINTS


class RMP:
    def __init__(
        self, task_map, policy, metric, Kp, Kd, z_position_emphasis, pitch_emphasis
    ):
        self.task_map = task_map
        self.policy = policy
        self.metric = metric
        self.Kp = Kp
        self.Kd = Kd
        self.z_position_emphasis = z_position_emphasis
        self.pitch_emphasis = pitch_emphasis
        self.policy_shape = len(MOVING_JOINTS) + 4  # 4 for z_position, pitch, vx, vy

    def compute_forces(self, config):
        task_state = self.task_map(config)
        forces = self.policy(task_state, self.Kp, self.Kd, self.policy_shape)
        metric = self.metric(
            task_state, self.z_position_emphasis, self.pitch_emphasis, self.policy_shape
        )
        return forces, metric


class GlobalRMPPolicy:
    def __init__(self, rmps):
        self.rmps = rmps

    def compute_global_forces(self, config):
        cfg = config["agent"]
        total_force = np.array([])
        total_metric = np.array([])
        for rmp in self.rmps:
            forces, metric = rmp.compute_forces(cfg)
            if not total_force.size:
                total_force = np.zeros(len(forces))
                total_metric = np.zeros_like(metric)
            total_force += forces
            total_metric += metric
        return np.linalg.inv(total_metric).dot(total_force)


class RMPRewardWrapper(gym.Env):
    def __init__(self, env, rmp_policy):
        super(RMPRewardWrapper, self).__init__()
        self.env = env
        self.rmp_policy = rmp_policy

        # Ensure observation and action space match the wrapped environment
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
        observation = self.env.reset(seed, options)
        return observation

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        rmp_forces = self.rmp_policy.compute_global_forces(obs)

        # Modify the reward based on the alignment with RMP forces
        reward += rmp_forces[0] + rmp_forces[1] + rmp_forces[-2] + rmp_forces[-1]
        return obs, reward, done, truncated, info

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()
