import math
import numpy as np
import gymnasium as gym
from typing import Tuple, Optional


import src.utils.telos_joints as tj
from src.utils.helper import load_yaml
from src.utils.PyBullet import PyBullet


class StandingTelosTaskEnv(gym.Env):
    def __init__(
        self, task, agent, sim_engine: PyBullet, rotating_plane: bool = False
    ) -> None:
        _config = load_yaml("src/pybullet_config.yaml")
        self.task = task
        self.agent = agent
        self.sim = sim_engine
        self.plane = self.sim.load_plane()
        self.rotating_plane = rotating_plane
        if self.rotating_plane:
            self.plane_angle_bounds = math.radians(
                _config["standing_task"]["plane_angle_bounds"]
            )
        observation, _ = self.reset()
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(observation["agent"]),),
                    dtype=np.float32,
                )
            }
        )
        self.low_angles = tj.low_angles
        self.high_angles = tj.high_angles
        self.action_space = gym.spaces.Box(
            low=self.low_angles, high=self.high_angles, shape=(12,), dtype=np.float32
        )

    def reset_plane(self):
        theta = np.random.uniform(
            low=-self.plane_angle_bounds, high=self.plane_angle_bounds
        )
        orientation = self.sim.get_quaternion_from_euler([theta, theta, 0])
        self.sim.reset_base_pos(self.plane, [0, 0, 0], orientation)

    def _get_obs(self):
        return {"agent": self.agent.get_obs(self.plane)}

    def _get_info(self):
        agent_pos = self._get_obs()["agent"][0:3]
        goal_pos = self.task.goal
        return {
            "agent_position": agent_pos,
            "goal_position": goal_pos,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        self.task.reset()
        self.agent.reset()
        if self.rotating_plane:
            self.reset_plane()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:

        self.agent.set_action(action)
        self.sim.step_simulation()
        obs = self._get_obs()
        reward_obs = obs["agent"][7:31]
        end_effector_pos = obs["agent"][-12:]
        end_effector_zs = end_effector_pos[2::3]

        reward_obs = np.concatenate(
            (obs["agent"][0:3], reward_obs[::2], end_effector_zs)
        )
        terminated = self.task.is_terminated(obs["agent"][0:3])
        reward = self.task.compute_reward(reward_obs)
        info = self._get_info()
        return obs, reward, terminated, False, info

    def close(self):
        self.sim.close()

    def wait_for_contact(self):
        contact, contact_points = None, None
        while True:
            contact, contact_points = self.sim.get_contact_points_with_ground(
                self.agent.robot_agent, self.plane, zero_z=False
            )
            if contact and contact_points.shape[0] >= 4:
                break
            self.sim.step_simulation()

    def render(self):
        return


def make_standing_env(task, agent, sim):
    return StandingTelosTaskEnv(task, agent, sim)
