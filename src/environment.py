import numpy as np
import gymnasium as gym

from typing import Optional, Tuple
import utils.telos_joints as tj
from utils.PyBullet import PyBullet


class TelosTaskEnv(gym.Env):
    def __init__(self, task, agent, sim_engine: PyBullet) -> None:
        self.task = task
        self.agent = agent
        self.sim = sim_engine
        self.sim.load_plane()
        observation, _ = self.reset()
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(observation["agent"]),),
                    dtype=np.float32,
                ),
                "target": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(observation["target"]),),
                    dtype=np.float32,
                ),
            }
        )
        self.low_angles = tj.low_angles
        self.high_angles = tj.high_angles
        self.action_space = gym.spaces.Box(
            low=self.low_angles, high=self.high_angles, shape=(12,), dtype=np.float32
        )

    def _get_obs(self):
        return {"agent": self.agent.get_obs(), "target": self.task.goal}

    def _get_info(self):
        agent_pos = self._get_obs()["agent"][0:3]
        return {
            "distance": np.linalg.norm(agent_pos - self.task.goal),
            "is_success": bool(self.task.is_success(agent_pos, self.task.goal)),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        self.task.reset()
        self.agent.reset()
        self._target_location = self.task.goal
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.agent.set_action(action)
        self.sim.step_simulation()
        obs = self._get_obs()
        reward = self.task.compute_reward(obs["agent"][0:3], self.task.goal)
        done = self.task.is_success(obs["agent"][0:3], self.task.goal)
        done = done or self.task.is_terminated()
        info = self._get_info()
        return obs, reward, done, False, info

    def close(self):
        self.sim.close()

    def render(self):
        return


def make_env(task, agent, sim):
    return TelosTaskEnv(task, agent, sim)
