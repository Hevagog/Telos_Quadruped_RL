from sb3_contrib import TQC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_checker import check_env
import torch as th
import numpy as np

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from src.agent import TelosAgent
from src.walking_task.walking_task import WalkingTelosTask
from src.walking_task.walking_environment import (
    WalkingTelosTaskEnv,
    make_walking_env,
)
from src.utils.PyBullet import PyBullet


def curriculum_scheduler(progress, max_difficulty=1.0, start_difficulty=0.01) -> float:
    return start_difficulty + progress * (max_difficulty - start_difficulty)


class CurriculumCallback(BaseCallback):
    def __init__(self, vec_env, max_difficulty=1.0, start_timesteps=0, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.vec_env = vec_env  # Vectorized environment
        self.max_difficulty = max_difficulty
        self.start_timesteps = start_timesteps

    def _on_step(self):
        current_step = self.num_timesteps
        # Wait until after start_timesteps to apply curriculum
        if current_step < self.start_timesteps:
            return True

        total_timesteps = self.locals["total_timesteps"]
        progress = (current_step - self.start_timesteps) / (
            total_timesteps - self.start_timesteps
        )
        progress = min(1.0, max(0.0, progress))

        current_difficulty = curriculum_scheduler(progress, self.max_difficulty)

        for i in range(self.vec_env.num_envs):
            if hasattr(self.vec_env.envs[i], "set_difficulty"):
                self.vec_env.envs[i].set_difficulty(current_difficulty)

        if self.verbose > 0:
            print(f"Progress: {progress:.2f}, Difficulty: {current_difficulty:.4f}")

        return True


if __name__ == "__main__":
    pb = PyBullet(render_mode="rgb_array", renderer="Tiny")
    telos_agent = TelosAgent(pb)
    telos_task = WalkingTelosTask(agent=telos_agent, sim_engine=pb)
    telos_env = WalkingTelosTaskEnv(task=telos_task, agent=telos_agent, sim_engine=pb)
    n_steps, n_envs = 8192, 32
    env = make_vec_env(
        make_walking_env,
        n_envs=n_envs,
        env_kwargs={"task": telos_task, "agent": telos_agent, "sim": pb},
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    policy_ppo = {
        "net_arch": dict(pi=[512, 512, 256], vf=[512, 512, 256]),
        "activation_fn": th.nn.Softsign,
    }
    name = "n_ppo_sensing_walking_01"

    checkpoint_callback = CheckpointCallback(
        save_freq=int(5_000_000 / n_envs),  # every 5_000_000 steps
        save_path="./checkpoint/",
        name_prefix=name,
        save_vecnormalize=True,
    )

    curriculum_callback = CurriculumCallback(vec_env=env, verbose=0)

    model_ppo = PPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_ppo,
        verbose=1,
        learning_rate=0.0002,
        n_steps=n_steps,
        batch_size=n_steps * n_envs,
        n_epochs=5,
        gamma=0.9988,
        tensorboard_log="./tensorboard/n_ppoTelos/",
    )

    try:
        model_ppo.learn(
            120_000_000, callback=[checkpoint_callback, curriculum_callback]
        )
        model_ppo.save(name)
    except KeyboardInterrupt:
        model_ppo.save(name)
