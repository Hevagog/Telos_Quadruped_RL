from sb3_contrib import TQC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_checker import check_env
import torch as th
import numpy as np

from stable_baselines3.common.callbacks import CheckpointCallback
from src.agent import TelosAgent
from src.walking_task.walking_task import WalkingTelosTask
from src.walking_task.walking_environment import (
    WalkingTelosTaskEnv,
    make_walking_env,
)
from src.utils.PyBullet import PyBullet


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

    name = "ppo_walking_14_r"

    checkpoint_callback = CheckpointCallback(
        save_freq=int(5_000_000 / n_envs),  # every 5_000_000 steps
        save_path="./checkpoint/",
        name_prefix=name,
    )

    model_ppo = PPO.load("checkpoint/ppo_walking_14_10000000_steps", env=env)

    try:
        model_ppo.learn(120_000_000, callback=checkpoint_callback)
        model_ppo.save(name)
    except KeyboardInterrupt:
        model_ppo.save(name)
