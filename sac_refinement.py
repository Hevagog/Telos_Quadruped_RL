from src.agent import TelosAgent
from src.standing_task.standing_task import StandingTelosTask
from src.standing_task.standing_environment import (
    StandingTelosTaskEnv,
    make_standing_env,
)
from src.utils.PyBullet import PyBullet
from sb3_contrib import TQC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
import numpy as np


if __name__ == "__main__":
    pb = PyBullet(render_mode="rgb_array", renderer="Tiny")
    telos_agent = TelosAgent(pb)
    telos_task = StandingTelosTask(agent=telos_agent, sim_engine=pb)
    telos_env = StandingTelosTaskEnv(task=telos_task, agent=telos_agent, sim_engine=pb)

    env = make_vec_env(
        make_standing_env,
        n_envs=32,
        env_kwargs={"task": telos_task, "agent": telos_agent, "sim": pb},
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = SAC.load("sac_standing_6", env=env)

    checkpoint_callback = CheckpointCallback(
        save_freq=156250,  # every 5_000_000 steps
        save_path="./checkpoiner/",
        name_prefix="sac_standing_6_r2",
    )

    try:
        model.learn(30_000_000, callback=checkpoint_callback)
        model.save("sac_standing_6_r2")
    except KeyboardInterrupt:
        model.save("sac_standing_6_r2")
