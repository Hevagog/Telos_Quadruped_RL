from src.agent import TelosAgent
from src.standing_task.standing_task import StandingTelosTask
from src.standing_task.standing_environment import (
    StandingTelosTaskEnv,
    make_standing_env,
)
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from src.utils.PyBullet import PyBullet
from sb3_contrib import TQC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_checker import check_env
import torch as th
import numpy as np


if __name__ == "__main__":
    pb = PyBullet(render_mode="rgb_array", renderer="Tiny")
    telos_agent = TelosAgent(pb)
    telos_task = StandingTelosTask(agent=telos_agent, sim_engine=pb)
    telos_env = StandingTelosTaskEnv(task=telos_task, agent=telos_agent, sim_engine=pb)

    n_steps, n_envs = 8192, 32

    env = make_vec_env(
        make_standing_env,
        n_envs=n_envs,
        env_kwargs={"task": telos_task, "agent": telos_agent, "sim": pb},
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    name = "ppo_sensing_standing_00"

    policy_ppo = {
        "net_arch": dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        "activation_fn": th.nn.ReLU,
    }

    checkpoint_callback = CheckpointCallback(
        save_freq=int(5_000_000 / n_envs),  # every 5_000_000 steps
        save_path="./stand_checkpoint/",
        name_prefix=name,
        save_vecnormalize=True,
    )

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
        tensorboard_log="./tensorboard/standppoTelos/",
    )

    try:
        model_ppo.learn(30_000_000, callback=[checkpoint_callback])
        model_ppo.save("ppo_standing_6")
    except KeyboardInterrupt:
        model_ppo.save("ppo_standing_6")
