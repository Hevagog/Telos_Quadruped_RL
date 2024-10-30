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
import torch as th

if __name__ == "__main__":
    pb = PyBullet(render_mode="rgb_array", renderer="Tiny")
    telos_agent = TelosAgent(pb)
    telos_task = StandingTelosTask(agent=telos_agent, sim_engine=pb)
    telos_env = StandingTelosTaskEnv(task=telos_task, agent=telos_agent, sim_engine=pb)

    env = make_vec_env(
        make_standing_env,
        n_envs=16,
        env_kwargs={"task": telos_task, "agent": telos_agent, "sim": pb},
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    policy_tqc = dict(
        net_arch=dict(pi=[512, 512, 512], qf=[512, 512, 512]),
        n_critics=12,
        n_quantiles=25,
    )

    model = TQC(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_tqc,
        verbose=1,
        learning_rate=1e-4,
        ent_coef="auto",
        buffer_size=1_200_000,
        learning_starts=5_000,
        batch_size=2048,
        tau=0.005,
        gamma=0.995,
        train_freq=1,
        gradient_steps=1,
        target_entropy="auto",
        action_noise=None,
        tensorboard_log="./tensorboard/tqcTelos/",
    )

    model.learn(1_500_000)

    model.save("tqc_standing_15")
