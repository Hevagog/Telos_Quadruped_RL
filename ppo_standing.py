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
        n_envs=32,
        env_kwargs={"task": telos_task, "agent": telos_agent, "sim": pb},
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    policy_ppo = {
        "net_arch": (dict(pi=[512, 512, 512], vf=[512, 512, 512])),
        "activation_fn": th.nn.ReLU,
    }

    model_ppo = PPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_ppo,
        verbose=1,
        learning_rate=0.0001,
        n_steps=2048,
        batch_size=2048,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.1,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./tensorboard/ppoTelos/",
    )

    model_ppo.learn(15_00_000)

    model_ppo.save("ppo_standing_2")
