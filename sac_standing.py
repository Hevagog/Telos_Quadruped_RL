import numpy as np
from src.agent import TelosAgent
from src.standing_task.standing_task import StandingTelosTask
from src.standing_task.standing_environment import (
    StandingTelosTaskEnv,
    make_standing_env,
)
from src.utils.PyBullet import PyBullet
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import SAC, PPO

if __name__ == "__main__":
    pb = PyBullet(render_mode="rgb_array", renderer="Tiny")
    telos_agent = TelosAgent(pb)
    telos_task = StandingTelosTask(agent=telos_agent, sim_engine=pb)
    telos_env = StandingTelosTaskEnv(task=telos_task, agent=telos_agent, sim_engine=pb)

    env = make_vec_env(
        make_standing_env,
        n_envs=12,
        env_kwargs={"task": telos_task, "agent": telos_agent, "sim": pb},
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )

    policy_sac = dict(net_arch=dict(pi=[512, 512, 512], qf=[512, 512, 512]))

    model_sac = SAC(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_sac,
        learning_rate=1e-4,
        buffer_size=int(1e6),
        batch_size=2048 * 8,
        ent_coef="auto",
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        action_noise=action_noise,
        target_update_interval=1,
        verbose=1,
        tensorboard_log="./tensorboard/sacTelos/",
    )

    model_sac.learn(2_000_000)

    model_sac.save("sac_standing_1")
