import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import SAC, PPO

from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from src.agent import TelosAgent
from src.standing_task.standing_task import StandingTelosTask
from src.standing_task.standing_environment import (
    StandingTelosTaskEnv,
    make_standing_env,
)
from src.utils.PyBullet import PyBullet
from src.utils.tensorboard_callbacks import TensorboardCallback

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

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )

    policy_sac = dict(net_arch=dict(pi=[1024, 1024, 1024], qf=[1024, 1024, 1024]))

    # custom_callback = TensorboardCallback(agent=telos_agent, log_frequency=1, verbose=1)
    name = "sac_standing_8"

    checkpoint_callback = CheckpointCallback(
        save_freq=156250,  # every 5_000_000 steps
        save_path="./checkpoiner/",
        name_prefix=name,
    )

    model_sac = SAC(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_sac,
        learning_rate=2e-4,
        buffer_size=int(1e6),
        batch_size=2048 * 8,
        ent_coef="auto",
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        action_noise=action_noise,
        target_update_interval=1,
        verbose=0,
        tensorboard_log="./tensorboard/tTelos/",
    )

    # callback = CallbackList([custom_callback, checkpoint_callback])

    try:
        model_sac.learn(60_000_000, callback=checkpoint_callback)
        model_sac.save(name)
    except KeyboardInterrupt:
        model_sac.save(name)
