import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import SAC

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
    n_steps, n_envs = 8192, 64
    env = make_vec_env(
        make_walking_env,
        n_envs=n_envs,
        env_kwargs={"task": telos_task, "agent": telos_agent, "sim": pb},
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    n_actions = env.action_space.shape[-1]

    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.002 * np.ones(n_actions)
    )

    policy_sac = dict(net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256]))

    name = "simple_sac_walking_00_noise_gamma"

    checkpoint_callback = CheckpointCallback(
        save_freq=int(2_000_000 / n_envs),  # roughly every 2_000_000 steps
        save_path="./checkpoints/",
        name_prefix=name,
        save_vecnormalize=True,
    )

    curriculum_callback = CurriculumCallback(vec_env=env, verbose=0)

    model_sac = SAC(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_sac,
        learning_rate=2e-4,
        buffer_size=int(1e6),
        batch_size=2048 * 8,
        ent_coef="auto",
        gamma=0.997,  # 0.9988 like in the paper
        train_freq=1,
        gradient_steps=1,
        action_noise=action_noise,
        target_update_interval=1,
        verbose=0,
        tensorboard_log="./tensorboard/n_sacTelos/",
    )

    # callback = CallbackList([custom_callback, checkpoint_callback])

    try:
        model_sac.learn(60_000_000, callback=[checkpoint_callback, curriculum_callback])
        model_sac.save(name)
    except KeyboardInterrupt:
        model_sac.save(name)
