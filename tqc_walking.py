import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import SAC
from sb3_contrib import TQC

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
    n_envs = 16
    env = make_vec_env(
        make_walking_env,
        n_envs=n_envs,
        env_kwargs={"task": telos_task, "agent": telos_agent, "sim": pb},
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    n_actions = env.action_space.shape[-1]

    # action_noise = NormalActionNoise(
    #     mean=np.zeros(n_actions), sigma=0.002 * np.ones(n_actions)
    # )

    policy_tqc = dict(
        net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256]),
        n_critics=8,
        n_quantiles=25,
    )

    model = TQC(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_tqc,
        verbose=0,
        learning_rate=1e-4,
        ent_coef="auto",
        buffer_size=2_400_000,
        learning_starts=5_000,
        batch_size=2 * 2048,
        gamma=0.997,
        train_freq=1,
        gradient_steps=1,
        target_entropy="auto",
        # action_noise=action_noise,
        tensorboard_log="./tensorboard/tqcTelos/",
    )

    name = "simple_tqc_walking_00_no_noise"

    checkpoint_callback = CheckpointCallback(
        save_freq=int(500_000 / n_envs),  # roughly every 500_000 steps
        save_path="./checkpoints/",
        name_prefix=name,
        save_vecnormalize=True,
    )
    curriculum_callback = CurriculumCallback(vec_env=env, verbose=0)

    try:
        model.learn(25_000_000, callback=[checkpoint_callback, curriculum_callback])
        model.save(name)
    except KeyboardInterrupt:
        model.save(name)
