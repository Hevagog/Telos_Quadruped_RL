import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


from src.utils.helper import load_yaml


class TensorboardCallback(BaseCallback):
    def __init__(self, agent, log_frequency=1000, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        _config = load_yaml("src/pybullet_config.yaml")
        self.fall_threshold = _config["task"]["fall_threshold"]
        self.up_threshold = _config["standing_task"]["up_threshold"]
        self.max_angle_dip = _config["standing_task"]["max_angle_dip"]
        self.max_robot_angular_velocity = _config["pybullet"]["robot"][
            "max_robot_angular_velocity"
        ]
        self.agent_start_pos = np.array(
            [*_config["pybullet"]["robot"]["start_orientation"]]
        )
        self.agent = agent
        self.log_frequency = log_frequency

    def _on_step(self) -> bool:
        if self.n_calls % self.log_frequency == 0:
            max_angle_diff = np.max(
                np.abs(
                    self.agent_start_pos[:2]
                    - self.agent.sim.get_orientation(self.agent.robot_agent)[:2]
                )
            )

            self.logger.record("angle/max_angle_diff", max_angle_diff)
            self.logger.record("position/z", self.agent.get_obs()[2])
            self.logger.record("custom/max_agent_vel", max(self.agent.get_obs()[3:6]))
            self.logger.record(
                "custom/max_joint_velocity",
                max(abs(self.agent.get_joints_velocities())),
            )

            # if max_angle_diff > self.max_angle_dip:
            #     self.logger.record("termination/max_angle_diff", 1)
            # else:
            #     self.logger.record("termination/max_angle_diff", 0)

            # if (
            #     self.agent.get_obs()[2] > self.up_threshold
            #     or self.agent.get_obs()[2] < self.fall_threshold
            # ):
            #     self.logger.record("termination/z", 1)
            # else:
            #     self.logger.record("termination/z", 0)

            # if (
            #     max(abs(self.agent.get_joints_velocities()))
            #     > self.max_robot_angular_velocity
            # ):
            #     self.logger.record("termination/max_joint_velocity", 1)
            # else:
            #     self.logger.record("termination/max_joint_velocity", 0)
        return True
