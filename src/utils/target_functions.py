import numpy as np


def target_policy(task_state, Kp, Kd, policy_shape=0):
    agent_position = task_state[:2]  #  [x, y]
    goal_position = task_state[4:6]  # [x, y]
    velocity = task_state[2:4]  # [vx, vy]
    position_deviation = agent_position - goal_position

    # Simple PD control for x and y position
    force = -Kp * position_deviation - Kd * velocity
    if policy_shape == 0:
        return force
    assert policy_shape >= len(force)
    return np.array([*np.zeros(policy_shape - len(force)), *force])


def target_metric(task_state, force_x_emphasis, force_y_emphasis, policy_shape=0):
    metric = np.identity(len(task_state) if policy_shape == 0 else policy_shape)
    metric[-2, -2] = force_x_emphasis
    metric[-1, -1] = force_y_emphasis
    return metric
