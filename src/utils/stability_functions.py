import numpy as np
import pybullet as p


def get_stability_task_map_from_obs(observation: dict) -> list:
    position = observation[:3]
    orientation = observation[3:7]

    # Compute pitch angle from orientation quaternion
    _, pitch, _ = p.getEulerFromQuaternion(orientation)

    # Extract joint velocities from the observation
    joint_velocities = observation[8:32:2]

    # Stability task state: [z_position, pitch, joint_velocities]
    z_position = position[2]
    task_state = np.concatenate(([z_position, pitch], joint_velocities))

    return task_state


def target_task_map(observation: list, target: np.ndarray) -> np.ndarray:
    position = observation[:2]
    base_velocity = observation[-3:]
    task_position = target[:2]
    task_state = np.concatenate((position, base_velocity[:2], task_position))
    return task_state


def complex_target_task_map(observation: list, target: np.ndarray) -> np.ndarray:
    position = observation
    current_position = np.array(position[:2])  # [x, y]
    if target.shape == (3,):
        target = target[:2]
    task_state = current_position - target
    joint_velocities = observation[8:32:2]
    body_velocity = observation[31:34]
    task_state = np.array([*task_state, *joint_velocities, *body_velocity])
    return task_state


def stability_policy(task_state, Kp, Kd, policy_shape=0):
    """
    A simple stability policy using a PD controller.

    Args:
    - task_state (ndarray): The state of the task, e.g., the deviation from a stable posture.

    Returns:
    - forces (ndarray): The corrective forces to apply.
    """

    z_position = task_state[0]
    pitch = task_state[1]
    joint_velocities = task_state[2:]

    # Simple PD control for z position and pitch
    force_z = (
        -Kp * z_position - Kd * joint_velocities[0]
    )  # Assuming first velocity is relevant
    force_pitch = (
        -Kp * pitch - Kd * joint_velocities[1]
    )  # Assuming second velocity is relevant

    forces = np.array([force_z, force_pitch, *joint_velocities])
    if policy_shape == 0:
        return forces
    assert policy_shape >= len(forces)
    return np.array([*forces, *np.zeros(policy_shape - len(forces))])


def stability_metric(task_state, z_position_emphasis, pitch_emphasis, policy_shape=0):
    """
    A stability metric that emphasizes the importance of maintaining balance.

    Args:
    - task_state (ndarray): The state of the task.

    Returns:
    - metric (ndarray): The Riemannian metric.
    """
    metric = np.identity(len(task_state) if policy_shape == 0 else policy_shape)
    metric[0, 0] = z_position_emphasis
    metric[1, 1] = pitch_emphasis
    return metric
