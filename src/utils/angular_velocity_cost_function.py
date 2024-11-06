import numpy as np


def angular_velocity_cost(joint_velocities: np.ndarray, min_angular_velocity=0.05):
    """
    Calculate the cost of the angular velocity of the robot.

    Parameters:
        joint_velocities (np.ndarray): Current joint velocities in rad/s.
        min_angular_velocity (float): Minimum angular velocity in rad/s.

    Returns:
        float: Computed total cost.
    """
    joint_velocities = np.abs(joint_velocities)
    costs = np.where(
        joint_velocities > min_angular_velocity,
        (joint_velocities - min_angular_velocity) ** 2,
        0,
    )
    total_cost = np.sum(costs)
    return -total_cost
