import numpy as np


def orientation_cost(
    current_orientation: np.ndarray,
    desired_orientation: np.ndarray,
    lower_threshold_deg: float = 5.0,
    upper_threshold_deg: float = 45,
    weight_per_axis: np.ndarray = np.ones(2),
) -> float:
    """
    Calculate the cost of the orientation of the robot.

    Parameters:
        current_orientation (np.ndarray): Current [pitch, roll, yaw] in radians.
        start_orientation (np.ndarray): Desired [pitch, roll, yaw] in radians.
        lower_threshold_deg (float): Threshold in degrees below which cost is minimal.
        upper_threshold_deg (float): Threshold in degrees where cost saturates.
        max_cost_per_axis (np.ndarray): Maximum cost contribution per axis.

    Returns:
        float: Computed total cost.
    """
    lower_threshold_rad = np.deg2rad(lower_threshold_deg)
    upper_threshold_rad = np.deg2rad(upper_threshold_deg)

    delta = (current_orientation - desired_orientation + np.pi) % (2 * np.pi) - np.pi
    abs_delta = np.abs(delta)

    x = (abs_delta - lower_threshold_rad) / (upper_threshold_rad - lower_threshold_rad)
    x_clip = np.clip(x, 0, 1)

    smoothstep = 3 * x_clip**2 - 2 * x_clip**3
    cost_per_axis = weight_per_axis * smoothstep

    return np.sum(cost_per_axis)


def unbounded_orientation_cost(
    current_orientation: np.ndarray,
    desired_orientation: np.ndarray,
    lower_threshold_deg: float = 5.0,
    weight_per_axis: np.ndarray = np.ones(2),
):
    """
    Calculate the cost of the orientation of the robot.

    Parameters:
        current_orientation (np.ndarray): Current [pitch, roll, yaw] in radians.
        start_orientation (np.ndarray): Desired [pitch, roll, yaw] in radians.
        lower_threshold_deg (float): Threshold in degrees below which cost is minimal.

    Returns:
        float: Computed total cost.
    """
    lower_threshold_rad = np.deg2rad(lower_threshold_deg)

    delta = np.abs(current_orientation) - np.abs(desired_orientation)
    abs_delta = np.abs(delta)

    cost_per_axis = np.where(
        abs_delta > lower_threshold_rad,
        (1 + abs_delta) ** 3,
        0,
    )
    cost_per_axis = weight_per_axis * cost_per_axis

    return -np.sum(cost_per_axis)
