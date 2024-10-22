import numpy as np

def jacobian_com(q):
    # Placeholder for the Jacobian of the center of mass
    # Jacobian dimensions: 3x12 (3 task-space dimensions, 12 joint-space dimensions)
    J = np.zeros((3, 12))
    J[0, 0] = np.cos(q[0])
    J[1, 1] = np.cos(q[1])
    J[2, 2] = -np.sin(q[2])
    return J


# Riemannian Metric A (identity for simplicity, could be adapted)
def metric_A(q):
    # In practice, this could depend on the configuration q
    return np.eye(3)


# Target policy: Pull the CoM to a desired position (e.g., upright stance)
def target_policy(q, q_dot, xg,center_of_mass, alpha=1.0, beta=0.5, c=1.0):
    J = jacobian_com(q)  # Jacobian for velocity mapping
    x_dot = J @ q_dot  # velocity in task space

    # Error in task space
    s = lambda v: v / (
        np.linalg.norm(v) + c * np.log(1 + np.exp(-2 * c * np.linalg.norm(v)))
    )

    # Control force in task space
    f_g = alpha * s(xg - center_of_mass) - beta * x_dot

    return f_g


# Full RMP including the transformation from task space to joint space
def rmp_standing(q, q_dot, xg, center_of_mass):
    """
    Parameters:
    q -- current joint angles
    q_dot -- current joint velocities
    xg -- goal position in task space
    Returns:
    f_joint -- control force in joint space
    A_joint -- metric for the joint space
    """
    # Task-space policy (targeting center of mass)
    f_g = target_policy(q, q_dot, xg, center_of_mass)

    J = jacobian_com(q)
    A = metric_A(q)

    f_joint = (
        J.T @ A @ f_g
    )
    A_joint = J.T @ A @ J 
    return f_joint, A_joint