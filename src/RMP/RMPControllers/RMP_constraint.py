import numpy as np

def sigmoid(q, low_angles, high_angles, a=1.0):
    """
    Sigmoid function maps the constrained space to an unconstrained space
    based on specific joint limits.
    
    Parameters:
    q -- current joint angles
    low_angles -- minimum joint limits for each joint
    high_angles -- maximum joint limits for each joint
    a -- steepness of the sigmoid

    Returns:
    Unconstrained joint angles
    """
    normalized_q = (q - low_angles) / (high_angles - low_angles)
    return (2 * np.arctan(np.tanh(a * (normalized_q - 0.5))) / np.pi)

def inverse_sigmoid(q_unconstrained, low_angles, high_angles, a=1.0):
    """
    Inverse sigmoid maps the unconstrained space back to the joint limits.
    
    Parameters:
    q_unconstrained -- joint angles in the unconstrained space
    low_angles -- minimum joint limits for each joint
    high_angles -- maximum joint limits for each joint
    a -- steepness of the sigmoid

    Returns:
    Constrained joint angles within limits
    """
    tan_input = np.tan(np.pi * q_unconstrained / 2)
    tan_input = np.clip(tan_input, -0.999999, 0.999999)
    normalized_q = 0.5 + np.arctanh(tan_input) / (a * 2)
    return normalized_q * (high_angles - low_angles) + low_angles


def pushforward_force_and_metric(f_joint, A_joint, J_phi):
    """
    Pushforward the joint space force and metric to Cartesian space.

    Parameters:
    f_joint -- force vector in joint space
    A_joint -- metric matrix in joint space
    J_phi -- Jacobian matrix of the task map (from joint to Cartesian)

    Returns:
    f_cartesian -- force vector in Cartesian space
    A_cartesian -- metric matrix in Cartesian space
    """
    # Pushforward the force to Cartesian space
    f_cartesian = J_phi.T @ f_joint

    # Pushforward the metric to Cartesian space
    A_cartesian = J_phi.T @ A_joint @ J_phi

    return f_cartesian, A_cartesian


def RMP_joint_limits(q, dq, low_angles, high_angles, alpha=1.0, beta=0.5, a=1.0):
    """
    RMP for handling joint limits using a nonlinear sigmoid mapping.
    
    Parameters:
        q  -- current joint angles
        dq -- current joint velocities
        low_angles -- minimum joint limits for each joint
        high_angles -- maximum joint limits for each joint
        alpha -- gain for positional error
        beta -- gain for velocity damping
        a -- steepness of the sigmoid mapping
    
    Returns:
    f -- control force in joint space
    A -- metric for the joint limits
    """

    q_unconstrained = inverse_sigmoid(q, low_angles, high_angles, a)
    
    q_desired_unconstrained = np.zeros_like(q_unconstrained)
    
    f_unconstrained = -alpha * (q_unconstrained - q_desired_unconstrained) - beta * dq
    
    dsigmoid_dq = (a / 2) * (1 - np.tanh(a * (q_unconstrained - 0.5)) ** 2)
    f = dsigmoid_dq * f_unconstrained
    
    A = np.diag(dsigmoid_dq**2) 
    
    return f, A

