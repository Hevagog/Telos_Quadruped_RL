
def force_to_angle(q, forces, Kp):
    """
    Convert forces from RMP directly into joint angles using a proportional controller.

    Parameters:
    q -- current joint angles (in radians)
    forces -- torques/forces from RMP (in Nm)
    Kp -- proportional gain to convert force to angle
    
    Returns:
    Updated joint angles
    """
    # Update the joint angles based on the forces and proportional gain
    q_new = q + Kp * forces

    return q_new

