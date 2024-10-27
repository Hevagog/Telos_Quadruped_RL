import numpy as np


def combine_rmp(f1, A1, f2, A2):
    """
    Combine two RMPs using the metric-weighted average formula.

    Parameters:
    f1 -- force vector for RMP1
    A1 -- metric matrix for RMP1
    f2 -- force vector for RMP2
    A2 -- metric matrix for RMP2

    Returns:
    f_combined -- combined force vector
    A_combined -- combined metric matrix
    """
    A_combined = A1 + A2

    # Compute the sum of metric-weighted forces
    f_combined = np.linalg.pinv(A_combined) @ (A1 @ f1 + A2 @ f2)

    return f_combined, A_combined


def combine_rmps(rmps):
    """
    Combine a collection of RMPs using the metric-weighted average formula.

    Parameters:
        rmps: list of tuples, where each tuple contains:
                - f: force vector for the RMP
                - A: metric matrix for the RMP

    Returns:
    f_combined -- combined force vector
    A_combined -- combined metric matrix
    """
    # Initialize sums for the metric matrix and the metric-weighted forces
    A_sum = np.zeros_like(rmps[0][1])
    f_weighted_sum = np.zeros_like(rmps[0][0])

    # Iterate over all RMPs and compute the sums
    for f, A in rmps:
        A_sum += A
        f_weighted_sum += A @ f

    # Compute the combined force vector using the pseudoinverse
    f_combined = np.linalg.pinv(A_sum) @ f_weighted_sum

    return f_combined, A_sum
