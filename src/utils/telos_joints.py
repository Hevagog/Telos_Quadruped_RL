import enum
import numpy as np

HIP_ANGLE = 0

HIP_MIN_ANGLE = -1.0472
HIP_REL_MIN_ANGLE = -1.0472
HIP_MAX_ANGLE = 0.5235987756
HIP_REL_MAX_ANGLE = 0.5235987756

THIGH_HIP_ANGLE = -0.7853981633974483  # FOR WALKING TASK
# THIGH_HIP_ANGLE = -1.48353  # FOR STANDING TASK

THIGH_MIN_ANGLE = -2.09439510239
THIGH_REL_MIN_ANGLE = -1.3089969389925518
THIGH_MAX_ANGLE = 0.5235987756
THIGH_REL_MAX_ANGLE = 1.3089969389925518

KNEE_ANGLE = -1.48352986  # FOR WALKING TASK
# KNEE_ANGLE = -2.35619  # FOR STANDING TASK

KNEE_MIN_ANGLE = -2.35619
KNEE_REL_MIN_ANGLE = -0.8726601399999998
KNEE_MAX_ANGLE = 0.5235987756
KNEE_REL_MAX_ANGLE = 0.9599310844

HIP_THIGH_FORCE = 1.5  # in Nm
KNEE_FORCE = 1.5  # in Nm

# MOVING_JOINTS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15] # COMPLEX, NO SENSING
# MOVING_JOINTS = [2, 3, 6, 7, 10, 11, 14, 15] # SIMPLE, NO SENSING

# MOVING_JOINTS = [1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18]  # COMPLEX, SENSING
MOVING_JOINTS = [2, 3, 7, 8, 12, 13, 17, 18]  # SIMPLE, SENSING


# class TelosJoints(enum.Enum):  # FOR NO SENSING
#     RIGID_FR = 0
#     REVOLUTE_FR_HIP = 1
#     REVOLUTE_FR_THIGH = 2
#     REVOLUTE_FR_KNEE = 3
#     RIGID_FL = 4
#     REVOLUTE_FL_HIP = 5
#     REVOLUTE_FL_THIGH = 6
#     REVOLUTE_FL_KNEE = 7
#     RIGID_BR = 8
#     REVOLUTE_BR_HIP = 9
#     REVOLUTE_BR_THIGH = 10
#     REVOLUTE_BR_KNEE = 11
#     RIGID_BL = 12
#     REVOLUTE_BL_HIP = 13
#     REVOLUTE_BL_THIGH = 14
#     REVOLUTE_BL_KNEE = 15


class TelosJoints(enum.Enum):  # FOR SENSING
    RIGID_FR = 0
    REVOLUTE_FR_HIP = 1
    REVOLUTE_FR_THIGH = 2
    REVOLUTE_FR_KNEE = 3
    FIXED_smol_FR = 4

    RIGID_FL = 5
    REVOLUTE_FL_HIP = 6
    REVOLUTE_FL_THIGH = 7
    REVOLUTE_FL_KNEE = 8
    FIXED_smol_FL = 9

    RIGID_BR = 10
    REVOLUTE_BR_HIP = 11
    REVOLUTE_BR_THIGH = 12
    REVOLUTE_BR_KNEE = 13
    FIXED_smol_BR = 14

    RIGID_BL = 15
    REVOLUTE_BL_HIP = 16
    REVOLUTE_BL_THIGH = 17
    REVOLUTE_BL_KNEE = 18
    FIXED_smol_BL = 19


HIP_IDX = [
    TelosJoints.REVOLUTE_FR_HIP.value,
    TelosJoints.REVOLUTE_FL_HIP.value,
    TelosJoints.REVOLUTE_BR_HIP.value,
    TelosJoints.REVOLUTE_BL_HIP.value,
]

TIP_IDX = [  # FOR SENSING
    TelosJoints.FIXED_smol_FR.value,
    TelosJoints.FIXED_smol_FL.value,
    TelosJoints.FIXED_smol_BR.value,
    TelosJoints.FIXED_smol_FL.value,
]

# SENSING
DEFAULT_ANGLES = [0, HIP_ANGLE, THIGH_HIP_ANGLE, KNEE_ANGLE, 0] * 4

# NO SENSING
# DEFAULT_ANGLES = [0, HIP_ANGLE, THIGH_HIP_ANGLE, KNEE_ANGLE] * 4

# # SIMPLE
DEFAULT_MOVING_ANGLES = [THIGH_HIP_ANGLE, KNEE_ANGLE] * 4
REL_LOW_ANGLES = np.array([THIGH_REL_MIN_ANGLE, KNEE_REL_MIN_ANGLE] * 4)
REL_HIGH_ANGLES = np.array([THIGH_REL_MAX_ANGLE, KNEE_REL_MAX_ANGLE] * 4)

# # COMPLEX
# DEFAULT_MOVING_ANGLES = [HIP_ANGLE, THIGH_HIP_ANGLE, KNEE_ANGLE] * 4

# REL_LOW_ANGLES = np.array(
#     [HIP_REL_MIN_ANGLE, THIGH_REL_MIN_ANGLE, KNEE_REL_MIN_ANGLE] * 4
# )
# REL_HIGH_ANGLES = np.array(
#     [HIP_REL_MAX_ANGLE, THIGH_REL_MAX_ANGLE, KNEE_REL_MAX_ANGLE] * 4
# )

low_angles = np.array(
    [
        HIP_MIN_ANGLE,
        THIGH_MIN_ANGLE,
        KNEE_MIN_ANGLE,
    ]
    * 4
)
high_angles = np.array(
    [
        HIP_MAX_ANGLE,
        THIGH_MAX_ANGLE,
        KNEE_MAX_ANGLE,
    ]
    * 4
)
