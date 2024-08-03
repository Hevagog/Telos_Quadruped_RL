import os
import time
from utils.telos_joints import TelosJoints
import pybullet as p
import pybullet_data

physicsClient = p.connect(p.GUI)  # or DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -9.8)
planeId = p.loadURDF("plane.urdf")
startPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([1.5707963267948966, 0, 0])
_current_dir = os.path.dirname(os.path.realpath(__file__))
_urdf_root_path = _current_dir + "/urdf"
_urdf_robot_path = _urdf_root_path + "/tt.urdf"
robotId = p.loadURDF(
    _urdf_robot_path,
    startPos,
    cubeStartOrientation,
)

for i in range(p.getNumJoints(robotId)):
    print(p.getJointInfo(robotId, i))

target_thigh_angle = -0.7853981633974483
target_knee_angle = -1.48352986
all_thigh_and_knee_joints = [
    TelosJoints.REVOLUTE_BL_THIGH.value,
    TelosJoints.REVOLUTE_BL_KNEE.value,
    TelosJoints.REVOLUTE_BR_THIGH.value,
    TelosJoints.REVOLUTE_BR_KNEE.value,
    TelosJoints.REVOLUTE_FL_THIGH.value,
    TelosJoints.REVOLUTE_FL_KNEE.value,
    TelosJoints.REVOLUTE_FR_THIGH.value,
    TelosJoints.REVOLUTE_FR_KNEE.value,
]
all_thigh_and_knee_angles = [target_thigh_angle, target_knee_angle] * 4

p.setJointMotorControlArray(
    robotId,
    all_thigh_and_knee_joints,
    p.POSITION_CONTROL,
    targetPositions=all_thigh_and_knee_angles,
)
while True:
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
