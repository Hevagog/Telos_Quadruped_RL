import pybullet as p
import os
import time
import pybullet_data

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
planeId = p.loadURDF("plane.urdf")
startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])

urdfRootPath = "/home/user/simulation_training/"
robot = p.loadURDF(
    os.path.join(urdfRootPath, "telos_2.urdf"),
    startPos,
    startOrientation,
    useFixedBase=True,
)

while True:
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
