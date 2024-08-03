import pybullet as p
import time
import pybullet_data

# or p.DIRECT for non-graphical version
physicsClient = p.connect(p.GUI, options="--opengl2")
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("tt.urdf", startPos, startOrientation)
# set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
for i in range(10000):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos, cubeOrn)
p.disconnect()
