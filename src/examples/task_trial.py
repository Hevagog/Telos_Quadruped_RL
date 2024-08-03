import math
import time
import numpy as np
import pybullet as p

from agent import TelosAgent
from task import TelosTask

if __name__ == "__main__":

    quadruped_agent = TelosAgent(renderer="OpenGL")
    task = TelosTask(quadruped_agent)

    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_CAPSULE,
        rgbaColor=[
            1.0,
            0.25,
            1.0,
            1.0,
        ],
        radius=0.1,
        length=100,
    )
    body_id = p.createMultiBody(
        basePosition=[*task.goal], baseVisualShapeIndex=visual_shape_id
    )
    # while True:
    #     # p.stepSimulation()
    #     # quadruped_agent.step_simulation()
    #     # print(task.compute_reward(quadruped_agent.get_obs()[0:3], task.goal))
    #     time.sleep(1.0 / 60.0)
    for _ in range(4):
        for angle in range(-45, 46):
            angles = np.array([math.radians(angle)] * 16)
            angles = quadruped_agent.default_angles + angles
            indices = [0, 4, 8, 12]
            angles[indices] = 0
            quadruped_agent.set_action(angles)
            quadruped_agent.step_simulation()
            time.sleep(0.01)
        for angle in range(45, -46, -1):
            angles = np.array([math.radians(angle)] * 16)
            angles = quadruped_agent.default_angles + angles
            indices = [0, 4, 8, 12]
            angles[indices] = 0
            quadruped_agent.set_action(angles)
            quadruped_agent.step_simulation()
            time.sleep(0.01)

        # Step simulation
        # Sleep for a short duration to observe the movement
        # quadruped_agent.reset_position()

        time.sleep(1)
    task.reset()
    quadruped_agent.step_simulation()
    time.sleep(10)

    quadruped_agent.disconnect()
