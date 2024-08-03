import time
import math
import numpy as np
from agent import TelosAgent

if __name__ == "__main__":

    quadruped_agent = TelosAgent(renderer="OpenGL")

    time.sleep(1)

    for _ in range(100):
        for angle in range(-45, 46):
            angles = np.array([math.radians(angle)] * 16)
            angles = quadruped_agent.default_angles + angles
            indices = [0, 4, 8, 12]
            angles[indices] = 0
            quadruped_agent.set_action(angles)
            quadruped_agent.step_simulation()
            print(quadruped_agent.get_observation())
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
        quadruped_agent.reset_position()
        time.sleep(1)

    quadruped_agent.disconnect()
