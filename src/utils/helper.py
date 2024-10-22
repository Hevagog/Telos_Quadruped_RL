import yaml
import math
import numpy as np

def load_yaml(path: str) -> dict:
    yaml_data = None
    with open("pybullet_config.yaml", "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

def soft_normalization(v):
    return v / h(np.linalg.norm(v), 0.05)

def h(z,c):
    return z + c *math.log(1 + math.exp(-2*c*z))