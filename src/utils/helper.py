import yaml
import math
import numpy as np


def load_yaml(path: str = "pybullet_config.yaml") -> dict:
    yaml_data = None
    with open(path, "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data


def soft_normalization(v):
    return v / h(np.linalg.norm(v), 0.05)


def h(z, c):
    return z + c * math.log(1 + math.exp(-2 * c * z))


def x_position_reward_function_exp(
    x,
    old_x,
    a: float = 0.09977489919830067,
    b: float = 450709068.7900891,
    lower_bound: float = 1e-13,
    upper_bound: float = 6e-9,
) -> float:
    """Calculate forward motion reward based on the x position of the agent. Using exponential function to calculate the reward, with precalculated a and b values. The reward is calculated as: reward = a * exp(b * (x - old_x))"""
    if old_x > x or x < lower_bound:
        return 0.0
    if x > upper_bound:
        return 1.0
    raw_reward = a * math.exp(b * (x - old_x))
    clamp_reward = min(1, max(0, raw_reward))
    return clamp_reward


def x_position_reward_function_quad(
    x,
    old_x,
    a: float = 340068010201360.2,
    b: float = -3400.6801020136018,
    c: float = 0.10000000850170025,
    lower_bound: float = 1e-13,
    upper_bound: float = 6e-8,
) -> float:
    """Calculate forward motion reward based on the x position of the agent. Using quadratic function to calculate the reward, with precalculated a and b values. The reward is calculated as: reward = a * (x - old_x)^2 + b * (x - old_x) + c"""
    if old_x > x or x < lower_bound:
        return 0.0
    if x > upper_bound:
        return 1.0
    raw_reward = a * (x - old_x) ** 2 + b * (x - old_x) + c
    clamp_reward = min(1, max(0, raw_reward))
    return clamp_reward


def softsign(x, incline=0.04):
    return x / (incline + abs(x))


def vel_reward_function(
    x: float,
    back_penalty: float = -0.1,
    shift_x: float = 0.6,
    x_min: float = 0.4,
    rew_max: float = 1.0,
) -> float:
    if x <= 0:
        return back_penalty
    if x < x_min:
        return x / 2
    return rew_max - (1 / ((x + shift_x) ** 4))
