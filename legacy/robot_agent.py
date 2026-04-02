# Core Gymnasium API
import gymnasium as gym

import numpy as np

# policy -> go straight towards the target position, with a maximum step size of 0.5 in any direction.
def calc_delta_action(obs):
    effector_vec = np.array(obs[0:3])
    target_vec = np.array(obs[3:6])
    res_vec = target_vec - effector_vec
    norm_res_vec = np.linalg.norm(res_vec)
    # unit_vec = res_vec / norm_res_vec if norm_res_vec > 0 else np.zeros_like(res_vec)
    # print(unit_vec)
    if norm_res_vec < 0.01:
        return [0.0, 0.0, 0.0] # No movement needed
    max_delta = np.max(np.abs(res_vec))
    scaled_res_vec = res_vec / (2 * max_delta) # Scale to [-0.5, 0.5] (to respect action space)
    return scaled_res_vec.tolist()

