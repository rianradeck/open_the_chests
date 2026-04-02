from pathlib import Path

import gymnasium as gym
import colored_chest_kuka_env
import io
import imageio
import numpy as np


from agilerl.algorithms.ppo import PPO
from robot_agent import calc_delta_action
from utils import get_env, plot_reward_vs_distance

def create_rollout(env, policy):
    
    frames = []
    num_steps = 500
    obs, info = env.reset()

    for step_idx in range(num_steps):
        frame = env.render()
        frames.append(frame)

        # action = env.action_space.sample()
        # action, *_ = agent.get_action(obs)
        action, *_ = policy(obs)
        print(f"Step {step_idx:03d} | Action: {action}")

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"obs={obs}, reward={reward}, terminated={terminated}, truncated={truncated}, info={info}")

        # Optional debug print for quick inspection during development.
        # print(
        #    f"step={step_idx:03d} | reward={reward:.3f} | "
        #    f"terminated={terminated} | truncated={truncated}"
        #)

        # Stop the rollout cleanly if the episode has ended.
        if terminated or truncated:
            frames.append(env.render())
            print("Episode finished early.")
            break
    
    return frames


def save_frames_as_video(frames, filename):
    # Convert the list of frames (numpy arrays) into an in-memory video.
    frames = [frame.astype(np.uint8) for frame in frames]
    with io.BytesIO() as video_buffer:
        imageio.mimwrite(video_buffer, frames, format='MP4', fps=40)
        video_buffer.seek(0)
        with open(filename, 'wb') as f:
            f.write(video_buffer.read())
    
if __name__ == "__main__":
    plot_reward_vs_distance()
    checkpoint_path = Path(__file__).parent / "trained_agents/agilerl_PPO_basic.pt"
    agent = PPO.load(checkpoint_path)
    frames = []
    for i in range(3):
        env = get_env()
        frames.extend(create_rollout(env, lambda obs: agent.get_action(obs)))
    save_frames_as_video(frames, "rollout.mp4")