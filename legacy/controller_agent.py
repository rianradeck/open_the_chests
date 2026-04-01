import gymnasium as gym
import numpy as np
import colored_chest_kuka_env  # registers ColoredChestKuka-v0

import io
import imageio
from IPython.display import Image, display
from stable_baselines3 import PPO, SAC
from tqdm import tqdm   
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt

env = gym.make(
    "ColoredChestKuka-v0",
    # render_mode="rgb_array",
    reward_type="advanced",
    max_steps=200,
)

obs, info = env.reset()
print("Environment created successfully.")
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Initial info:", info)

eval_mode = True  # Set to True to train the model, False to evaluate
model = "SAC"
def lr_schedule(progress):
    return 1e-4 + (3e-4 - 1e-4) * progress


if eval_mode and model == "PPO":
    # model = PPO.load("ppo_chest_ent", env=env, force_reset=True, device="cpu")
    model = PPO.load("ppo_chest_best_model/best_model.zip", env=env, force_reset=True, device="cpu")
elif eval_mode and model == "SAC":
    model = SAC.load("sac_chest_best_model/best_model.zip", env=env, force_reset=True, device="cpu")
elif model == "PPO":
    # model = PPO("MlpPolicy", env, verbose=1, gamma=0.99, n_steps=2048, batch_size=128, n_epochs=10, device="cpu")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        gamma=0.99,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        ent_coef=0.005,
        gae_lambda=0.90,
        learning_rate=lr_schedule,
        policy_kwargs=dict(net_arch=dict(
            pi=[256, 256],
            vf=[256, 256]
        )),
        device="cpu",
        tensorboard_log="./ppo_chest_tensorboard/"
    )

    eval_callback = EvalCallback(
        env,
        eval_freq=10_000,
        n_eval_episodes=20,
        best_model_save_path="./ppo_chest_best_model",
        verbose=1,
    )
    model.learn(total_timesteps=1_000_000, callback=eval_callback)

    model.save("ppo_chest_ent")
elif model == "SAC":
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        gamma=0.99,
        batch_size=128,
        learning_rate=lr_schedule,
        tensorboard_log="./sac_chest_tensorboard/",
        policy_kwargs=dict(net_arch=dict(
            pi=[256, 256],
            qf=[256, 256]
        )),
        learning_starts=1000,
        buffer_size=1_000_000,
        device="cuda",
    )

    eval_callback = EvalCallback(
        env,
        eval_freq=10_000,
        n_eval_episodes=20,
        best_model_save_path="./sac_chest_best_model",
        verbose=1,
    )
    model.learn(total_timesteps=1_000_000, callback=eval_callback)

    model.save("sac_chest")


frames = []
num_steps = 150
# chest_names = ["red", "green", "blue"]
# episode_rewards = {name: [] for name in chest_names}
# episode_distances = {name: [] for name in chest_names}
total_runs = 5000
successfull_runs = 0
sum_final_dist = 0
for target_idx in tqdm(range(total_runs)):
    # name = chest_names[target_idx]
    # obs, info = env.reset(options={"target_idx": target_idx})
    obs, info = env.reset()
    # print(f"\n--- Episode: target = {name} chest ---")

    for step_idx in range(num_steps):
        # frames.append(env.render())

        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # episode_rewards[name].append(reward)
        # episode_distances[name].append(info["distance_to_target"])
        # print(f"obs={obs}, reward={reward}, terminated={terminated}, truncated={truncated}, info={info}")

        if terminated or truncated:
            frames.append(env.render())
            
            if info.get("is_success", False):
                successfull_runs += 1
                print("Success!")
            else:
                sum_final_dist += info.get("distance_to_target", 0)
            print("Episode finished early.")
            break

print(f"Total runs: {total_runs}, Successful runs: {successfull_runs}")
print(f"Average final distance for unsuccessful runs: {sum_final_dist / (total_runs - successfull_runs) if total_runs - successfull_runs > 0 else 0:.4f} m")


################ PLLOT 3 RUNS REWARDS #####
# fig, axes = plt.subplots(3, 1, figsize=(12, 12))
# colors = {"red": "red", "green": "green", "blue": "blue"}

# for ax, name in zip(axes, chest_names):
#     ax2 = ax.twinx()

#     ax.plot(episode_rewards[name], color=colors[name], label="reward")
#     ax2.plot(episode_distances[name], color=colors[name], linestyle="--", alpha=0.6, label="distance")
#     ax2.axhline(y=0.06, color="gray", linestyle=":", label="success threshold")

#     ax.set_title(f"{name} chest")
#     ax.set_xlabel("Step")
#     ax.set_ylabel("Reward", color=colors[name])
#     ax2.set_ylabel("Distance (m)", color="gray")
#     ax.grid(True)

#     lines1, labels1 = ax.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

# plt.tight_layout()
# plt.show()


# # Save frames as MP4
# video_path = "kuka_episode.mp4"
# frames_np = [np.asarray(f, dtype=np.uint8) for f in frames]
# imageio.mimsave(video_path, frames_np, fps=40, codec="libx264", macro_block_size=None)
# print("Video generated:", video_path)

# from IPython.display import Video
# Video(video_path, embed=True)

env.close()


if meta := env.metadata:
    print("Environment metadata:", meta)
else:
    print("No metadata found for this environment.")
