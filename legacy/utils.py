import gymnasium as gym
import colored_chest_kuka_env
import numpy as np


def reward_from_distance(
    distance: float,
    *,
    reward_type: str = "advanced",
    shaping_radius: float = 0.20,
    success_distance: float = 0.06,
    assume_success: bool = False,
) -> float:
    """Analytical reward as a function of distance.

    This re-implements the parts of `ColoredChestKukaEnv._compute_reward_and_success`
    that depend on distance, but **overrides** the environment distance with the
    provided `distance`.

    Notes
    -----
    - The "chest moved" penalty term is omitted here (assumed 0) because it depends
      on simulation state, not just distance.
    - Success in the env depends on holding close distance for multiple steps.
      Here, `assume_success=True` means: if `distance < success_distance`, add the
      success bonus (+20) as if the hold condition were already satisfied.
    """
    distance = float(distance)
    reward = -distance

    if reward_type == "advanced":
        if distance <= shaping_radius:
            bonus_scale = 1.0 - (distance / shaping_radius)
            reward += 10.0 * bonus_scale
    elif reward_type == "log":
        reward = np.log(1 / distance) if distance > 0 else 100.0
        # if distance < success_distance:
        #     reward += 20.0
    
    # if assume_success and distance < success_distance:
    #     reward += 20.0

    return float(reward)


def plot_reward_vs_distance(
    *,
    reward_type: str = "advanced",
    distance_max: float = 0.6,
    num_points: int = 400,
    shaping_radius: float = 0.20,
    success_distance: float = 0.06,
    show_success_curve: bool = True,
    save_path: str | None = None,
) -> object:
    """Plot reward vs distance using the analytical reward function.

    In headless/non-interactive environments (e.g. WSL), this will save a PNG
    instead of calling `plt.show()`.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from pathlib import Path

    distances = np.linspace(0.0, float(distance_max), int(num_points), dtype=np.float32)
    rewards = [
        reward_from_distance(
            d,
            reward_type=reward_type,
            shaping_radius=shaping_radius,
            success_distance=success_distance,
            assume_success=False,
        )
        for d in distances
    ]

    plt.figure(figsize=(6, 4))
    plt.plot(distances, rewards, label="reward (no success bonus)")

    if show_success_curve:
        rewards_success = [
            reward_from_distance(
                d,
                reward_type=reward_type,
                shaping_radius=shaping_radius,
                success_distance=success_distance,
                assume_success=True,
            )
            for d in distances
        ]
        plt.plot(distances, rewards_success, label="reward (assuming success)")

    plt.axvline(success_distance, linestyle="--", linewidth=1, label="success_distance")
    if reward_type == "advanced":
        plt.axvline(shaping_radius, linestyle=":", linewidth=1, label="shaping_radius")

    plt.xlabel("Distance to target (m)")
    plt.ylabel("Reward")
    plt.title(f"Reward vs Distance ({reward_type})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    backend = str(matplotlib.get_backend()).lower()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    elif "agg" in backend:
        default_path = Path.cwd() / f"reward_vs_distance_{reward_type}.png"
        plt.savefig(default_path, dpi=150)
        print(f"Saved plot to {default_path}")
    else:
        plt.show()

    return plt.gcf()


def save_figures_to_one_png(
    figures: list,
    out_path: str,
    *,
    cols: int = 2,
    dpi: int = 150,
) -> None:
    """Combine multiple Matplotlib Figure objects into one PNG (grid)."""
    import math
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if not figures:
        raise ValueError("figures must be a non-empty list of matplotlib Figure objects")

    cols = max(1, int(cols))
    rows = int(math.ceil(len(figures) / cols))

    images = []
    for fig in figures:
        if fig.canvas is None or not hasattr(fig.canvas, "draw"):
            FigureCanvasAgg(fig)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        images.append(buf[..., :3])

    out_fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, img in enumerate(images):
        axes[i].imshow(img)

    out_fig.tight_layout(pad=0.1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_fig.savefig(out_path, dpi=dpi)
    plt.close(out_fig)


def get_env(reward_type="advanced"):
    # Create the environment through Gymnasium's registry.
    
    # check the `colored_chest_kuka_env.py` file for details on the constructor
    env = gym.make(
        "ColoredChestKuka-v0",
        render_mode="rgb_array",
        reward_type=reward_type,
        max_steps=150,
    )
    # Reset the environment to obtain the initial observation and info dictionary.
    obs, info = env.reset()

    # Print a few useful details so you can quickly verify the environment loaded.
    # as expected and inspect its spaces.
    print("Environment created successfully.")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Initial info:", info)
    
    return env

if __name__ == "__main__":
    fig_basic = plot_reward_vs_distance(reward_type="basic", save_path="basic.png")
    fig_adv = plot_reward_vs_distance(reward_type="advanced", save_path="advanced.png")
    fig_log = plot_reward_vs_distance(reward_type="log", save_path="log.png")
    save_figures_to_one_png([fig_basic, fig_adv, fig_log], "combined.png", cols=3)