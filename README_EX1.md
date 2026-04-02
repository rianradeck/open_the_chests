AGENT FOR KUKA MOVEMENTS:

To improve the agent's performance on the colored chest reaching task, a series of experiments were conducted across multiple dimensions of the training pipeline.
On the architecture side, the policy and value function networks were progressively scaled from a lightweight [64, 64] configuration to a more expressive [256, 256] one, allowing the agent to represent more complex behaviors.
Several PPO hyperparameters were also swept: total training timesteps ranged from 100k to 1M, various learning rate schedules were tested, batch sizes and gae_lambda values were tuned, and the entropy coefficient ent_coef was varied to balance exploration and exploitation.
In parallel, the reward function was redesigned. The original distance-based signal was replaced with a delta reward — rewarding the agent for reductions in distance to the target at each step — complemented by a set of milestone bonuses triggered at decreasing distance thresholds. This shaped the agent's behavior more effectively toward the goal.
Despite these improvements, the most significant performance gain came from a change to the observation space. Rather than providing only the end-effector position and the target position separately and expecting the network to implicitly learn their relationship, the vector delta = target_pos - ee_pos was added as an explicit input. This meant the neural network no longer needed to dedicate capacity to learning a simple subtraction — it could focus entirely on learning the control policy. This single addition proved to be the most impactful change across all experiments.

PPO:
V1: = BATCH SIZE 128
V2 = 0.005 (ENT_COEF)
V3 = 0.003 (ENT_COEF)
Vfinal:
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

SAC:
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
