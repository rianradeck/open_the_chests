# based on the PPO tutorial in the AgileRL documentation
# https://docs.agilerl.com/en/latest/tutorials/gymnasium/agilerl_ppo_tutorial.html

import os

from tqdm import trange
import imageio
import gymnasium as gym
import numpy as np
import torch
from utils import get_env

from agilerl.algorithms import PPO
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_on_policy import train_on_policy
from agilerl.utils.utils import create_population, make_vect_envs
from agilerl.rollouts.on_policy import collect_rollouts

# Initial hyperparameters
INIT_HP = {
    "POP_SIZE": 4,  # Population size
    "BATCH_SIZE": 128,  # Batch size
    "LR": 0.001,  # Learning rate
    "LEARN_STEP": 1024,  # Learning frequency
    "GAMMA": 0.99,  # Discount factor
    "GAE_LAMBDA": 0.95,  # Lambda for general advantage estimation
    "ACTION_STD_INIT": 0.6,  # Initial action standard deviation
    "CLIP_COEF": 0.2,  # Surrogate clipping coefficient
    "ENT_COEF": 0.01,  # Entropy coefficient
    "VF_COEF": 0.5,  # Value function coefficient
    "MAX_GRAD_NORM": 0.5,  # Maximum norm for gradient clipping
    "TARGET_KL": None,  # Target KL divergence threshold
    "UPDATE_EPOCHS": 4,  # Number of policy update epochs
    "TARGET_SCORE": 200.0,  # Target score that will beat the environment
    "MAX_STEPS": 150000,  # Maximum number of steps an agent takes in an environment
    "EVO_STEPS": 10000,  # Evolution frequency
    "EVAL_STEPS": None,  # Number of evaluation steps per episode
    "EVAL_LOOP": 3,  # Number of evaluation episodes
    "TOURN_SIZE": 2,  # Tournament size
    "ELITISM": True,  # Elitism in tournament selection
}

# Mutation parameters
MUT_P = {
    # Mutation probabilities
    "NO_MUT": 0.4,  # No mutation
    "ARCH_MUT": 0.2,  # Architecture mutation
    "NEW_LAYER": 0.2,  # New layer mutation
    "PARAMS_MUT": 0.2,  # Network parameters mutation
    "ACT_MUT": 0.2,  # Activation layer mutation
    "RL_HP_MUT": 0.2,  # Learning HP mutation
    "MUT_SD": 0.1,  # Mutation strength
    "RAND_SEED": 42,  # Random seed
}

# RL hyperparameters configuration for mutation during training
hp_config = HyperparameterConfig(
    lr = RLParameter(min=1e-4, max=1e-2),
    batch_size = RLParameter(min=8, max=1024),
)

def train(env_name, num_envs=16, save_path = "PPO_trained_agent.pt"):
    print("Starting training...")
    env = make_vect_envs(env_name, num_envs)
    observation_space = env.single_observation_space
    action_space = env.single_action_space
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    if device == "cuda":
        print("GPU detected:", torch.cuda.get_device_name(0))

    # Define the network configuration of a simple mlp with two hidden layers, each with 64 nodes
    net_config = {"head_config": {"hidden_size": [64, 64]}}

    # Define a population
    pop = create_population(
        algo="PPO",  # RL algorithm
        observation_space=observation_space,  # State dimension
        action_space=action_space,  # Action dimension
        net_config=net_config,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameter
        hp_config=hp_config,  # RL hyperparameter configuration
        population_size=INIT_HP["POP_SIZE"],  # Population size
        num_envs=num_envs,
        device=device,
    )
    
    tournament = TournamentSelection(
        INIT_HP["TOURN_SIZE"],
        INIT_HP["ELITISM"],
        INIT_HP["POP_SIZE"],
        INIT_HP["EVAL_LOOP"],
    )
    
    mutations = Mutations(
        no_mutation=MUT_P["NO_MUT"],
        architecture=MUT_P["ARCH_MUT"],
        new_layer_prob=MUT_P["NEW_LAYER"],
        parameters=MUT_P["PARAMS_MUT"],
        activation=MUT_P["ACT_MUT"],
        rl_hp=MUT_P["RL_HP_MUT"],
        mutation_sd=MUT_P["MUT_SD"],
        rand_seed=MUT_P["RAND_SEED"],
        device=device,
    )

    trained_pop, pop_fitnesses = train_on_policy(
        env=env,
        env_name=env_name,
        algo="PPO",
        pop=pop,
        INIT_HP=INIT_HP,
        MUT_P=MUT_P,
        max_steps=INIT_HP["MAX_STEPS"],
        evo_steps=INIT_HP["EVO_STEPS"],
        eval_steps=INIT_HP["EVAL_STEPS"],
        eval_loop=INIT_HP["EVAL_LOOP"],
        tournament=tournament,
        mutation=mutations,
        wb=False,  # Boolean flag to record run with Weights & Biases
        save_elite=True,  # Boolean flag to save the elite agent in the population
        elite_path=save_path,
    )
    
if __name__ == "__main__":
    from pathlib import Path
    env = get_env("custom")
    train("ColoredChestKuka-v0", save_path=str(Path(__file__).parent / "trained_agents" / "agilerl_PPO_custom.pt"))