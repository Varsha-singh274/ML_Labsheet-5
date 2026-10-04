"""
Program 5 - Simulate random actions in the FrozenLake environment
Lab Sheet-05 : Lab Assessment on Reinforcement Learning
MCA III Semester (Session 2026-2027)

This file is self-contained: it also includes the code from Programs 1-4
that this program depends on (the Q-table, the trained agents, helper functions,
etc.), exactly as a Jupyter Notebook keeps state from earlier cells.
The NEW code for THIS program is marked below.
"""

import sys, os, time, random, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # save plots to file instead of requiring a GUI/inline backend
import matplotlib.pyplot as plt
import gymnasium as gym
warnings.filterwarnings("ignore")


# ---- inherited from Program 1 (Install and configure the Gymnasium (or OpenAI Gym) library) ----

# Run this cell once. Remove the leading '#' if the libraries are not installed yet.
# !pip install gymnasium numpy pandas matplotlib scikit-learn torch

import sys, time, random, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym

warnings.filterwarnings("ignore")

print("Python      :", sys.version.split()[0])
print("NumPy       :", np.__version__)
print("Pandas      :", pd.__version__)
print("Matplotlib  :", matplotlib.__version__)
print("Gymnasium   :", gym.__version__)
print("\nGymnasium configured successfully.")

# ---- inherited from Program 2 (Create and execute a simple Reinforcement Learning environment) ----

# Create a simple RL environment (4x4 FrozenLake, deterministic surface)
environment = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)

initial_state, info = environment.reset(seed=42)
print("Environment created :", environment.spec.id)
print("Initial state       :", initial_state)
print("Reset info          :", info)

# Execute one full interaction step: agent chooses an action, environment responds
chosen_action = 2                                   # 2 = move RIGHT
next_state, reward, terminated, truncated, info = environment.step(chosen_action)

print("\n--- One interaction step ---")
print("Action taken  :", chosen_action)
print("Next state    :", next_state)
print("Reward        :", reward)
print("Terminated    :", terminated)
print("Truncated     :", truncated)

environment.close()

# ---- inherited from Program 3 (Explore the observation space and action space of an RL environment) ----

def describe_spaces(env_name, **kwargs):
    """Print the observation space and action space of a Gymnasium environment."""
    env = gym.make(env_name, **kwargs)
    print(f"Environment        : {env_name}")
    print(f"Observation space  : {env.observation_space}")
    print(f"Action space       : {env.action_space}")
    if isinstance(env.observation_space, gym.spaces.Discrete):
        print(f"Number of states   : {env.observation_space.n}")
    else:
        print(f"Observation shape  : {env.observation_space.shape}")
        print(f"Low  bounds        : {np.round(env.observation_space.low, 2)}")
        print(f"High bounds        : {np.round(env.observation_space.high, 2)}")
    print(f"Number of actions  : {env.action_space.n}")
    print(f"Sample observation : {env.observation_space.sample()}")
    print(f"Sample action      : {env.action_space.sample()}")
    print("-" * 60)
    env.close()

describe_spaces("FrozenLake-v1", map_name="4x4", is_slippery=False)
describe_spaces("CartPole-v1")

# ---- inherited from Program 4 (Display the states, actions, rewards and termination conditions) ----

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
lake_map = env.unwrapped.desc.astype(str)            # the 4x4 tile layout

print("FrozenLake 4x4 map  (S=Start, F=Frozen, H=Hole, G=Goal)")
for row in lake_map:
    print("   " + "  ".join(row))

# ---- States -------------------------------------------------------------
tile_meaning = {"S": "Start", "F": "Frozen (safe)", "H": "Hole (episode ends, reward 0)",
                "G": "Goal (episode ends, reward 1)"}
print("\nSTATES (16 discrete states, numbered row-wise 0..15):")
state_rows = []
for state_id in range(env.observation_space.n):
    row, col = divmod(state_id, 4)
    tile = lake_map[row][col]
    state_rows.append([state_id, f"({row},{col})", tile, tile_meaning[tile]])
print(pd.DataFrame(state_rows, columns=["State", "Position", "Tile", "Meaning"]).to_string(index=False))

# ---- Actions ------------------------------------------------------------
print("\nACTIONS (4 discrete actions):")
for action_id, name in enumerate(["LEFT", "DOWN", "RIGHT", "UP"]):
    print(f"   {action_id} -> {name}")

# ---- Rewards and termination -------------------------------------------
print("\nREWARDS:")
print("   +1  on reaching the Goal tile 'G'")
print("    0  for every other transition (including falling into a hole)")
print("\nTERMINATION CONDITIONS:")
print("   terminated = True  -> agent reaches the Goal (G) or falls into a Hole (H)")
print("   truncated  = True  -> time limit of the environment is reached (100 steps)")
env.close()


# ======================================================================
# PROGRAM 5 - Simulate random actions in the FrozenLake environment  (new code starts here)
# ======================================================================

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
ACTION_NAMES = ["LEFT", "DOWN", "RIGHT", "UP"]

random_episode_rewards = []
NUM_RANDOM_EPISODES = 10

for episode in range(NUM_RANDOM_EPISODES):
    state, _ = env.reset(seed=100 + episode)
    total_reward, step_count, done = 0.0, 0, False
    if episode == 0:
        print("Trace of episode 1 (random policy)")
        print(f"{'Step':>4} {'State':>6} {'Action':>7} {'Next':>5} {'Reward':>7} {'Done':>6}")
    while not done and step_count < 100:
        action = env.action_space.sample()                 # random action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if episode == 0:
            print(f"{step_count:>4} {state:>6} {ACTION_NAMES[action]:>7} {next_state:>5} "
                  f"{reward:>7.1f} {str(done):>6}")
        state = next_state
        total_reward += reward
        step_count += 1
    random_episode_rewards.append(total_reward)

print("\nRewards of the 10 random episodes :", random_episode_rewards)
print(f"Average reward (random policy)    : {np.mean(random_episode_rewards):.2f}")
print(f"Success rate (random policy)      : {100 * np.mean(random_episode_rewards):.1f}%")
env.close()


print("\nProgram 5 finished successfully.")