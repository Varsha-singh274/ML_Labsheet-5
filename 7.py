"""
Program 7 - Initialize and update the Q-table during training
Lab Sheet-05 : Lab Assessment on Reinforcement Learning
MCA III Semester (Session 2026-2027)

This file is self-contained: it also includes the code from Programs 1-6
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

# ---- inherited from Program 5 (Simulate random actions in the FrozenLake environment) ----

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

# ---- inherited from Program 6 (Implement the Q-Learning algorithm for the FrozenLake environment) ----

def argmax_with_random_tiebreak(q_values, rng):
    """Return the index of the largest Q-value, breaking ties uniformly at random.

    This matters a great deal in practice: at the start of training every Q-value is 0,
    so a plain np.argmax would always return action 0 and the agent would repeat the same
    action forever instead of discovering the environment.
    """
    best_indices = np.flatnonzero(q_values == q_values.max())
    return int(rng.choice(best_indices))


def epsilon_greedy_action(q_table, state, epsilon, action_space_size, rng):
    """Choose a random action with probability epsilon, otherwise the greedy (best) action."""
    if rng.random() < epsilon:
        return int(rng.integers(action_space_size))       # EXPLORATION
    return argmax_with_random_tiebreak(q_table[state], rng)   # EXPLOITATION


def train_q_learning(env, num_episodes=5000, learning_rate=0.8, discount_factor=0.95,
                     epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.999,
                     max_steps=100, seed=0, track_delta=False):
    """Train a tabular Q-Learning agent and return the Q-table plus training statistics."""
    rng = np.random.default_rng(seed)
    num_states, num_actions = env.observation_space.n, env.action_space.n

    q_table = np.zeros((num_states, num_actions))          # Step 1: initialise Q(s,a) = 0
    epsilon = epsilon_start
    rewards_per_episode, epsilon_history, max_delta_history, steps_per_episode = [], [], [], []

    for episode in range(num_episodes):
        state, _ = env.reset(seed=int(rng.integers(1e6)))
        total_reward, episode_delta, done, step = 0.0, 0.0, False, 0

        while not done and step < max_steps:
            action = epsilon_greedy_action(q_table, state, epsilon, num_actions, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # ---- Q-Learning update rule -------------------------------------
            best_next_value = 0.0 if terminated else np.max(q_table[next_state])
            td_target = reward + discount_factor * best_next_value
            td_error = td_target - q_table[state, action]
            q_table[state, action] += learning_rate * td_error
            # -----------------------------------------------------------------

            episode_delta += abs(learning_rate * td_error)   # total change of the Q-table
            state = next_state
            total_reward += reward
            step += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)   # decay exploration
        rewards_per_episode.append(total_reward)
        epsilon_history.append(epsilon)
        steps_per_episode.append(step)
        if track_delta:
            max_delta_history.append(episode_delta)

    return {"q_table": q_table, "rewards": np.array(rewards_per_episode),
            "epsilons": np.array(epsilon_history), "deltas": np.array(max_delta_history),
            "steps": np.array(steps_per_episode)}

print("Q-Learning implementation ready.")


# ======================================================================
# PROGRAM 7 - Initialize and update the Q-table during training  (new code starts here)
# ======================================================================

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
ALPHA, GAMMA = 0.8, 0.95

# ---- Initialisation -----------------------------------------------------
demo_q_table = np.zeros((env.observation_space.n, env.action_space.n))
print("Q-table after initialisation (shape {}):".format(demo_q_table.shape))
print(demo_q_table[:6], "...\n")

# ---- A few manual updates ----------------------------------------------
state, _ = env.reset(seed=7)
rng = np.random.default_rng(7)
print(f"{'Step':>4} {'S':>3} {'A':>3} {'S_next':>7} {'R':>4} {'old Q':>8} {'TD target':>10} {'new Q':>8}")
for step in range(6):
    action = epsilon_greedy_action(demo_q_table, state, 1.0, env.action_space.n, rng)
    next_state, reward, terminated, truncated, _ = env.step(action)
    old_value = demo_q_table[state, action]
    td_target = reward + GAMMA * (0.0 if terminated else np.max(demo_q_table[next_state]))
    demo_q_table[state, action] = old_value + ALPHA * (td_target - old_value)
    print(f"{step:>4} {state:>3} {action:>3} {next_state:>7} {reward:>4.0f} "
          f"{old_value:>8.4f} {td_target:>10.4f} {demo_q_table[state, action]:>8.4f}")
    state = next_state
    if terminated or truncated:
        state, _ = env.reset(seed=8)

print("\nNon-zero entries after 6 updates:", int(np.count_nonzero(demo_q_table)))
env.close()


print("\nProgram 7 finished successfully.")