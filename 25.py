"""
Program 25 - Evaluate the trained DQN agent
Lab Sheet-05 : Lab Assessment on Reinforcement Learning
MCA III Semester (Session 2026-2027)

This file is self-contained: it also includes the code from Programs 1-24
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

# ---- inherited from Program 7 (Initialize and update the Q-table during training) ----

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

# ---- inherited from Program 8 (Train the agent for multiple episodes using Q-Learning) ----

frozen_lake = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)

NUM_EPISODES = 5000
start_time = time.time()
q_learning_result = train_q_learning(frozen_lake, num_episodes=NUM_EPISODES,
                                     learning_rate=0.8, discount_factor=0.95,
                                     epsilon_start=1.0, epsilon_min=0.01,
                                     epsilon_decay=0.999, seed=0, track_delta=True)
q_learning_training_time = time.time() - start_time

rewards = q_learning_result["rewards"]
print(f"Training finished in {q_learning_training_time:.2f} s over {NUM_EPISODES} episodes.\n")
print("Success rate per block of 500 episodes:")
for block in range(0, NUM_EPISODES, 500):
    block_rewards = rewards[block:block + 500]
    print(f"   episodes {block:>5}-{block + 499:<5} : average reward = {block_rewards.mean():.3f}")
print(f"\nFinal epsilon value : {q_learning_result['epsilons'][-1]:.4f}")

# ---- inherited from Program 9 (Display the learned Q-table after training) ----

learned_q_table = q_learning_result["q_table"]
ACTION_NAMES = ["LEFT", "DOWN", "RIGHT", "UP"]

q_dataframe = pd.DataFrame(np.round(learned_q_table, 4), columns=ACTION_NAMES)
q_dataframe.index.name = "State"
q_dataframe["Best action"] = [ACTION_NAMES[a] for a in np.argmax(learned_q_table, axis=1)]
q_dataframe["Tile"] = [lake_map[s // 4][s % 4] for s in range(16)]
print("LEARNED Q-TABLE\n")
print(q_dataframe.to_string())

# ---- Greedy policy shown on the 4x4 grid -------------------------------
ARROWS = {0: "<-", 1: "v ", 2: "->", 3: "^ "}
print("\nGREEDY POLICY DERIVED FROM THE Q-TABLE:")
for row in range(4):
    line = ""
    for col in range(4):
        state = row * 4 + col
        tile = lake_map[row][col]
        line += ("  H " if tile == "H" else "  G " if tile == "G"
                 else "  " + ARROWS[int(np.argmax(learned_q_table[state]))])
    print(line)

# ---- inherited from Program 10 (Evaluate the trained Q-Learning agent) ----

def evaluate_q_agent(env, q_table, num_episodes=100, max_steps=100, seed=500, verbose_first=False):
    """Run greedy (epsilon = 0) episodes and return success rate, mean reward and mean steps."""
    rng = np.random.default_rng(seed)
    successes, total_rewards, step_counts = 0, [], []
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        done, episode_reward, steps, path = False, 0.0, 0, [state]
        while not done and steps < max_steps:
            action = argmax_with_random_tiebreak(q_table[state], rng)   # purely greedy
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            path.append(state)
        successes += int(episode_reward > 0)
        total_rewards.append(episode_reward)
        step_counts.append(steps)
        if verbose_first and episode == 0:
            print("Path followed in test episode 1 :", path)
    return {"success_rate": 100 * successes / num_episodes,
            "mean_reward": float(np.mean(total_rewards)),
            "mean_steps": float(np.mean(step_counts))}

q_eval = evaluate_q_agent(frozen_lake, learned_q_table, num_episodes=100, verbose_first=True)
print(f"\nEVALUATION OF THE TRAINED Q-LEARNING AGENT (100 greedy episodes)")
print(f"   Success rate      : {q_eval['success_rate']:.1f} %")
print(f"   Average reward    : {q_eval['mean_reward']:.3f}")
print(f"   Average steps     : {q_eval['mean_steps']:.1f}")
print(f"   Random baseline   : {100 * np.mean(random_episode_rewards):.1f} % success")

# ---- inherited from Program 11 (Plot the cumulative rewards obtained during training) ----

cumulative_rewards = np.cumsum(rewards)
window = 100
moving_average = pd.Series(rewards).rolling(window).mean()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(cumulative_rewards, color="tab:blue")
axes[0].set_title("Cumulative reward during training")
axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Cumulative reward")

axes[1].plot(moving_average, color="tab:green")
axes[1].set_title(f"Moving-average reward (window = {window})")
axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Average reward"); axes[1].set_ylim(-0.05, 1.05)

axes[2].plot(q_learning_result["epsilons"], color="tab:red")
axes[2].set_title("Epsilon decay")
axes[2].set_xlabel("Episode"); axes[2].set_ylabel("Epsilon")

for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()  # in a plain terminal run, add plt.savefig(...) to save the figure

print(f"Total reward collected over {NUM_EPISODES} episodes : {cumulative_rewards[-1]:.0f}")

# ---- inherited from Program 12 (Study the effect of different learning rates on Q-Learning performance) ----

LEARNING_RATES = [0.01, 0.1, 0.5, 0.8, 1.0]
alpha_study = {}

for alpha in LEARNING_RATES:
    env_tmp = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    result = train_q_learning(env_tmp, num_episodes=5000, learning_rate=alpha,
                              discount_factor=0.95, epsilon_decay=0.999, seed=1)
    evaluation = evaluate_q_agent(env_tmp, result["q_table"], num_episodes=200)
    alpha_study[alpha] = {"rewards": result["rewards"], **evaluation}
    env_tmp.close()

plt.figure(figsize=(9, 5))
for alpha, data in alpha_study.items():
    plt.plot(pd.Series(data["rewards"]).rolling(100).mean(), label=f"alpha = {alpha}")
plt.title("Effect of the learning rate on Q-Learning (slippery FrozenLake)"); plt.xlabel("Episode")
plt.ylabel("Average reward (window = 100)"); plt.legend(); plt.grid(alpha=0.3); plt.show()  # in a plain terminal run, add plt.savefig(...) to save the figure

alpha_table = pd.DataFrame([{"Learning rate (alpha)": a,
                             "Mean training reward": round(d["rewards"].mean(), 3),
                             "Greedy success rate (%)": d["success_rate"],
                             "Mean steps": round(d["mean_steps"], 1)}
                            for a, d in alpha_study.items()])
print(alpha_table.to_string(index=False))
print("\nalpha too small -> the agent barely updates its estimates and learns far too slowly.")
print("alpha = 1.0      -> each new (noisy) sample completely overwrites the old estimate, so the")
print("                    Q-values oscillate and never settle. Moderate values work best.")

# ---- inherited from Program 13 (Study the effect of different discount factor (Gamma) values) ----

DISCOUNT_FACTORS = [0.0, 0.3, 0.6, 0.9, 0.99]
gamma_study = {}

for gamma in DISCOUNT_FACTORS:
    env_tmp = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    result = train_q_learning(env_tmp, num_episodes=5000, learning_rate=0.1,
                              discount_factor=gamma, epsilon_decay=0.999, seed=2)
    evaluation = evaluate_q_agent(env_tmp, result["q_table"], num_episodes=200)
    gamma_study[gamma] = {"rewards": result["rewards"], **evaluation}
    env_tmp.close()

plt.figure(figsize=(9, 5))
for gamma, data in gamma_study.items():
    plt.plot(pd.Series(data["rewards"]).rolling(100).mean(), label=f"gamma = {gamma}")
plt.title("Effect of the discount factor on Q-Learning (slippery FrozenLake)"); plt.xlabel("Episode")
plt.ylabel("Average reward (window = 100)"); plt.legend(); plt.grid(alpha=0.3); plt.show()  # in a plain terminal run, add plt.savefig(...) to save the figure

gamma_table = pd.DataFrame([{"Discount factor (gamma)": g,
                             "Mean training reward": round(d["rewards"].mean(), 3),
                             "Greedy success rate (%)": d["success_rate"]}
                            for g, d in gamma_study.items()])
print(gamma_table.to_string(index=False))
print("\nNote: gamma = 0 makes the agent completely myopic. Only the single transition that")
print("lands on the goal ever receives value; everywhere else the Q-values stay tied at 0,")
print("so the agent wanders at random and cannot follow a path to the goal.")

# ---- inherited from Program 14 (Compare exploration and exploitation using different epsilon values) ----

EPSILON_SETTINGS = {"Pure exploitation (eps = 0)":      dict(epsilon_start=0.0, epsilon_min=0.0, epsilon_decay=1.0),
                    "Low exploration (eps = 0.1 fixed)": dict(epsilon_start=0.1, epsilon_min=0.1, epsilon_decay=1.0),
                    "High exploration (eps = 0.5 fixed)":dict(epsilon_start=0.5, epsilon_min=0.5, epsilon_decay=1.0),
                    "Pure exploration (eps = 1.0)":      dict(epsilon_start=1.0, epsilon_min=1.0, epsilon_decay=1.0),
                    "Decaying eps (1.0 -> 0.01)":        dict(epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.998)}
epsilon_study = {}

for label, params in EPSILON_SETTINGS.items():
    env_tmp = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    result = train_q_learning(env_tmp, num_episodes=5000, learning_rate=0.1,
                              discount_factor=0.95, seed=3, **params)
    evaluation = evaluate_q_agent(env_tmp, result["q_table"], num_episodes=200)
    epsilon_study[label] = {"rewards": result["rewards"], **evaluation}
    env_tmp.close()

plt.figure(figsize=(10, 5))
for label, data in epsilon_study.items():
    plt.plot(pd.Series(data["rewards"]).rolling(100).mean(), label=label)
plt.title("Exploration vs exploitation (slippery FrozenLake)"); plt.xlabel("Episode")
plt.ylabel("Average reward (window = 100)"); plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.show()  # in a plain terminal run, add plt.savefig(...) to save the figure

eps_table = pd.DataFrame([{"Strategy": k,
                           "Mean reward while training": round(v["rewards"].mean(), 3),
                           "Greedy success rate (%)": v["success_rate"]}
                          for k, v in epsilon_study.items()])
print(eps_table.to_string(index=False))
print("\nThe two columns answer different questions: the middle column is the reward collected")
print("WHILE learning (exploration is expensive), the last column is how good the final policy")
print("is. The decaying schedule wins on both - it explores when the estimates are worthless")
print("and exploits once they are reliable.")

# ---- inherited from Program 15 (Implement an epsilon-greedy action selection policy) ----

def epsilon_greedy_policy(q_values, epsilon, rng):
    """Return an action index: random with probability epsilon, greedy otherwise.

    Ties among equal Q-values are broken at random (see Program 6).
    """
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))            # explore
    return argmax_with_random_tiebreak(q_values, rng)      # exploit

# Verification on a fixed set of Q-values where action 2 is clearly the best
sample_q_values = np.array([0.10, 0.25, 0.90, 0.05])
greedy_action = int(np.argmax(sample_q_values))
rng = np.random.default_rng(11)
TRIALS = 10000

print(f"Q-values      : {sample_q_values}   (greedy action = {greedy_action})")
print(f"{'epsilon':>8} {'greedy chosen':>15} {'expected':>10}")
for epsilon in [0.0, 0.1, 0.3, 0.5, 1.0]:
    chosen = [epsilon_greedy_policy(sample_q_values, epsilon, rng) for _ in range(TRIALS)]
    observed = 100 * np.mean(np.array(chosen) == greedy_action)
    expected = 100 * ((1 - epsilon) + epsilon / len(sample_q_values))
    print(f"{epsilon:>8.1f} {observed:>14.1f}% {expected:>9.1f}%")

# ---- inherited from Program 16 (Design a simple Grid World environment) ----

class GridWorldEnvironment:
    """A 5x5 deterministic grid world with walls, one trap and one goal."""

    ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
    MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size=5):
        self.size = size
        self.start_position = (0, 0)
        self.goal_position = (4, 4)
        self.trap_position = (3, 4)
        self.walls = {(0, 3), (1, 1), (1, 3), (2, 1), (3, 3), (4, 0)}
        self.num_states = size * size
        self.num_actions = 4
        self.max_steps = 50
        self.agent_position = self.start_position
        self.step_counter = 0

    # -- helpers ---------------------------------------------------------
    def state_index(self, position):
        return position[0] * self.size + position[1]

    # -- Gymnasium-style API --------------------------------------------
    def reset(self, seed=None):
        self.agent_position = self.start_position
        self.step_counter = 0
        return self.state_index(self.agent_position), {}

    def step(self, action):
        delta_row, delta_column = self.MOVES[action]
        new_row = self.agent_position[0] + delta_row
        new_column = self.agent_position[1] + delta_column
        candidate = (new_row, new_column)

        # A move into a wall or outside the grid leaves the agent where it is
        inside_grid = 0 <= new_row < self.size and 0 <= new_column < self.size
        if inside_grid and candidate not in self.walls:
            self.agent_position = candidate

        self.step_counter += 1
        reward, terminated = -1.0, False                    # step penalty
        if self.agent_position == self.goal_position:
            reward, terminated = 10.0, True
        elif self.agent_position == self.trap_position:
            reward, terminated = -10.0, True
        truncated = self.step_counter >= self.max_steps
        return self.state_index(self.agent_position), reward, terminated, truncated, {}

    def render(self):
        for row in range(self.size):
            line = ""
            for column in range(self.size):
                cell = (row, column)
                if cell == self.agent_position: line += " A "
                elif cell in self.walls:        line += " # "
                elif cell == self.goal_position:line += " G "
                elif cell == self.trap_position:line += " T "
                elif cell == self.start_position:line += " S "
                else:                            line += " . "
            print(line)


grid_world = GridWorldEnvironment()
state, _ = grid_world.reset()
print(f"Grid World created: {grid_world.num_states} states, {grid_world.num_actions} actions")
print(f"Initial state index: {state}\n")
grid_world.render()

print("\nDemonstration of three moves (RIGHT, DOWN, DOWN):")
for action in [3, 1, 1]:
    s, r, term, trunc, _ = grid_world.step(action)
    print(f"  action={GridWorldEnvironment.ACTION_NAMES[action]:<5} state={s:<3} "
          f"reward={r:<6.1f} terminated={term}")

# ---- inherited from Program 17 (Train a Q-Learning agent in the custom Grid World) ----

def train_q_learning_gridworld(env, num_episodes=2000, learning_rate=0.1, discount_factor=0.95,
                               epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.997, seed=0):
    """Tabular Q-Learning for the custom Grid World environment."""
    rng = np.random.default_rng(seed)
    q_table = np.zeros((env.num_states, env.num_actions))
    epsilon = epsilon_start
    rewards_per_episode, steps_per_episode, delta_history = [], [], []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done, total_reward, steps, largest_delta = False, 0.0, 0, 0.0
        while not done:
            action = epsilon_greedy_policy(q_table[state], epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            best_next = 0.0 if terminated else np.max(q_table[next_state])
            td_error = reward + discount_factor * best_next - q_table[state, action]
            q_table[state, action] += learning_rate * td_error

            largest_delta += abs(learning_rate * td_error)
            state, total_reward, steps = next_state, total_reward + reward, steps + 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        delta_history.append(largest_delta)

    return {"q_table": q_table, "rewards": np.array(rewards_per_episode),
            "steps": np.array(steps_per_episode), "deltas": np.array(delta_history)}


grid_world = GridWorldEnvironment()
start_time = time.time()
grid_result = train_q_learning_gridworld(grid_world, num_episodes=2000, seed=5)
grid_training_time = time.time() - start_time

grid_rewards = grid_result["rewards"]
print(f"Grid World training completed in {grid_training_time:.2f} s\n")
for block in range(0, 2000, 400):
    chunk = grid_rewards[block:block + 400]
    print(f"   episodes {block:>4}-{block+399:<4} : mean reward = {chunk.mean():>7.2f} , "
          f"mean steps = {grid_result['steps'][block:block+400].mean():.1f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(pd.Series(grid_rewards).rolling(50).mean(), color="tab:purple")
axes[0].set_title("Grid World - average reward"); axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Reward (window = 50)")
axes[1].plot(pd.Series(grid_result["steps"]).rolling(50).mean(), color="tab:orange")
axes[1].set_title("Grid World - steps per episode"); axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Steps (window = 50)")
for ax in axes: ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()  # in a plain terminal run, add plt.savefig(...) to save the figure

# ---- inherited from Program 18 (Visualize the optimal path learned by the agent) ----

grid_q_table = grid_result["q_table"]
ARROW_SYMBOLS = {0: "^", 1: "v", 2: "<", 3: ">"}

# ---- Greedy policy map --------------------------------------------------
print("LEARNED POLICY (arrows = greedy action):")
for row in range(grid_world.size):
    line = ""
    for column in range(grid_world.size):
        cell = (row, column)
        if cell in grid_world.walls:          line += " # "
        elif cell == grid_world.goal_position: line += " G "
        elif cell == grid_world.trap_position: line += " T "
        else:
            line += " " + ARROW_SYMBOLS[int(np.argmax(grid_q_table[row * grid_world.size + column]))] + " "
    print(line)

# ---- Roll out the greedy policy ----------------------------------------
state, _ = grid_world.reset()
optimal_path, done, total = [grid_world.agent_position], False, 0.0
while not done and len(optimal_path) < 30:
    action = int(np.argmax(grid_q_table[state]))          # policy is fully learned here
    state, reward, terminated, truncated, _ = grid_world.step(action)
    optimal_path.append(grid_world.agent_position)
    total += reward
    done = terminated or truncated

print(f"\nOptimal path  : {optimal_path}")
print(f"Path length   : {len(optimal_path) - 1} steps")
print(f"Total reward  : {total:.1f}")

# ---- Matplotlib visualisation ------------------------------------------
value_grid = np.max(grid_q_table, axis=1).reshape(grid_world.size, grid_world.size)
for wall in grid_world.walls:
    value_grid[wall] = np.nan

fig, ax = plt.subplots(figsize=(6, 6))
image = ax.imshow(value_grid, cmap="YlGn")
plt.colorbar(image, ax=ax, label="State value  max_a Q(s,a)")
path_rows = [p[0] for p in optimal_path]
path_cols = [p[1] for p in optimal_path]
ax.plot(path_cols, path_rows, "o-", color="crimson", linewidth=2.5, markersize=9, label="Optimal path")
for wall in grid_world.walls:
    ax.text(wall[1], wall[0], "WALL", ha="center", va="center", fontsize=7, color="black")
ax.text(grid_world.goal_position[1], grid_world.goal_position[0], "GOAL", ha="center", va="center",
        fontsize=8, fontweight="bold")
ax.text(grid_world.trap_position[1], grid_world.trap_position[0], "TRAP", ha="center", va="center",
        fontsize=8, fontweight="bold", color="darkred")
ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_title("Optimal path learned in Grid World")
ax.legend(loc="upper left"); plt.tight_layout(); plt.show()  # in a plain terminal run, add plt.savefig(...) to save the figure

# ---- inherited from Program 19 (Compare agent performance in FrozenLake and Grid World environments) ----

# Greedy evaluation of the Grid World agent
def evaluate_gridworld(env, q_table, num_episodes=50):
    successes, rewards_list, steps_list = 0, [], []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done, total, steps = False, 0.0, 0
        while not done:
            state, reward, terminated, truncated, _ = env.step(int(np.argmax(q_table[state])))
            total += reward; steps += 1
            done = terminated or truncated
        successes += int(env.agent_position == env.goal_position)
        rewards_list.append(total); steps_list.append(steps)
    return {"success_rate": 100 * successes / num_episodes,
            "mean_reward": float(np.mean(rewards_list)), "mean_steps": float(np.mean(steps_list))}

grid_eval = evaluate_gridworld(grid_world, grid_q_table)

comparison = pd.DataFrame([
    {"Environment": "FrozenLake-v1 (4x4)", "States": 16, "Actions": 4,
     "Training episodes": 5000, "Training time (s)": round(q_learning_training_time, 2),
     "Greedy success rate (%)": q_eval["success_rate"],
     "Mean reward": round(q_eval["mean_reward"], 2), "Mean steps": round(q_eval["mean_steps"], 1)},
    {"Environment": "Custom Grid World (5x5)", "States": 25, "Actions": 4,
     "Training episodes": 2000, "Training time (s)": round(grid_training_time, 2),
     "Greedy success rate (%)": grid_eval["success_rate"],
     "Mean reward": round(grid_eval["mean_reward"], 2), "Mean steps": round(grid_eval["mean_steps"], 1)}])
print(comparison.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(pd.Series(rewards).rolling(100).mean(), label="FrozenLake", color="tab:blue")
axes[0].set_title("FrozenLake learning curve"); axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Average reward")
axes[1].plot(pd.Series(grid_rewards).rolling(50).mean(), label="Grid World", color="tab:purple")
axes[1].set_title("Grid World learning curve"); axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Average reward")
for ax in axes: ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()  # in a plain terminal run, add plt.savefig(...) to save the figure

print("\nObservation: the reward scales differ (FrozenLake gives 0/1, Grid World gives -10..+10),")
print("so the curves are compared in shape - both converge, Grid World needs fewer episodes")
print("because its reward signal is dense (a penalty at every step) instead of sparse.")

# ---- inherited from Program 20 (Analyze the convergence behavior of the Q-Learning algorithm) ----

delta_history = q_learning_result["deltas"]
smoothed_delta = pd.Series(delta_history).rolling(50).mean()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(smoothed_delta, color="tab:red")
axes[0].set_title("Max |delta Q| per episode (FrozenLake)"); axes[0].set_xlabel("Episode")
axes[0].set_ylabel("max |delta Q|")

axes[1].semilogy(smoothed_delta.clip(lower=1e-8), color="tab:red")
axes[1].set_title("Same curve on a logarithmic scale"); axes[1].set_xlabel("Episode")
axes[1].set_ylabel("max |delta Q| (log)")

axes[2].plot(pd.Series(grid_result["deltas"]).rolling(50).mean(), color="tab:purple")
axes[2].set_title("Max |delta Q| per episode (Grid World)"); axes[2].set_xlabel("Episode")
for ax in axes: ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()  # in a plain terminal run, add plt.savefig(...) to save the figure

# The curve first RISES (the reward is discovered and starts propagating backwards through
# the table) and then FALLS towards zero. Convergence is therefore measured relative to the
# peak of the curve, not against a fixed constant.
smoothed_values = smoothed_delta.fillna(0).values
peak_episode = int(np.argmax(smoothed_values))
peak_value = smoothed_values[peak_episode]
after_peak = np.where(smoothed_values[peak_episode:] < 0.01 * peak_value)[0]
converged_at = int(peak_episode + after_peak[0]) if len(after_peak) else None

print(f"Peak of the update curve      : episode {peak_episode} (total |delta Q| = {peak_value:.4f})")
print(f"Updates fall below 1 % of peak : episode {converged_at}")
print(f"Mean total |delta Q| over episodes 0-199      : {np.mean(delta_history[:200]):.5f}")
print(f"Mean total |delta Q| over the last 200 episodes: {np.mean(delta_history[-200:]):.8f}")
final_change = np.mean(delta_history[-200:])
if final_change < 1e-9:
    print("Over the last 200 episodes the Q-table is essentially frozen (total change < 1e-9),")
    print("which means the algorithm has fully converged.")
else:
    print(f"The Q-table changes {peak_value / final_change:,.0f}x less than at the peak.")
print("\nInterpretation: almost nothing changes at the very beginning (every Q-value is 0 and")
print("most episodes end in a hole with reward 0). Once the goal is found the updates grow")
print("sharply as the reward propagates backwards, and they then decay towards zero as the")
print("Q-values settle on the optimal action-value function Q*.")

# ---- inherited from Program 21 (Install the required libraries for Deep Q-Network implementation) ----

# Run once if PyTorch is not present:
# !pip install torch                 (CPU build is sufficient for this lab)
# Alternative:  !pip install tensorflow

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in use   :", device)

torch.manual_seed(0); np.random.seed(0); random.seed(0)     # reproducibility
print("\nDeep Q-Network libraries configured successfully.")

# ---- inherited from Program 22 (Implement a basic Deep Q-Network using PyTorch) ----

class QNetwork(nn.Module):
    """Multi-layer perceptron that maps a state vector to one Q-value per action."""

    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_size))

    def forward(self, state):
        return self.network(state)


class ReplayMemory:
    """Fixed-size circular buffer that stores (s, a, r, s', done) transitions."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, random_sampling=True):
        """Random mini-batch (true experience replay) or the most recent transitions.

        random_sampling=False reproduces 'no replay': the network learns from the last
        batch_size transitions, which are consecutive and therefore strongly correlated.
        """
        if random_sampling:
            batch = random.sample(self.buffer, batch_size)
        else:
            batch = list(self.buffer)[-batch_size:]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)), torch.LongTensor(actions),
                torch.FloatTensor(rewards), torch.FloatTensor(np.array(next_states)),
                torch.FloatTensor(dones))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent with experience replay and a target network."""

    def __init__(self, state_size, action_size, learning_rate=1e-3, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=64, memory_capacity=10000,
                 use_replay=True, use_target_network=True, target_update_frequency=10):
        self.state_size, self.action_size = state_size, action_size
        self.gamma, self.batch_size = discount_factor, batch_size
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon_start, epsilon_min, epsilon_decay
        self.use_replay, self.use_target_network = use_replay, use_target_network
        self.target_update_frequency = target_update_frequency

        self.policy_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()
        self.memory = ReplayMemory(memory_capacity)
        self.losses = []

    # -- action selection -------------------------------------------------
    def select_action(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            q_values = self.policy_network(torch.FloatTensor(state).unsqueeze(0))
        return int(q_values.argmax().item())

    # -- learning ---------------------------------------------------------
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, random_sampling=self.use_replay)

        current_q = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            network = self.target_network if self.use_target_network else self.policy_network
            max_next_q = network(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_function(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), 10.0)
        self.optimizer.step()
        self.losses.append(loss.item())

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


cartpole = gym.make("CartPole-v1")
print("State size  :", cartpole.observation_space.shape[0])
print("Action size :", cartpole.action_space.n)
demo_agent = DQNAgent(cartpole.observation_space.shape[0], cartpole.action_space.n)
print("\nQ-Network architecture:")
print(demo_agent.policy_network)
print("\nTrainable parameters :",
      sum(p.numel() for p in demo_agent.policy_network.parameters() if p.requires_grad))

# ---- inherited from Program 23 (Train a DQN agent on the CartPole environment) ----

def train_dqn(num_episodes=300, max_steps=500, seed=0, use_replay=True,
              use_target_network=True, target_update_frequency=10, verbose=True,
              epsilon_decay=0.97, log_every=20):
    """Train a DQN agent on CartPole-v1 and return the agent and the reward history."""
    env = gym.make("CartPole-v1")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n,
                     use_replay=use_replay, use_target_network=use_target_network,
                     target_update_frequency=target_update_frequency,
                     epsilon_decay=epsilon_decay)
    episode_rewards = []
    start = time.time()

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed * 1000 + episode)
        total_reward, done, step = 0.0, False, 0

        while not done and step < max_steps:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.push(state, action, reward, next_state, float(terminated))
            agent.learn()
            state = next_state
            total_reward += reward
            step += 1

        agent.decay_epsilon()
        if agent.use_target_network and episode % agent.target_update_frequency == 0:
            agent.update_target_network()
        episode_rewards.append(total_reward)

        if verbose and (episode + 1) % log_every == 0:
            print(f"   Episode {episode + 1:>4}/{num_episodes} | reward = {total_reward:>6.1f} | "
                  f"avg(last 20) = {np.mean(episode_rewards[-20:]):>6.1f} | "
                  f"epsilon = {agent.epsilon:.3f}")
    env.close()
    return agent, np.array(episode_rewards), time.time() - start


print("Training the DQN agent on CartPole-v1 (this takes a few minutes) ...")
dqn_agent, dqn_rewards, dqn_training_time = train_dqn(num_episodes=250, max_steps=400, seed=1, log_every=25)
print(f"\nTraining finished in {dqn_training_time:.1f} s")
print(f"Best episode reward        : {dqn_rewards.max():.0f}")
print(f"Average reward (last 20)   : {dqn_rewards[-20:].mean():.1f}")
print(f"Average reward (first 20)  : {dqn_rewards[:20].mean():.1f}")
print(f"Final epsilon              : {dqn_agent.epsilon:.3f}")

# ---- inherited from Program 24 (Plot the episode-wise reward obtained during DQN training) ----

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(dqn_rewards, alpha=0.4, color="tab:gray", label="Episode reward")
axes[0].plot(pd.Series(dqn_rewards).rolling(20).mean(), color="tab:blue", linewidth=2,
             label="Moving average (20)")
axes[0].set_title("DQN on CartPole - episode reward"); axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Total reward"); axes[0].legend()

axes[1].plot(np.cumsum(dqn_rewards), color="tab:green")
axes[1].set_title("Cumulative reward"); axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Cumulative reward")

axes[2].plot(pd.Series(dqn_agent.losses).rolling(500).mean(), color="tab:red")
axes[2].set_title("Training loss (MSE, smoothed)"); axes[2].set_xlabel("Gradient update")
axes[2].set_ylabel("Loss")
for ax in axes: ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()  # in a plain terminal run, add plt.savefig(...) to save the figure


# ======================================================================
# PROGRAM 25 - Evaluate the trained DQN agent  (new code starts here)
# ======================================================================

def evaluate_dqn(agent, num_episodes=20, max_steps=500, seed=900, render_summary=True):
    """Run greedy episodes (no exploration) with the trained network."""
    env = gym.make("CartPole-v1")
    rewards_list = []
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        done, total, step = False, 0.0, 0
        while not done and step < max_steps:
            action = agent.select_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward; step += 1
        rewards_list.append(total)
    env.close()
    rewards_array = np.array(rewards_list)
    if render_summary:
        print(f"Episodes evaluated : {num_episodes}")
        print(f"Mean reward        : {rewards_array.mean():.1f}")
        print(f"Std deviation      : {rewards_array.std():.1f}")
        print(f"Min / Max reward   : {rewards_array.min():.0f} / {rewards_array.max():.0f}")
        print(f"Solved (>= 195)    : {100 * np.mean(rewards_array >= 195):.0f} % of episodes")
    return rewards_array

print("GREEDY EVALUATION OF THE TRAINED DQN AGENT")
dqn_eval_rewards = evaluate_dqn(dqn_agent, num_episodes=20)

plt.figure(figsize=(9, 4))
plt.bar(range(1, len(dqn_eval_rewards) + 1), dqn_eval_rewards, color="tab:blue")
plt.axhline(dqn_eval_rewards.mean(), color="red", linestyle="--",
            label=f"Mean = {dqn_eval_rewards.mean():.1f}")
plt.axhline(195, color="green", linestyle=":", label="Solved threshold (195)")
plt.title("DQN evaluation - reward per test episode"); plt.xlabel("Test episode")
plt.ylabel("Total reward"); plt.legend(); plt.grid(alpha=0.3, axis="y"); plt.show()  # in a plain terminal run, add plt.savefig(...) to save the figure


print("\nProgram 25 finished successfully.")