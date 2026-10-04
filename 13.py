"""
Program 13 - Study the effect of different discount factor (Gamma) values
Lab Sheet-05 : Lab Assessment on Reinforcement Learning
MCA III Semester (Session 2026-2027)

This file is self-contained: it also includes the code from Programs 1-12
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


# ======================================================================
# PROGRAM 13 - Study the effect of different discount factor (Gamma) values  (new code starts here)
# ======================================================================

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


print("\nProgram 13 finished successfully.")