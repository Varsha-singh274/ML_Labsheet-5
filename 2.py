"""
Program 2 - Create and execute a simple Reinforcement Learning environment
Lab Sheet-05 : Lab Assessment on Reinforcement Learning
MCA III Semester (Session 2026-2027)

This file is self-contained: it also includes the code from Programs 1-1
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


# ======================================================================
# PROGRAM 2 - Create and execute a simple Reinforcement Learning environment  (new code starts here)
# ======================================================================

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


print("\nProgram 2 finished successfully.")