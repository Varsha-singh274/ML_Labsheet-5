"""
Program 1 - Install and configure the Gymnasium (or OpenAI Gym) library
Lab Sheet-05 : Lab Assessment on Reinforcement Learning
MCA III Semester (Session 2026-2027)

This file is self-contained: it also includes the code from Programs 1-0
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



# ======================================================================
# PROGRAM 1 - Install and configure the Gymnasium (or OpenAI Gym) library  (new code starts here)
# ======================================================================

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


print("\nProgram 1 finished successfully.")