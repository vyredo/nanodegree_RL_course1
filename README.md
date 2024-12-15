# Banana Navigation - Reinforcement Learning Project

This repository contains the solution to the Udacity Nanodegree project for the Banana Navigation environment. The agent is trained to collect bananas using a Dueling Deep Q-Network (Dueling DQN) with Prioritized Experience Replay (PER).

---

## Getting Started

Follow the steps below to run the project and train the agent:

1. Open the Jupyter notebook: `p1_navigation/Navigation.ipynb`

2. Configure your Python environment:  
   Update the notebook with the correct paths to your Python binaries and library locations. Replace `[USERNAME]` with your system username in the snippet below:

```python
import os
os.environ['PATH'] = f"{os.environ['PATH']}:/home/[USERNAME]/.local/bin"
os.environ['PATH'] = f"{os.environ['PATH']}:/home/[USERNAME]/mambaforge/envs/py310/lib/python3.10/site-packages"
```

3. Run all the cells in the notebook to start training the agent.

---

### Project Details

#### Key Features

- Algorithm: Dueling DQN with Prioritized Replay Memory (PER) for efficient learning.
- Pretrained Model: The trained model is saved at:

```bash
runs/Banana_Linux.pt
```

- Logs: Training logs are available at:

```bash
runs/Banana_Linux.log
```

---

### Training Progress

Below are the training results from the last 700 episodes. The agent achieved a **total mean reward of 12.75** after 10,700 episodes of training:

```mathematica
Episode 10000: Total Mean reward = 12.610, Mean Reward last 50 = 13.12, Epsilon = 0.01, Best Rewards = 26.0
Episode 10050: Total Mean reward = 12.584, Mean Reward last 50 = 7.44, Epsilon = 0.01, Best Rewards = 26.0
Episode 10100: Total Mean reward = 12.559, Mean Reward last 50 = 7.52, Epsilon = 0.01, Best Rewards = 26.0
Episode 10150: Total Mean reward = 12.582, Mean Reward last 50 = 17.24, Epsilon = 0.01, Best Rewards = 26.0
...
Episode 10650: Total Mean reward = 12.737, Mean Reward last 50 = 16.88, Epsilon = 0.01, Best Rewards = 26.0
Episode 10700: Total Mean reward = 12.751, Mean Reward last 50 = 15.76, Epsilon = 0.01, Best Rewards = 26.0
```

---

### Code Overview

Here is an overview of the main components:

- **Q-Network**:
  Implementation of the Dueling DQN can be found in:

```bash
p1_navigation/dqn_dueling.py
```

- **Prioritized Replay Memory**:
  Code for Prioritized Experience Replay is in:

```bash
p1_navigation/prioritized_replay.py
```

- **Agent Logic**:
  The core agent implementation and training logic are in:

```bash
p1_navigation/Navigation.ipynb
```
