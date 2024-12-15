# Reinforcement Learning Nanodegree - Banana Unity Environment Report

## Pretrained execution

Below is the youtube video of running pretrained model.
<a href="https://www.youtube.com/watch?v=SCSqUcMtb_k">
<img src="https://github.com/vyredo/nanodegree_RL_course1/blob/main/screenshot.jpg"/>
</a>

- You can open file `Navigation_with_pretrained.ipynb` to run it.
- pretrained model is located at [`runs/Banana_Linux.pt`](https://github.com/vyredo/nanodegree_RL_course1/blob/main/p1_navigation/runs/Banana_Linux.pt)

From the video we can see that the score for each episode are above 13

```
Episode 1	Score: 14.0
Episode 2	Score: 20.0
Episode 3	Score: 17.0
Episode 4	Score: 17.0
Episode 5	Score: 17.0
Average score over 5 episodes: 17.0
```

## Learning Algorithm

The agent is implemented using:

1. **Dueling Deep Q-Network (DQN)**:

   - The Dueling DQN separates the Q-value into two components:
     - **Value Stream**: Estimates the value of being in a given state.
     - **Advantage Stream**: Estimates the relative importance of each action in a given state.

2. **Prioritized Experience Replay**:
   - Instead of sampling experiences randomly, the agent prioritizes experiences with higher temporal-difference errors (i.e., larger prediction errors).
   - The prioritization is achieved using a **priority replay buffer** with a sampling probability

## Log of Rewards

For the full log, check [this link](https://github.com/vyredo/nanodegree_RL_course1/blob/main/p1_navigation/runs/Banana_Linux.log)

After training over **2400 episodes** the `mean score reach 13`. But the mean score still fluctuates
See log below

```
Episode 2550: Total Mean reward = 8.075264602116818, Mean Reward last 50 = 11.38, Epsilon = 0.01, best_rewards = 24.0
Episode 2600: Total Mean reward = 8.125720876585929, Mean Reward last 50 = 10.7, Epsilon = 0.01, best_rewards = 24.0
Episode 2650: Total Mean reward = 8.225575254620898, Mean Reward last 50 = 13.42, Epsilon = 0.01, best_rewards = 24.0
Episode 2700: Total Mean reward = 8.27360236949278, Mean Reward last 50 = 10.82, Epsilon = 0.01, best_rewards = 24.0
Episode 2750: Total Mean reward = 8.342420937840785, Mean Reward last 50 = 12.06, Epsilon = 0.01, best_rewards = 24.0
```

The score stabilized and consistently reached **13** after **3750 episodes**, as shown in the log below:

```
Episode 3700: Total Mean reward = 9.065117535801134, Mean Reward last 50 = 12.58, Epsilon = 0.01, best_rewards = 24.0
Episode 3750: Total Mean reward = 9.119168221807518, Mean Reward last 50 = 13.12, Epsilon = 0.01, best_rewards = 24.0
Episode 3800: Total Mean reward = 9.158379373848987, Mean Reward last 50 = 12.1, Epsilon = 0.01, best_rewards = 24.0
Episode 3850: Total Mean reward = 9.229291093222539, Mean Reward last 50 = 14.62, Epsilon = 0.01, best_rewards = 24.0
Episode 3900: Total Mean reward = 9.288387592924892, Mean Reward last 50 = 13.84, Epsilon = 0.01, best_rewards = 24.0
Episode 3950: Total Mean reward = 9.358643381422425, Mean Reward last 50 = 14.84, Epsilon = 0.01, best_rewards = 24.0
Episode 4000: Total Mean reward = 9.414896275931017, Mean Reward last 50 = 13.86, Epsilon = 0.01, best_rewards = 24.0
```

The whole training is done for **10000 episodes**.

## Plot of Rewards

This are the plot for 10000 episodes

<img src="https://github.com/vyredo/nanodegree_RL_course1/blob/main/p1_navigation/runs/Banana_Linux.png" alt="Reward Plot" />
