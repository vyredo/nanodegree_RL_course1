import random
import numpy as np
from collections import deque


class PrioritizedReplayMemory:
    def __init__(self, maxlen, alpha=0.6, seed=None):
        self.memory = deque([], maxlen=maxlen)
        self.priorities = deque([], maxlen=maxlen)
        self.alpha = alpha
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def append(self, transition, priority=1.0):
        """
        Adds a transition to the memory with a specified priority.
        """
        self.memory.append(transition)
        self.priorities.append(priority)

    def sample(self, sample_size, beta=0.4):
        # Calculate sampling probabilities
        scaled_priorities = np.array(self.priorities) ** self.alpha
        sampling_probs = scaled_priorities / sum(scaled_priorities)

        # Sample indices
        indices = np.random.choice(
            len(self.memory), sample_size, p=sampling_probs)
        transitions = [self.memory[i] for i in indices]

        # Importance-sampling weights
        total = len(self.memory)
        weights = (total * sampling_probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability

        return transitions, weights, indices

    def update_priorities(self, indices, priorities):
        """
        Updates priorities for sampled transitions.s
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)
