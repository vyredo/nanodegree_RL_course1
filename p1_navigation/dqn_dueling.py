import torch
from torch import nn
import torch.nn.functional as F


class DQNDueling(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQNDueling, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)

        # value stream
        self.fc_value = nn.Linear(hidden_dim, 256)
        self.value = nn.Linear(256, 1)

        # Advantage stream
        self.fc_advantages = nn.Linear(hidden_dim, 256)
        self.advantages = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        # Value calc
        v = F.relu(self.fc_value(x))
        V = self.value(v)

        # Advantages calc
        a = F.relu(self.fc_advantages(x))
        A = self.advantages(a)

        # Calc !Q
        Q = V + A - torch.mean(A, dim=1, keepdim=True)

        return Q
