import torch.nn as nn


class DuelingDQN(nn.Module):
    def __init__(self, num_actions, input_dim, hidden_dim=256, dropout_rate=0.2):
        super(DuelingDQN, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x):
        features = self.feature_layer(x)

        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals
