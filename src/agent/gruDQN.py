import torch.nn as nn
import torch.nn.functional as F

class GRU_DQN(nn.Module):
    def __init__(self, num_actions, input_dim, hidden_dim=128, num_layers=1, dropout_rate=0.2):
        super(GRU_DQN, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_q = nn.Linear(hidden_dim, num_actions) 

    def forward(self, state_seq):
        _, hidden = self.gru(state_seq)
        last_hidden = hidden[-1]
        
        x = F.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        q_values = self.fc_q(x)
        
        return q_values