import torch
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
        
        # Q-Head
        # Takes the final hidden state from GRU and predicts Q-values
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_q = nn.Linear(hidden_dim, num_actions) # Output: Q-value for every movie

    def forward(self, state_seq):
        # state_seq shape: [Batch_Size, History_Window, Feature_Dim]
        
        # Pass through GRU
        gru_out, hidden = self.gru(state_seq)
        
        last_hidden = hidden[-1] #[Batch_Size, Hidden_Dim]
        
        # Pass through Q-Network
        x = F.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        q_values = self.fc_q(x)
        
        return q_values