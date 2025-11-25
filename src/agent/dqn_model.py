import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, num_actions, feature_size, hidden_dim=128, dropout_rate=0.2):
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(feature_size, hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        
        self.fc3 = nn.Linear(hidden_dim//2, hidden_dim//4)
        
        self.out = nn.Linear(hidden_dim//4, num_actions)
        
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x) 
        

        x = self.act(self.fc2(x))
        x = self.dropout(x)   
        
        x = self.act(self.fc3(x))
        
        q = self.out(x)
        
        return q