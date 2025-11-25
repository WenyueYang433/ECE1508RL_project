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
    

class DuelingDQN(nn.Module):
    def __init__(self, num_actions, feature_size, hidden_dim=128, dropout_rate=0.2):
        super(DuelingDQN, self).__init__()
        # Feature Extractor
        self.fc1 = nn.Linear(feature_size, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.act = nn.ReLU()

        # Value Stream 
        self.val_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.val_out = nn.Linear(hidden_dim // 2, 1)
        # Advantage Stream
        self.adv_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.adv_out = nn.Linear(hidden_dim // 2, num_actions)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        # Value
        v = self.act(self.val_fc(x))
        v = self.dropout(v)
        v = self.val_out(v)
        # Advantage
        a = self.act(self.adv_fc(x))
        a = self.dropout(a)
        a = self.adv_out(a)
        #Q = V + (A - mean(A))
        return v + (a - a.mean(dim=1, keepdim=True))