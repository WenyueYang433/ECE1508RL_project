
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np       
from utils.replay_buffer import ReplayBuffer

class DQN(nn.Module):
    def __init__(self, num_actions, feature_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(feature_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, num_actions)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        q = self.out(x)
        return q


class Agent:
    def __init__(self, env, hyperparameters, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            

        self.env = env
        self.hp = hyperparameters

        # epsilon 
        self.epsilon = 0.99
        self.loss_list = []
        self.current_loss = 0.0
        self.episode_counts = 0


        self.num_actions = env.n_actions
        feature_size = env.state_dim  

        self.replay_buffer = ReplayBuffer(self.hp.buffer_size)


 
        self.onlineDQN = DQN(num_actions=self.num_actions,  feature_size=feature_size).to(self.device)
        self.targetDQN = DQN(num_actions=self.num_actions, feature_size=feature_size).to(self.device)
        self.targetDQN.load_state_dict(self.onlineDQN.state_dict())
        self.targetDQN.eval()

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.onlineDQN.parameters(),    lr=self.hp.learning_rate)

        #
        self._preload_buffer_from_env()

    def _preload_buffer_from_env(self, split: str = "train"):
        trans = self.env.get_all_transitions(split=split)
        states = trans["state"]        # [N, state_dim]
        actions = trans["action"]      # [N]
        rewards = trans["reward"]      # [N]
        next_states = trans["next_state"]
        dones = trans["done"]

        n = states.shape[0]
        for i in range(n):
            self.replay_buffer.push(
                s=states[i],              # state
                a=int(actions[i]),        # action
                r=float(rewards[i]),      # reward
                s_next=next_states[i],    # next_state
                d=bool(dones[i])         # done
            )

        print(f"[Agent] Preloaded {len(self.replay_buffer)} transitions into replay buffer.")
    
     
    def update_epsilon(self):
        # reduce epsilon by the decay factor
        self.epsilon = max(0.01, self.epsilon * self.hp.epsilon_decay)


    #offline
    def train(self, n_updates):
        losses_all = []

        self.current_loss = 0.0
        self.episode_counts = 0

        log_interval = getattr(self.hp, "log_interval", 1000)

        for step in range(1, n_updates + 1):
            if len(self.replay_buffer) < self.hp.batch_size:
                continue

            self.apply_SGD(ended=(step % self.hp.target_update == 0))
            
            self.update_epsilon()

            avg_loss = (self.current_loss / self.episode_counts
                        if self.episode_counts > 0 else 0.0)
            losses_all.append(avg_loss)

            if step % log_interval == 0 or step == 1:
                print(f"Update {step:6d}/{n_updates} | Loss: {avg_loss:.4f}")

        return losses_all

    def select_action(self, state): #epsilon-greedy
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            state_t = torch.tensor(state, dtype=torch.float32,
                                   device=self.device).unsqueeze(0)  # [1, F]
            with torch.no_grad():
                q_values = self.onlineDQN(state_t)  # [1, num_actions]
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def apply_SGD(self, ended: bool):
        states, actions, rewards, next_states, terminals = \
            self.replay_buffer.sample(self.hp.batch_size)

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)       # [B, F]
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)       # [B]
        next_states = torch.as_tensor(next_states, dtype=torch.float32,
                                      device=self.device)                              # [B, F]
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)    # [B]
        terminals = torch.as_tensor(terminals, dtype=torch.bool, device=self.device)   # [B]

        actions   = actions.unsqueeze(1)      # [B, 1]
        rewards   = rewards.unsqueeze(1)      # [B, 1]
        terminals = terminals.unsqueeze(1)    # [B, 1] bool
        

        # Q(s, a)
        Q_all = self.onlineDQN(states)              # [B, num_actions]
        Q_hat = Q_all.gather(1, actions)            # [B, 1]

        # r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            Q_next_all = self.targetDQN(next_states)                     # [B, num_actions]
            next_target_q_value, _ = Q_next_all.max(dim=1, keepdim=True)  # [B, 1]
            next_target_q_value[terminals] = 0.0

            y = rewards + self.hp.gamma * next_target_q_value           # [B, 1]

        # MSE loss
        loss = self.loss_function(Q_hat, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.current_loss += loss.item()
        self.episode_counts += 1

        # 
        if ended:
            self.targetDQN.load_state_dict(self.onlineDQN.state_dict())

