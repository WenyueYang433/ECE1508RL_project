
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np       
from utils.replay_buffer import ReplayBuffer
from agent.dqn_model import DQN
from agent.ddqn_dueling_model import DuelingDQN

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
        self.epsilon = 0.0 # Default to Greedy (Offline RL)
        self.loss_list = []
        self.current_loss = 0.0
        self.episode_counts = 0


        self.num_actions = env.n_actions
        feature_size = env.state_dim  

        self.replay_buffer = ReplayBuffer(self.hp.buffer_size)

        if getattr(self.hp, "use_dueling", False):
            print("--- Using Dueling Network Architecture ---")
            Net = DuelingDQN
        else:
            print("--- Using Standard Network Architecture ---")
            Net = DQN
 
        self.onlineDQN = Net(num_actions=self.num_actions,  
                             feature_size=feature_size,
                             hidden_dim=self.hp.hidden_dim, 
                             dropout_rate=self.hp.dropout_rate).to(self.device)
        self.targetDQN =  Net(num_actions=self.num_actions, 
                             feature_size=feature_size,
                             hidden_dim=self.hp.hidden_dim,  
                             dropout_rate=self.hp.dropout_rate).to(self.device)
        self.targetDQN.load_state_dict(self.onlineDQN.state_dict())
        self.targetDQN.eval()
        
        print(self.onlineDQN)

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.onlineDQN.parameters(), 
            lr=self.hp.learning_rate, 
            weight_decay=getattr(self.hp, "weight_decay", 1e-5)
        )

        
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
    
     
    #offline
    def train(self, n_updates, val_data=None, save_path=None):
        """
        Runs the full training loop with periodic evaluation.
        Returns dictionary of history stats for plotting.
        """
        from evaluation import eval_prcp, make_dqn_recommender
        
        print(f"Starting training for {n_updates} steps...")
        history = {"loss": [], "ndcg": [], "q_val": [], "steps": []}
        best_ndcg = 0.0

        for step in range(1, n_updates + 1):
            if len(self.replay_buffer) < self.hp.batch_size:
                continue
            
            self.apply_SGD(ended=(step % self.hp.target_update == 0))
            
            avg_loss = self.current_loss / self.episode_counts if self.episode_counts > 0 else 0
            history["loss"].append(avg_loss)

            if val_data and step % 200 == 0:
                
                #NDCG
                rec_func = make_dqn_recommender(
                    self.onlineDQN, val_data['user_state'], val_data['seen_train'], 
                    val_data['movie_ids_by_key'], val_data['n_actions'], self.device
                )
                metrics = eval_prcp(
                    val_data['test_df_id'], val_data['test_df_id'], 
                    val_data['n_actions'], rec_func, val_data['movieid_to_features'], N=10
                )
                ndcg = metrics['NDCG@10']
                
                # Q-Value Sanity Check
                dummy_keys = list(val_data['user_state'].keys())[:32]
                dummy_states = np.array([val_data['user_state'][k] for k in dummy_keys])
                t_states = torch.tensor(dummy_states, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    avg_q = self.onlineDQN(t_states).mean().item()

                history["ndcg"].append(ndcg)
                history["q_val"].append(avg_q)
                history["steps"].append(step)
                
                print(f"Step {step:4d} | Loss: {avg_loss:.4f} | NDCG@10: {ndcg:.4f} | Avg Q: {avg_q:.3f}")

                # Save Best
                if save_path and ndcg >= best_ndcg:
                    best_ndcg = ndcg
                    torch.save(self.onlineDQN.state_dict(), save_path)
                    print("--- Best Model Saved!")
        
        return history

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
        

        # Q(s, a): current q from Online Network
        Q_all = self.onlineDQN(states)              # [B, num_actions]
        Q_hat = Q_all.gather(1, actions)            # [B, 1]

        # r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            if self.hp.use_ddqn:
                # --- Double DQN (DDQN) ---
                # Online Net selects action, Target Net evaluates it
                next_actions = self.onlineDQN(next_states).argmax(dim=1, keepdim=True)
                Q_next = self.targetDQN(next_states).gather(1, next_actions)
            else:
                # --- Standard DQN ---
                # Target Net does BOTH selection and evaluation
                Q_next_all = self.targetDQN(next_states)
                Q_next, _ = Q_next_all.max(dim=1, keepdim=True)

            # Bellman Equation
            Q_next[terminals] = 0.0
            y = rewards + self.hp.gamma * Q_next

        # MSE loss
        loss = self.loss_function(Q_hat, y)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.onlineDQN.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.current_loss += loss.item()
        self.episode_counts += 1

        # 
        if ended:
            self.targetDQN.load_state_dict(self.onlineDQN.state_dict())

