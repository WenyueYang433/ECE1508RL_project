import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.replay_buffer import ReplayBuffer
from agent.dqn_model import DQN
from agent.ddqn_dueling_model import DuelingDQN
from agent.gruDQN import GRU_DQN


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

        self.epsilon = 0.0  # Default to Greedy
        self.loss_list = []
        self.current_loss = 0.0
        self.episode_counts = 0

        self.num_actions = env.n_actions

        if hasattr(env, "item_matrix"):
            # For GRU
            single_movie_dim = env.item_matrix.shape[1]
        else:
            single_movie_dim = env.state_dim // env.history_window
        # For MLP we need total flattened size
        flattened_dim = env.state_dim

        self.replay_buffer = ReplayBuffer(self.hp.buffer_size)

        if self.hp.model_arch == "GRU":
            Net = GRU_DQN
            input_arg = single_movie_dim
        elif self.hp.model_arch == "Dueling":
            Net = DuelingDQN
            input_arg = flattened_dim
        else:
            # Standard MLP
            Net = DQN
            input_arg = flattened_dim

        self.onlineDQN = Net(
            num_actions=self.num_actions,
            input_dim=input_arg,
            hidden_dim=self.hp.hidden_dim,
            dropout_rate=self.hp.dropout_rate,
        ).to(self.device)
        self.targetDQN = Net(
            num_actions=self.num_actions,
            input_dim=input_arg,
            hidden_dim=self.hp.hidden_dim,
            dropout_rate=self.hp.dropout_rate,
        ).to(self.device)
        self.targetDQN.load_state_dict(self.onlineDQN.state_dict())
        self.targetDQN.eval()

        print(self.onlineDQN)

        self.loss_function = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(
            self.onlineDQN.parameters(),
            lr=self.hp.learning_rate,
            weight_decay=getattr(self.hp, "weight_decay", 1e-5),
        )

        self._preload_buffer_from_env()

    def _preload_buffer_from_env(self, split: str = "train"):
        trans = self.env.get_all_transitions(split=split)
        states = trans["state"]
        actions = trans["action"]
        rewards = trans["reward"]
        next_states = trans["next_state"]
        dones = trans["done"]

        n = states.shape[0]
        for i in range(n):
            self.replay_buffer.push(
                s=states[i],
                a=int(actions[i]),
                r=float(rewards[i]),
                s_next=next_states[i],
                d=bool(dones[i]),
            )

        print(
            f"[Agent] Preloaded {len(self.replay_buffer)} transitions into replay buffer."
        )

    def train(self, n_updates, val_data=None, save_path=None):
        """
        Runs the full training loop with periodic evaluation.
        Returns dictionary of history stats for plotting.
        """
        from evaluation import eval_prcp, make_dqn_recommender

        print(f"Starting training for {n_updates} steps...")
        history = {"loss": [], "ndcg": [], "q_val": [], "reward": [], "steps": []}
        best_ndcg = 0.0

        for step in range(1, n_updates + 1):
            if len(self.replay_buffer) < self.hp.batch_size:
                continue

            self.apply_SGD(ended=(step % self.hp.target_update == 0))

            avg_loss = (
                self.current_loss / self.episode_counts
                if self.episode_counts > 0
                else 0
            )
            history["loss"].append(avg_loss)

            if val_data and step % 200 == 0:
                rec_func = make_dqn_recommender(
                    self.onlineDQN,
                    val_data["user_state"],
                    val_data["seen_train"],
                    val_data["movie_ids_by_key"],
                    val_data["n_actions"],
                    self.device,
                )
                metrics = eval_prcp(
                    val_data["test_df_id"],
                    val_data["test_df_id"],
                    val_data["n_actions"],
                    rec_func,
                    val_data["movieid_to_features"],
                    N=10,
                )
                ndcg = metrics["NDCG@10"]
                realized_reward = metrics["AverageReward"]

                dummy_keys = list(val_data["user_state"].keys())[:32]
                dummy_states = np.array([val_data["user_state"][k] for k in dummy_keys])
                t_states = torch.tensor(
                    dummy_states, dtype=torch.float32, device=self.device
                )
                with torch.no_grad():
                    avg_q = self.onlineDQN(t_states).mean().item()

                history["ndcg"].append(ndcg)
                history["q_val"].append(avg_q)
                history["steps"].append(step)
                history["reward"].append(realized_reward)

                print(
                    f"Step {step:4d} | Loss: {avg_loss:.4f} | NDCG@10: {ndcg:.4f} | Avg Q: {avg_q:.3f} | Reward: {realized_reward:.3f}"
                )

                if save_path and ndcg >= best_ndcg:
                    best_ndcg = ndcg
                    torch.save(self.onlineDQN.state_dict(), save_path)
                    print("--- Best Model Saved!")

        return history

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            state_t = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(
                0
            )  # [1, F]
            with torch.no_grad():
                q_values = self.onlineDQN(state_t)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def apply_SGD(self, ended: bool):

        self.onlineDQN.train()
        states, actions, rewards, next_states, terminals = self.replay_buffer.sample(
            self.hp.batch_size
        )

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        next_states = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        terminals = torch.as_tensor(terminals, dtype=torch.bool, device=self.device)

        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        terminals = terminals.unsqueeze(1)

        Q_all = self.onlineDQN(states)
        Q_hat = Q_all.gather(1, actions)

        with torch.no_grad():
            if self.hp.use_double_q:
                next_actions = self.onlineDQN(next_states).argmax(dim=1, keepdim=True)
                Q_next = self.targetDQN(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                Q_next_all = self.targetDQN(next_states)
                Q_next, _ = Q_next_all.max(dim=1, keepdim=True)

            Q_next[terminals] = 0.0
            y = rewards + self.hp.gamma * Q_next

        loss = self.loss_function(Q_hat, y)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.onlineDQN.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.current_loss += loss.item()
        self.episode_counts += 1

        if ended:
            self.targetDQN.load_state_dict(self.onlineDQN.state_dict())
