# MovieLens DQN Recommender

Traditional approaches such as random selection or collaborative filtering treat recommendation as a one-shot prediction problem, optimizing only the next item and ignoring long-term user engagement. 

In this project, we instead formulate movie recommendation as a **sequential decision-making** problem and build a **Deep Q-Learning (DQN)**–based system. By modeling user–system interactions as a **Markov Decision Process (MDP)**, the agent learns to maximize the discounted cumulative reward, enabling long-term engagement and adaptive, personalized recommendations.


## Repository Layout

```
ECE1508RL_project/
├── data/
│   └── ml-latest-small/        # Raw CSVs from MovieLens (ratings, movies, tags, links)
├── models/                     # Trained DQN models
├── reports/                    # All logs, generated plots and evaluation figures
├── src/
│   ├── __init__.py   
│   ├── candidate_re_rank.py    # Logic for re-ranking candidates          
│   ├── data_loader.py          # Basic CSV loader/validator
│   ├── data_processor.py       # Turns raw tables into feature matrices
│   ├── data_stats.py           # Quick stats + plots
│   ├── transitions.py          # Builds offline DQN transitions
│   ├── fine_tune_candidate.py  # Script for fine-tuning the model with hard negatives
│   ├── evaluation.py           #  Model evaluation
│   ├── main.py                 #  Train the DQN and save it 
│   ├── env/
│   │   └── recommender_env.py  # Offline MovieLens environment
│   ├── agent/
│   │   ├── dqn_agent.py        # RL Agent logic (training loop, action selection)
│   │   ├── ddqn_dueling_model.py # ddqn_dueling NN
│   │   └── dqn_model.py        # DQN NN
│   └── utils/
│       ├── collaborative.py    # User-based collaborative filtering (UserCF) baseline
│       ├── random_rec.py       # Random recommender baseline
│       ├── hyperparameters.py  # Hyperparameters
│       └── replay_buffer.py    # Simple experience replay buffer for DQN training
└── README.md

```

## Models and Baselines

We implemented the following methods and agents. Detailed implementations can be found in `src/agent` and `src/utils/collaborative.py`.

### Baselines

- **Random Recommender**: Shuffles the movie dataset and randomly selects a list of movies

- **Collaborative Filtering (CF)**: Uses user–user similarity based on historical ratings to generate recommendations

- **Standard DQN**: Our base RL benchmark using a fully connected network with a flattened sliding-window state

---

### Advanced RL Agents

- **Dueling DQN**: Splits the Q-network into separate Value (V(s)) and Advantage (A(s, a)) streams to improve learning efficiency

- **Double DQN (DDQN)**: Reduces Q-value overestimation by decoupling action selection and target evaluation

- **GRU-DQN**: Uses a GRU encoder to model temporal dependencies in the user’s viewing history

- **GRU-DDQN**: Combines GRU-based sequence encoding with the stability benefits of the Double DQN objective


## Setup

1. **Download data**  
   Place the MovieLens “ml-latest-small” folder under `data/` so the CSVs are available at `data/ml-latest-small/*.csv`.

2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### 1. Configure Settings
All model settings are managed in **`src/utils/hyperparameters.py`**.
- Set `self.model_arch` to `"MLP"`, `"Dueling"`, or `"GRU"`.
- Set `self.use_double_q` to `True` for Double DQN (DDQN).

### 2. Train the Base Model
Run the main script to train the agent on user ratings
```bash
python src/main.py
````

  * **Output:** Saves a timestamped model to `models/` (e.g., `GRU_DDQN_20251205_220227.pt`), a training log with all training information to `reports`, (e.g.`\training_log_20251205_220227.log` a training plot to `reports/figures/`.


### 3. Fine-Tune 

Fine tune the model with:

```bash
# Uses the trained base model as a starting point
python src/fine_tune_candidate.py --model-path "models/MLP_DDQN_20251201_134722.pt"
```

  * **Output:** Saves a new model ending in `_finetuned.pt` (e.g., `GRU_DDQN_20251205_220227_finetuned`), a finetunning log with all detail information to `reports`, (e.g.`fine_tune_log_20251205_221032.log`)


### 4. Evaluate a Model

Evaluate the trained model against the baselines using the following metrics (covers four key dimensions: Accuracy, Ranking Quality, Catalog Health, and Diversity):

* **Accuracy & Ranking:** We use **HitRate@K** (to measure successful retrieval) and **NDCG@K** (Normalized Discounted Cumulative Gain) to assess the position-aware quality of the recommendations.
* **Diversity & Health:** We monitor **Catalog Coverage**, **Novelty**, and **Intra-List Diversity (ILD)** to ensure the model avoids popularity bias and provides varied content.
* **Learning Performance:** We also track the **Average Cumulative Reward** to verify the agent's learning convergence over time.

To run the evaluation, replace the filename with the actual saved model path:

```bash
# Replace the filename with the actual saved model path
python src/evaluation.py --model-type custom --custom-path "models/MLP_DDQN_20251201_134722.pt"
```

  * **Output:** a evalaution log with all detail information to `reports`, (e.g.`evaluation_log_20251205_220602.log`)
