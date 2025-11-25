# MovieLens DQN Playground

Lightweight sandbox for turning the MovieLens “ml-latest-small” dataset into RL-friendly training data. Everything lives under `src/` and can be run as simple Python scripts.

## Repository layout

```
ECE1508RL_project/
├── data/
│   └── ml-latest-small/        # Raw CSVs from MovieLens (ratings, movies, tags, links)
├── models/                     # Trained DQN weights, logs, and optional plots
├── reports/                    # Generated plots and evaluation figures
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
│   │   └── dqn_model.py        # DQN NN
│   └── utils/
│       ├── collaborative.py    # User-based collaborative filtering (UserCF) baseline
│       ├── hyperparameters.py  # Hyperparameters
│       └── replay_buffer.py    # Simple experience replay buffer for DQN training
└── README.md

```

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
   pip install pandas numpy matplotlib torch
   ```
