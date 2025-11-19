# MovieLens DQN Playground

Lightweight sandbox for turning the MovieLens “ml-latest-small” dataset into RL-friendly training data. Everything lives under `src/` and can be run as simple Python scripts.

## Repository layout

```
ECE1508RL_project/
├── data/
│   └── ml-latest-small/        # Raw CSVs from MovieLens (ratings, movies, tags, links)
├── src/
│   ├── __init__.py             
│   ├── data_loader.py          # Basic CSV loader/validator
│   ├── data_processor.py       # Turns raw tables into feature matrices
│   ├── data_stats.py           # Quick stats + plots
│   └── transitions.py          # Builds offline DQN transitions
├── .gitignore
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
   pip install pandas numpy matplotlib
   ```