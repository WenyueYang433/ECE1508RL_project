from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from data_loader import MovieLensLoader
    from data_processor import DatasetPrep
    from transitions import _item_matrix
    from agent.dqn_model import DQN
    from agent.ddqn_dueling_model import DuelingDQN
    from agent.gruDQN import GRU_DQN
    from utils.collaborative import collaborative_filtering_recommend
    from utils.hyperparameters import Hyperparameters
    from utils.random_rec import random_recommend
    from utils.logger import Logger
except ImportError:
    from .data_loader import MovieLensLoader
    from .data_processor import DatasetPrep
    from .transitions import _item_matrix
    from .agent.dqn_model import DQN
    from .agent.ddqn_dueling_model import DuelingDQN
    from .agent.gruDQN import GRU_DQN
    from .utils.collaborative import collaborative_filtering_recommend
    from .utils.hyperparameters import Hyperparameters
    from .utils.random_rec import random_recommend
    from .utils.logger import Logger
    
def prepare_evaluation_data(
    data_dir: Path, 
    val_ratio: float, 
    keep_top_n: int, 
    min_ratings: int,
    history_window: int,
    is_gru: bool = False
) -> Dict[str, Any]:
    """
    Loads data using settings from Hyperparameters.
    """
    print(f"--- Loading Data (Top {keep_top_n} movies, Min {min_ratings} ratings) ---")
    loader = MovieLensLoader(str(data_dir)).load_all()
    prep = DatasetPrep(loader)
    
    # Filter Movies & Users
    movie_features = prep.encode_movies(keep_top_n=keep_top_n)
    prep.encode_users(min_ratings=min_ratings) # cold start fix
    ratings = prep.encode_ratings()
    
    # Train/Test Split
    train_df, test_df = prep.temporal_split(ratings, val_ratio=val_ratio)
    
    # Build ID Mappings (Internal ID -> External MovieID)
    mf_sorted = movie_features.sort_values("movie_key")
    movie_ids_by_key = mf_sorted["movieId"].to_numpy()
    key_to_movieid = dict(zip(mf_sorted["movie_key"].to_numpy(), mf_sorted["movieId"].to_numpy()))
    
    # DF with external IDs (for evaluation functions)
    train_df_id = train_df.copy()
    train_df_id["userId"] = train_df_id["user_key"]
    train_df_id["movieId"] = train_df_id["movie_key"].map(key_to_movieid)
    
    test_df_id = test_df.copy()
    test_df_id["userId"] = test_df_id["user_key"]
    test_df_id["movieId"] = test_df_id["movie_key"].map(key_to_movieid)
    
    # Build User State & History (Seen)
    item_matrix, _ = _item_matrix(movie_features)
    user_state = build_user_state_vectors(train_df, item_matrix, history_window,is_gru=is_gru)
    
    seen_train = defaultdict(set)
    for _, row in train_df.iterrows():
        seen_train[int(row["user_key"])].add(int(row["movie_key"]))
        
    # Item Features Map (for Diversity metrics)
    movieid_to_features = {}
    for _, row in mf_sorted.iterrows():
        k = int(row["movie_key"])
        if 0 <= k < item_matrix.shape[0]:
            movieid_to_features[int(row["movieId"])] = item_matrix[k]

    return {
        "train_df": train_df,
        "test_df": test_df,
        "train_df_id": train_df_id, 
        "test_df_id": test_df_id,   
        "user_state": user_state,
        "seen_train": seen_train,
        "movie_ids_by_key": movie_ids_by_key,
        "item_matrix": item_matrix,
        "movieid_to_features": movieid_to_features,
        "n_actions": item_matrix.shape[0],
        "feat_dim": item_matrix.shape[1]
    }
    
def _dcg(relevances: List[int]) -> float:
    if not relevances: 
        return 0.0
    gains = np.array(relevances, dtype=np.float32)
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(gains * discounts))

def _ndcg(hits: List[int], ideal_hits: int) -> float:
    if ideal_hits == 0: 
        return 0.0
    dcg_val = _dcg(hits)
    ideal_dcg = _dcg([1] * ideal_hits)
    return dcg_val / ideal_dcg if ideal_dcg > 0 else 0.0

def _average_precision(hits: List[int], max_rel: int) -> float:
    if max_rel == 0: 
        return 0.0
    cum_hits = 0
    ap = 0.0
    for idx, rel in enumerate(hits, start=1):
        if rel:
            cum_hits += 1
            ap += cum_hits / idx
    return ap / max_rel

def _intra_list_diversity(rec_items: List[int], item_features: Dict[int, np.ndarray]) -> float:
    """
    Calculates the average Cosine Distance between all pairs of items in a recommendation list.
    Higher ILD = More diverse (e.g., [Action, Romance, Sci-Fi])
    Lower ILD = Less diverse (e.g., [Action, Action, Action])
    """
    if len(rec_items) < 2:
        return 0.0
        
    # feature vectors for the recommended items
    features = []
    for mid in rec_items:
        # Safety check: ensure we have features for this movie
        if mid in item_features:
            features.append(item_features[mid])
            
    if len(features) < 2:
        return 0.0
    
    features = np.array(features, dtype=np.float32)
    
    # Normalize features (L2 norm) for Cosine Similarity calculation
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0 # Avoid division by zero
    normalized_features = features / norms
    
    #  Similarity Matrix 
    sim_matrix = normalized_features @ normalized_features.T
    

    dist_matrix = 1.0 - sim_matrix
    
    n = features.shape[0]
    upper_indices = np.triu_indices(n, k=1)
    
    avg_dist = np.mean(dist_matrix[upper_indices])
    
    return float(avg_dist)

def eval_prcp(
    train_df_id: pd.DataFrame,
    test_df_id: pd.DataFrame,
    n_items_total: int,
    recommend_func,
    item_features: Dict[int, np.ndarray],
    N: int = 10,
) -> dict[str, float]:
    
    #user actual rating
    rating_lookup = dict(zip(zip(test_df_id["userId"], test_df_id["movieId"]), test_df_id["rating"]))
    
    
    item_pop = train_df_id.groupby("movieId")["userId"].nunique()
    pop_dict = item_pop.to_dict()
    max_pop = max(pop_dict.values()) if pop_dict else 1.0

    users = test_df_id["userId"].unique()
    
    metrics = defaultdict(list)
    all_rec_items = []
    pop_sum = 0.0
    novelty_sum = 0.0
    hit_users = 0

    for u in users:
        test_items = set(test_df_id.loc[test_df_id["userId"] == u, "movieId"])
        if not test_items: 
            continue

        rec_items = recommend_func(u, N)
        if not rec_items: 
            continue
        rec_items = list(dict.fromkeys(rec_items))
        
        # cumulative reward
        user_cum_reward = 0.0
        hits = []
        for item in rec_items:
            if item in test_items:
                hits.append(1)
                rating_val = rating_lookup.get((u, item), 3.0) # default to 3.0 (neutral) if error
                reward_val = (rating_val - 3.0) / 2.0
                user_cum_reward += reward_val
            else:
                hits.append(0)

        metrics['cum_reward'].append(user_cum_reward)

        # hits = [1 if item in test_items else 0 for item in rec_items]
        rel_hits = sum(hits)
        if rel_hits > 0: hit_users += 1

        metrics['precision'].append(rel_hits / len(rec_items))
        metrics['recall'].append(rel_hits / len(test_items))
        metrics['ndcg'].append(_ndcg(hits, min(len(test_items), len(rec_items))))
        metrics['map'].append(_average_precision(hits, min(len(test_items), len(rec_items), N)))
        metrics['coverage_user'].append(len(set(rec_items)) / float(n_items_total))

        ild_score = _intra_list_diversity(rec_items, item_features)
        metrics['ild'].append(ild_score)
        
        all_rec_items.extend(rec_items)
        
        for i in rec_items:
            pop_val = pop_dict.get(int(i), 0.0)
            pop_sum += np.log1p(pop_val)
            novelty_sum += -np.log(((pop_val or 0.0) + 1e-9) / (max_pop + 1e-9))

    total_recs = len(all_rec_items)
    if total_recs == 0:
        return {"NDCG@10": 0.0, "Precision@10": 0.0, "Coverage@Catalog": 0.0}

    results = {
        f"Precision@{N}": np.mean(metrics['precision']),
        f"Recall@{N}": np.mean(metrics['recall']),
        f"NDCG@{N}": np.mean(metrics['ndcg']),
        f"MAP@{N}": np.mean(metrics['map']),
        "HitRate": hit_users / len(metrics['precision']) if metrics['precision'] else 0.0,
        "Coverage@Catalog": len(set(all_rec_items)) / float(n_items_total),
        "Coverage@User": np.mean(metrics['coverage_user']), #  this metrics is  no longer in use, it's  mathematically forced to be 0.01.
        "Popularity": pop_sum / total_recs,
        "Novelty": novelty_sum / total_recs,
        "Diversity(ILD)": np.mean(metrics['ild']),
        "AverageReward": np.mean(metrics['cum_reward'])
    }
    return results

def analyze_q_distribution(agent, data, hp, project_root):
    """
    Plots Q-values for Popular vs. Unpopular items to see the effect of the penalty.
    """
    print("\n--- Analyzing Q-Value Distribution ---")
    
    # 1. Identify Popular vs Niche items
    # (We use the same logic as your transitions.py)
    counts = data['train_df_id']['movieId'].value_counts()
    top_200 = set(counts.index[:200]) # Top 20% = Popular
    
    pop_q_values = []
    niche_q_values = []
    
    # 2. Sample 100 users
    test_users = list(data['user_state'].keys())[:100]
    
    device = next(agent.parameters()).device
    
    agent.eval()
    with torch.no_grad():
        for u in test_users:
            state = torch.tensor(data['user_state'][u], dtype=torch.float32).unsqueeze(0).to(device)
            q_values = agent(state).cpu().numpy().reshape(-1)
            
            for key_idx, q in enumerate(q_values):
                # Convert internal key -> external movieId
                movie_id = data['movie_ids_by_key'][key_idx]
                
                if movie_id in top_200:
                    pop_q_values.append(q)
                else:
                    niche_q_values.append(q)

    # 3. Plot Histograms
    plt.figure(figsize=(10, 6))
    plt.hist(pop_q_values, bins=50, alpha=0.6, label='Popular Items (Top 200)', density=True, color='red')
    plt.hist(niche_q_values, bins=50, alpha=0.6, label='Niche Items (Rest)', density=True, color='blue')
    
    plt.title(f"Q-Value Distribution (Penalty={hp.popularity_penalty})")
    plt.xlabel("Predicted Q-Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = project_root / "reports" / "figures" / f"q_dist_penalty_{hp.popularity_penalty}.png"
    plt.savefig(save_path)
    print(f"Q-Value plot saved to: {save_path}")
    plt.close()
    
    
def build_user_state_vectors(train_df, item_matrix: np.ndarray,
                             history_window: int, is_gru: bool = False) -> dict[int, np.ndarray]:
    user_movies = defaultdict(list)
    # Ensure sequential order
    train_df_sorted = train_df.sort_values(["user_key", "timestamp"])
    
    for _, row in train_df_sorted.iterrows():
        user_movies[int(row["user_key"])].append(int(row["movie_key"]))

    feat_dim = item_matrix.shape[1]
    empty_feat = np.zeros(feat_dim, dtype=np.float32)
    
    user_state = {}
    for u, movies in user_movies.items():
        
        # get the last N movie
        history_keys = movies[-history_window:]
        padding_needed = history_window - len(history_keys)
        
        vector_stack = []
        
        # add padding to front
        for _ in range(padding_needed):
            vector_stack.append(empty_feat)
            
        # add movie
        for key in history_keys:
            vector_stack.append(item_matrix[key])
        
        if is_gru:
            # (window, features)
            state_vec = np.array(vector_stack, dtype=np.float32)
        else:
            # flattened
            state_vec = np.concatenate(vector_stack).astype(np.float32)
            
        user_state[u] = state_vec
    return user_state

def make_dqn_recommender(dqn_model: DQN, user_state, seen_train, movie_ids_by_key, n_actions, device):
    dqn_model.eval()
    def recommend(user_id: int, N: int):
        u = user_id
        if u not in user_state: return []
        state_vec = user_state[u]
        state_t = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q = dqn_model(state_t).cpu().numpy().reshape(-1)
        mask = np.ones_like(q, dtype=bool)
        seen = seen_train.get(u, set())
        if seen:
            mask[list(seen)] = False # Mask watched
        cand_idx = np.where(mask)[0]
        if len(cand_idx) == 0: return []
        top_rel = np.argsort(-q[cand_idx])[:N]
        return list(movie_ids_by_key[cand_idx[top_rel]])
    return recommend

def make_cf_recommender(train_df_id: pd.DataFrame):
    def recommend(user_id: int, N: int):
        try:
            rec_df = collaborative_filtering_recommend(train_df_id, user_id=user_id, n_recs=N)
            return list(rec_df["movieId"].values) if (rec_df is not None and not rec_df.empty) else []
        except ValueError:
            return []
    return recommend

def make_random_recommender(train_df_id: pd.DataFrame, seed: int = 42):
    def recommend(user_id: int, N: int):
        rec_df = random_recommend(train_df_id, user_id=user_id, n_recs=N, seed=seed)
        return list(rec_df["movieId"].values) if (rec_df is not None and not rec_df.empty) else []
    return recommend

# --- PLOTTING ---
def _print_metrics_block(title: str, metrics: dict[str, float | dict], indent: int = 0) -> None:
    prefix = " " * indent
    print(f"{prefix}{title}:")
    for key, value in metrics.items():
        print(f"{' ' * (indent + 2)}{key}: {value:.4f}")

def _visualize_metrics(project_root, dqn_metrics, cf_metrics, rnd_metrics, top_k):
    figures_dir = project_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    metric_names = [f"Precision@{top_k}", f"Recall@{top_k}", f"NDCG@{top_k}", f"MAP@{top_k}"]
    dqn_vals = [float(dqn_metrics.get(name, 0.0)) for name in metric_names]
    cf_vals = [float(cf_metrics.get(name, 0.0)) for name in metric_names]
    rnd_vals = [float(rnd_metrics.get(name, 0.0)) for name in metric_names]

    x = np.arange(len(metric_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, dqn_vals, width, label="DQN", color="#4C72B0")
    ax.bar(x + width/2, cf_vals, width, label="CF", color="#55A868")
    ax.bar(x - width, rnd_vals, width, label="Random", color="#95a5a6")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=20, ha="right")
    ax.set_title(f"Top-{top_k} Ranking Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / f"topk_metrics_k{top_k}.png", dpi=200)
    plt.close(fig)

def run_evaluation(hp: Hyperparameters, model_path: Path = None, no_plots: bool = False, top_k: int = 10):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    
    # Resolve Paths
    data_dir = PROJECT_ROOT / hp.data_rel_path
    if model_path is None:
        model_path = PROJECT_ROOT / hp.model_base # Default to Base model

    is_gru = (hp.model_arch == "GRU")
    # Prepare Data
    data = prepare_evaluation_data(
        data_dir, 
        val_ratio=hp.val_ratio, 
        keep_top_n=hp.keep_top_n, 
        min_ratings=hp.min_ratings,
        history_window=hp.history_window,
        is_gru=is_gru
    )
    
    # Load Model
    device = torch.device(hp.device if torch.cuda.is_available() else "cpu")

    # Determine Dimensions
    single_movie_dim = data['feat_dim']
    flattened_dim = data['feat_dim'] * hp.history_window

    if hp.model_arch == "GRU":
        Net = GRU_DQN
        input_dim = single_movie_dim
    elif hp.model_arch == "Dueling":
        Net = DuelingDQN
        input_dim = flattened_dim
    else:
        Net = DQN
        input_dim = flattened_dim

    # Instantiate model; try common constructor signatures (input_dim vs feature_size)
    try:
        dqn = Net(
            num_actions=data['n_actions'], 
            input_dim=input_dim, 
            hidden_dim=hp.hidden_dim, 
            dropout_rate=hp.dropout_rate
        ).to(device)
    except TypeError:
        dqn = Net(
            num_actions=data['n_actions'], 
            feature_size=input_dim, 
            hidden_dim=hp.hidden_dim, 
            dropout_rate=hp.dropout_rate
        ).to(device)

    
    if model_path.exists():
        print(f"----Loading model from {model_path}-----")
        dqn.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
        print(dqn)
    else:
        print(f"Error: Model path {model_path} does not exist!")
        return None, None, None
    

    algo_name = "DDQN" if hp.use_double_q else "DQN"
    print(f"Evaluating {algo_name}...")
    
    dqn_rec = make_dqn_recommender(dqn, data['user_state'], data['seen_train'], data['movie_ids_by_key'], data['n_actions'], device)
    dqn_metrics = eval_prcp(data['train_df_id'], data['test_df_id'], data['n_actions'], dqn_rec, data['movieid_to_features'], N=top_k)
    
    # Evaluate CF (Baseline)
    print("Evaluating CF...")
    cf_rec = make_cf_recommender(data['train_df_id'])
    cf_metrics = eval_prcp(data['train_df_id'], data['test_df_id'], data['n_actions'], cf_rec, data['movieid_to_features'], N=top_k)
    
    # Evaluating  random rec
    print("Evaluating Random Baseline...")
    rnd_rec = make_random_recommender(data['train_df_id'])
    rnd_metrics = eval_prcp(data['train_df_id'], data['test_df_id'], data['n_actions'], rnd_rec, data['movieid_to_features'], N=top_k)

    print(f"\n=== Top-{top_k} Results Comparison ===")
    _print_metrics_block("Random (Lower Bound)", rnd_metrics)
    print("")
    _print_metrics_block("CF (Strong Baseline)", cf_metrics)
    print("")
    _print_metrics_block("DQN (Agent)", dqn_metrics)
    
    if not no_plots:
        _visualize_metrics(PROJECT_ROOT, dqn_metrics, cf_metrics, rnd_metrics, top_k)
        
        # for diversity visualization 
        # analyze_q_distribution(dqn, data, hp, PROJECT_ROOT)
        
        print("\nPlots saved to reports/figures/")

    return dqn_metrics, cf_metrics, rnd_metrics

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Offline evaluation using Hyperparameters.")
    parser.add_argument("--model-type", type=str, choices=["base", "finetuned", "custom"], default="base", help="Which config path to use.")
    parser.add_argument("--custom-path", type=str, help="Path if model-type is custom.")
    parser.add_argument("--no-plots", default=False,action="store_true", help="Disable plots.")
    
    args = parser.parse_args()
    
    hp = Hyperparameters()
    
    # Get Model Path
    root = Path(__file__).resolve().parents[1]
    
    log_dir = root / "reports"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Generate timestamped filename: evaluation_log_YYYYMMDD_HHMMSS.log
    log_filename = f"evaluation_log_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_filepath = log_dir / log_filename
    
    # Redirect print output to both console and file
    sys.stdout = Logger(log_filepath)
    print(f"--- Evaluation Log Started: {log_filename} ---")
    
    if args.model_type == "finetuned":
        path = root / hp.model_finetuned
    elif args.model_type == "base":
        path = root / hp.model_base
    else:
        path = Path(args.custom_path)
        
    run_evaluation(hp, path, args.no_plots)