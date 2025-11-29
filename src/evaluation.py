from __future__ import annotations
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

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
except ImportError:
    from .data_loader import MovieLensLoader
    from .data_processor import DatasetPrep
    from .transitions import _item_matrix
    from .agent.dqn_model import DQN
    from .agent.ddqn_dueling_model import DuelingDQN
    from .agent.gruDQN import GRU_DQN
    from .utils.collaborative import collaborative_filtering_recommend
    from .utils.hyperparameters import Hyperparameters

def prepare_evaluation_data(
    data_dir: Path, 
    val_ratio: float, 
    keep_top_n: int, 
    min_ratings: int,
    history_window: int
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
    user_state = build_user_state_vectors(train_df, item_matrix, history_window)
    
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

def eval_prcp(
    train_df_id: pd.DataFrame,
    test_df_id: pd.DataFrame,
    n_items_total: int,
    recommend_func,
    item_features: Dict[int, np.ndarray],
    N: int = 10,
) -> dict[str, float]:
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

        hits = [1 if item in test_items else 0 for item in rec_items]
        rel_hits = sum(hits)
        if rel_hits > 0: hit_users += 1

        metrics['precision'].append(rel_hits / len(rec_items))
        metrics['recall'].append(rel_hits / len(test_items))
        metrics['ndcg'].append(_ndcg(hits, min(len(test_items), len(rec_items))))
        metrics['map'].append(_average_precision(hits, min(len(test_items), len(rec_items), N)))
        metrics['coverage_user'].append(len(set(rec_items)) / float(n_items_total))

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
        "Coverage@User": np.mean(metrics['coverage_user']),
        "Popularity": pop_sum / total_recs,
        "Novelty": novelty_sum / total_recs,
    }
    return results

def build_user_state_vectors(train_df, item_matrix: np.ndarray,history_window: int) -> dict[int, np.ndarray]:
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
        
        #For GRU:    
        # state_vec = np.concatenate(vector_stack).astype(np.float32)
        state_vec = np.array(vector_stack, dtype=np.float32)
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

# --- PLOTTING ---
def _print_metrics_block(title: str, metrics: dict[str, float | dict], indent: int = 0) -> None:
    prefix = " " * indent
    print(f"{prefix}{title}:")
    for key, value in metrics.items():
        print(f"{' ' * (indent + 2)}{key}: {value:.4f}")

def _visualize_metrics(project_root, dqn_metrics, cf_metrics, top_k):
    figures_dir = project_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    metric_names = [f"Precision@{top_k}", f"Recall@{top_k}", f"NDCG@{top_k}", f"MAP@{top_k}"]
    dqn_vals = [float(dqn_metrics.get(name, 0.0)) for name in metric_names]
    cf_vals = [float(cf_metrics.get(name, 0.0)) for name in metric_names]
    
    x = np.arange(len(metric_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, dqn_vals, width, label="DQN", color="#4C72B0")
    ax.bar(x + width/2, cf_vals, width, label="CF", color="#55A868")
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

    # Prepare Data
    data = prepare_evaluation_data(
        data_dir, 
        val_ratio=hp.val_ratio, 
        keep_top_n=hp.keep_top_n, 
        min_ratings=hp.min_ratings,
        history_window=hp.history_window
    )
    
    # Load Model
    device = torch.device(hp.device if torch.cuda.is_available() else "cpu")

    # Determine Dimensions
    single_movie_dim = data['feat_dim']
    flattened_dim = data['feat_dim'] * hp.history_window
    
    if getattr(hp, "use_grudqn", False):
        print("--- Evaluator: Using GRU Architecture ---")
        Net = GRU_DQN
        input_dim = single_movie_dim
    elif getattr(hp, "use_dueling", False): 
        Net = DuelingDQN
        input_dim = flattened_dim
    else: 
        Net = DQN
        input_dim = flattened_dim

    if getattr(hp, "use_dueling", False): Net = DuelingDQN
    else: Net = DQN

    state_input_dim = data['feat_dim'] * hp.history_window
    # dqn = Net(num_actions=data['n_actions'], feature_size=state_input_dim).to(device)
    try:
        dqn = Net(num_actions=data['n_actions'], input_dim=input_dim).to(device)
    except TypeError:
        # Fallback for previous version
        dqn = Net(num_actions=data['n_actions'], feature_size=input_dim).to(device)

    
    if model_path.exists():
        print(f"Loading model from {model_path}")
        dqn.load_state_dict(torch.load(model_path, map_location=device))
        print(dqn)
    else:
        print(f"WARNING: Model path {model_path} does not exist! Using random weights.")

    # Evaluate 
    print(f"Evaluating {'DDQN' if hp.use_ddqn else 'DQN'}...")
    dqn_rec = make_dqn_recommender(dqn, data['user_state'], data['seen_train'], data['movie_ids_by_key'], data['n_actions'], device)
    dqn_metrics = eval_prcp(data['train_df_id'], data['test_df_id'], data['n_actions'], dqn_rec, data['movieid_to_features'], N=top_k)
    
    # Evaluate CF (Baseline)
    print("Evaluating CF...")
    cf_rec = make_cf_recommender(data['train_df_id'])
    cf_metrics = eval_prcp(data['train_df_id'], data['test_df_id'], data['n_actions'], cf_rec, data['movieid_to_features'], N=top_k)
    
    print(f"\n=== Top-{top_k} Results ===")
    _print_metrics_block("DQN", dqn_metrics)
    print("")
    _print_metrics_block("CF (Baseline)", cf_metrics)
    
    if not no_plots:
        _visualize_metrics(PROJECT_ROOT, dqn_metrics, cf_metrics, top_k)
        print("\nPlots saved to reports/figures/")

    return dqn_metrics, cf_metrics

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Offline evaluation using Hyperparameters.")
    parser.add_argument("--model-type", type=str, choices=["base", "finetuned", "custom"], default="base", help="Which config path to use.")
    parser.add_argument("--custom-path", type=str, help="Path if model-type is custom.")
    parser.add_argument("--no-plots", default=False,action="store_true", help="Disable plots.")
    
    args = parser.parse_args()
    
    hp = Hyperparameters()
    
    # Get Model Path
    root = Path(__file__).resolve().parents[1]
    if args.model_type == "finetuned":
        path = root / hp.model_finetuned
    elif args.model_type == "base":
        path = root / hp.model_base
    else:
        path = Path(args.custom_path)
        
    run_evaluation(hp, path, args.no_plots)