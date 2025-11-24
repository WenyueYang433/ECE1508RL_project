from __future__ import annotations
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
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
    from agent.dqn_agent import DQN
    from utils.collaborative import collaborative_filtering_recommend
except ImportError:  # pragma: no cover - fallback when executed as package
    from .data_loader import MovieLensLoader
    from .data_processor import DatasetPrep
    from .transitions import _item_matrix
    from .agent.dqn_agent import DQN
    from .utils.collaborative import collaborative_filtering_recommend

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
    ideal_list = [1] * ideal_hits
    ideal_dcg = _dcg(ideal_list)
    if ideal_dcg == 0:
        return 0.0
    return dcg_val / ideal_dcg


def _average_precision(hits: List[int], max_rel: int) -> float:
    if max_rel == 0:
        return 0.0
    cum_hits = 0
    ap = 0.0
    for idx, rel in enumerate(hits, start=1):
        if rel:
            cum_hits += 1
            ap += cum_hits / idx
    return ap / max_rel if max_rel > 0 else 0.0


#Calculates the performance metrics by comparing the agent's recommendations against the movies the user actually watched in the test set
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

    per_user_precision: List[float] = []
    per_user_recall: List[float] = []
    per_user_ndcg: List[float] = []
    per_user_map: List[float] = []
    per_user_coverage_ratio: List[float] = []

    all_rec_items = []
    total_rec_items = 0
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
        if not rec_items:
            continue

        hits = [1 if item in test_items else 0 for item in rec_items]
        rel_hits = sum(hits)
        if rel_hits > 0:
            hit_users += 1

        precision_u = rel_hits / len(rec_items)
        recall_u = rel_hits / len(test_items) if test_items else 0.0
        ideal_hits = min(len(test_items), len(rec_items))
        ndcg_u = _ndcg(hits, ideal_hits)
        ap_u = _average_precision(hits, min(len(test_items), len(rec_items), N))

        per_user_precision.append(precision_u)
        per_user_recall.append(recall_u)
        per_user_ndcg.append(ndcg_u)
        per_user_map.append(ap_u)

        coverage_ratio = len(set(rec_items)) / float(n_items_total)
        per_user_coverage_ratio.append(coverage_ratio)

        all_rec_items.extend(rec_items)
        total_rec_items += len(rec_items)

        for i in rec_items:
            pop_val = pop_dict.get(int(i), 0.0)
            pop_sum += np.log1p(pop_val)
            novelty_sum += -np.log(((pop_val or 0.0) + 1e-9) / (max_pop + 1e-9))

    if total_rec_items == 0:
        return {
            f"Precision@{N}": 0.0,
            f"Recall@{N}": 0.0,
            f"NDCG@{N}": 0.0,
            f"MAP@{N}": 0.0,
            "HitRate": 0.0,
            "Coverage@Catalog": 0.0,
            "Coverage@User": 0.0,
            "Popularity": 0.0,
            "Novelty": 0.0,
        }

    unique_recs = len(set(all_rec_items))
    coverage_catalog = unique_recs / float(n_items_total)

    results = {
        f"Precision@{N}": float(np.mean(per_user_precision)) if per_user_precision else 0.0,
        f"Recall@{N}": float(np.mean(per_user_recall)) if per_user_recall else 0.0,
        f"NDCG@{N}": float(np.mean(per_user_ndcg)) if per_user_ndcg else 0.0,
        f"MAP@{N}": float(np.mean(per_user_map)) if per_user_map else 0.0,
        "HitRate": hit_users / len(per_user_precision) if per_user_precision else 0.0,
        "Coverage@Catalog": coverage_catalog,
        "Coverage@User": float(np.mean(per_user_coverage_ratio)) if per_user_coverage_ratio else 0.0,
        "Popularity": pop_sum / total_rec_items,
        "Novelty": novelty_sum / total_rec_items,
    }

    return results

#Creates state for every user: calculate the average feature vector of all the movies they watched in the training data
def build_user_state_vectors(train_df, item_matrix: np.ndarray) -> dict[int, np.ndarray]:
    user_movies = defaultdict(list)
    for _, row in train_df.iterrows():
        u = int(row["user_key"])
        m = int(row["movie_key"])
        user_movies[u].append(m)

    user_state = {}
    for u, movies in user_movies.items():
        if not movies:
            continue
        vec = item_matrix[movies].mean(axis=0)
        user_state[u] = vec.astype(np.float32)
    return user_state


def make_dqn_recommender( dqn_model: DQN, user_state: dict[int, np.ndarray], seen_train: dict[int, set[int]],movie_ids_by_key: np.ndarray,n_actions: int,device: torch.device):
    dqn_model.eval()
    
    #pass state into DQN, mask out movies they have already seen, and selects the top N movies with the highest Q-values
    def recommend(user_id: int, N: int):
        u = user_id
        if u not in user_state:
            return []

        state_vec = user_state[u]
        state_t = torch.tensor(state_vec, dtype=torch.float32,
                               device=device).unsqueeze(0)
        with torch.no_grad():
            q = dqn_model(state_t).cpu().numpy().reshape(-1)  # [n_actions]
        mask = np.ones_like(q, dtype=bool)
        seen = seen_train.get(u, set())
        if seen:
            idx = np.fromiter(seen, dtype=int)
            mask[idx] = False

        cand_idx = np.where(mask)[0]
        if len(cand_idx) == 0:
            return []

        top_rel = np.argsort(-q[cand_idx])[:N]
        movie_keys = cand_idx[top_rel]
        movie_ids = movie_ids_by_key[movie_keys]
        return list(movie_ids)

    return recommend

def make_cf_recommender(train_df_id: pd.DataFrame):
   
    def recommend(user_id: int, N: int):
        try:
            rec_df = collaborative_filtering_recommend(
                train_df_id, user_id=user_id, n_recs=N
            )
        except ValueError:
            return []

        if rec_df is None or rec_df.empty:
            return []
        return list(rec_df["movieId"].values)

    return recommend


def _print_metrics_block(title: str, metrics: dict[str, float | dict], indent: int = 0) -> None:
    prefix = " " * indent
    print(f"{prefix}{title}:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            _print_metrics_block(key, value, indent + 2)
        else:
            print(f"{' ' * (indent + 2)}{key}: {value:.4f}")


def _plot_metric_bars(metric_names, dqn_metrics, cf_metrics, out_path: Path, title: str, ylabel: str = "") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dqn_vals = [float(dqn_metrics.get(name, 0.0)) for name in metric_names]
    cf_vals = [float(cf_metrics.get(name, 0.0)) for name in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, dqn_vals, width, label="DQN", color="#4C72B0")
    ax.bar(x + width / 2, cf_vals, width, label="CF", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=20, ha="right")
    ax.set_ylabel(ylabel or "Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _visualize_metrics(
    project_root: Path,
    dqn_metrics: dict[str, float],
    cf_metrics: dict[str, float],
    top_k: int,
) -> None:
    figures_dir = project_root / "reports" / "figures"

    ranking_metrics = [f"Precision@{top_k}", f"Recall@{top_k}", f"NDCG@{top_k}", f"MAP@{top_k}"]
    _plot_metric_bars(
        ranking_metrics,
        dqn_metrics,
        cf_metrics,
        figures_dir / f"topk_metrics_k{top_k}.png",
        title=f"Top-{top_k} Ranking Metrics",
    )

    catalog_metrics = ["HitRate", "Coverage@Catalog", "Coverage@User", "Novelty", "Popularity"]
    _plot_metric_bars(
        catalog_metrics,
        dqn_metrics,
        cf_metrics,
        figures_dir / f"catalog_metrics_k{top_k}.png",
        title="Catalog & Diversity Metrics",
        ylabel="Score / Ratio",
    )


def run_evaluation(
    model_path: Path | None = None,
    data_dir: Path | None = None,
    val_ratio: float = 0.1,
    top_k: int = 10,
) -> tuple[dict[str, float], dict[str, float]]:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = data_dir or (project_root / "data" / "ml-latest-small")
    model_path = model_path or (project_root / "models" / "dqn_movielens.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = MovieLensLoader(str(data_dir)).load_all()
    prep = DatasetPrep(loader)
    movie_features = prep.encode_movies()
    ratings = prep.encode_ratings()
    train_df, test_df = prep.temporal_split(ratings, val_ratio=val_ratio)

    mf_sorted = movie_features.sort_values("movie_key")
    movie_ids_by_key = mf_sorted["movieId"].to_numpy()
    n_items_total = len(mf_sorted["movieId"].unique())
    item_matrix, _ = _item_matrix(movie_features)

    movieid_to_features: Dict[int, np.ndarray] = {}
    for _, row in mf_sorted.iterrows():
        movie_id = int(row["movieId"])
        movie_key = int(row["movie_key"])
        if 0 <= movie_key < item_matrix.shape[0]:
            movieid_to_features[movie_id] = item_matrix[movie_key]

    train_df_id = train_df.copy()
    test_df_id = test_df.copy()
    train_df_id["userId"] = train_df_id["user_key"]
    test_df_id["userId"] = test_df_id["user_key"]
    train_df_id["movieId"] = movie_ids_by_key[train_df_id["movie_key"].to_numpy()]
    test_df_id["movieId"] = movie_ids_by_key[test_df_id["movie_key"].to_numpy()]

    user_state = build_user_state_vectors(train_df, item_matrix)
    seen_train = defaultdict(set)
    for _, row in train_df.iterrows():
        seen_train[int(row["user_key"])].add(int(row["movie_key"]))

    n_actions = item_matrix.shape[0]
    feat_dim = item_matrix.shape[1]

    dqn = DQN(num_actions=n_actions, feature_size=feat_dim).to(device)
    state = torch.load(model_path, map_location=device)
    dqn.load_state_dict(state)

    dqn_recommender = make_dqn_recommender(
        dqn_model=dqn,
        user_state=user_state,
        seen_train=seen_train,
        movie_ids_by_key=movie_ids_by_key,
        n_actions=n_actions,
        device=device,
    )
    cf_recommender = make_cf_recommender(train_df_id)

    dqn_metrics = eval_prcp(
        train_df_id=train_df_id,
        test_df_id=test_df_id,
        n_items_total=n_items_total,
        recommend_func=dqn_recommender,
        item_features=movieid_to_features,
        N=top_k,
    )

    cf_metrics = eval_prcp(
        train_df_id=train_df_id,
        test_df_id=test_df_id,
        n_items_total=n_items_total,
        recommend_func=cf_recommender,
        item_features=movieid_to_features,
        N=top_k,
    )

    return dqn_metrics, cf_metrics


def main():
    parser = argparse.ArgumentParser(description="Offline evaluation for MovieLens DQN.")
    parser.add_argument("--model-path", type=Path, default=None, help="Checkpoint to evaluate.")
    parser.add_argument("--data-dir", type=Path, default=None, help="MovieLens data directory.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K cutoff for ranking metrics.")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable saving matplotlib comparison plots.",
    )
    args = parser.parse_args()

    dqn_metrics, cf_metrics = run_evaluation(
        model_path=args.model_path,
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        top_k=args.top_k,
    )

    print(
        f"\n=== Top-{args.top_k} Evaluation (Precision / Recall / Coverage / Popularity / Advanced Metrics) ==="
    )
    _print_metrics_block("DQN", dqn_metrics)
    print()
    _print_metrics_block("CF (User-based collaborative)", cf_metrics)

    if not args.no_plots:
        project_root = Path(__file__).resolve().parents[1]
        _visualize_metrics(project_root, dqn_metrics, cf_metrics, args.top_k)
        print("\nSaved metric comparison plots to reports/figures/.")


if __name__ == "__main__":
    main()