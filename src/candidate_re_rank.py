"""Candidate generation + DQN re-ranking evaluation.

Run from repo root:
  python -m src.candidate_re_rank

This script:
- builds candidate pools (global-popular and CF per-user)
- loads saved DQN (`models/dqn_movielens.pt`) if present
- re-ranks candidates with the DQN and evaluates Precision/Recall/Coverage/Popularity
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch

from data_loader import MovieLensLoader
from data_processor import DatasetPrep
from transitions import _item_matrix
# from agent.dqn_agent import DQN
from agent.dqn_model  import DQN
from utils.collaborative import collaborative_filtering_recommend
from evaluation import build_user_state_vectors, eval_prcp, make_cf_recommender


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "dqn_movielens.pt"


def build_global_popular(train_df, K=200):
    # train_df uses movie_key
    pop = train_df.groupby("movie_key")["user_key"].nunique().sort_values(ascending=False)
    return list(pop.index[:K])


def build_cf_candidates(train_df_id, K=200):
    # train_df_id has columns userId, movieId, rating
    candidates = {}
    users = train_df_id["userId"].unique()
    for u in users:
        try:
            recs = collaborative_filtering_recommend(train_df_id, user_id=int(u), n_recs=K)
        except Exception:
            recs = None
        if recs is None or recs.empty:
            continue
        candidates[int(u)] = list(recs["movieId"].values)
    return candidates


def load_dqn(device, n_actions, feat_dim):
    if not MODEL_PATH.exists():
        print(f"No model at {MODEL_PATH}; re-ranker will not use DQN.")
        return None
    dqn = DQN(num_actions=n_actions, feature_size=feat_dim).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    dqn.load_state_dict(state)
    dqn.eval()
    return dqn


def dqn_reranker(dqn, user_state, seen_train, movie_ids_by_key, candidates_by_user, device, N=10):
    # returns a recommend(user_id, N) function that uses candidate keys
    def recommend(user_id: int, N_local: int = N):
        u = int(user_id)
        if u not in user_state:
            return []
        state_vec = user_state[u]
        state_t = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_all = dqn(state_t).cpu().numpy().reshape(-1)

        # get candidate movie_keys: prefer per-user candidates, else global
        cand_keys = candidates_by_user.get(u, candidates_by_user.get("global", []))
        # map seen filter
        seen = seen_train.get(u, set())
        cand_keys = [k for k in cand_keys if k not in seen]
        if not cand_keys:
            return []
        scores = np.array([float(q_all[int(k)]) if 0 <= int(k) < len(q_all) else -np.inf for k in cand_keys])
        top_idx = np.argsort(-scores)[:N_local]
        picked_keys = [cand_keys[i] for i in top_idx]
        return list(movie_ids_by_key[picked_keys])

    return recommend


def main():
    data_dir = PROJECT_ROOT / "data" / "ml-latest-small"
    loader = MovieLensLoader(str(data_dir)).load_all()
    prep = DatasetPrep(loader)

    # keep top 1000 to match training
    movie_features = prep.encode_movies(keep_top_n=1000)
    ratings = prep.encode_ratings()
    train_df, test_df = prep.temporal_split(ratings, val_ratio=0.1)

    # mapping
    mf_sorted = movie_features.sort_values("movie_key")
    movie_ids_by_key = mf_sorted["movieId"].to_numpy()

    # build train_df_id for CF baseline
    train_df_id = train_df.copy()
    train_df_id["userId"] = train_df_id["user_key"]
    train_df_id["movieId"] = movie_ids_by_key[train_df_id["movie_key"].to_numpy()]

    # candidates
    K = 200
    global_pop = build_global_popular(train_df, K=K)

    # CF candidates are movieIds; convert to movie_keys via mapping
    cf_cand_movieids = build_cf_candidates(train_df_id, K=K)

    # map movieId -> movie_key
    movieid_to_key = {mid: int(k) for k, mid in zip(mf_sorted["movie_key"].to_numpy(), mf_sorted["movieId"].to_numpy())}

    candidates_by_user = {"global": global_pop}
    for u, mids in cf_cand_movieids.items():
        keys = [movieid_to_key[m] for m in mids if m in movieid_to_key]
        if keys:
            candidates_by_user[int(u)] = keys

    # build user_state and seen_train
    item_matrix, _ = _item_matrix(movie_features)
    movieid_to_features = {}
    for _, row in mf_sorted.iterrows():
        movie_id = int(row["movieId"])
        movie_key = int(row["movie_key"])
        if 0 <= movie_key < item_matrix.shape[0]:
            movieid_to_features[movie_id] = item_matrix[movie_key]
    user_state = build_user_state_vectors(train_df, item_matrix)
    seen_train = defaultdict(set)
    for _, row in train_df.iterrows():
        seen_train[int(row["user_key"])].add(int(row["movie_key"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn = load_dqn(device, n_actions=item_matrix.shape[0], feat_dim=item_matrix.shape[1])

    if dqn is None:
        print("No DQN model found — exiting.")
        return

    recommender = dqn_reranker(dqn, user_state, seen_train, movie_ids_by_key, candidates_by_user, device, N=10)

    # prepare test/train id mapping for evaluation
    test_df_id = test_df.copy()
    test_df_id["userId"] = test_df_id["user_key"]
    test_df_id["movieId"] = movie_ids_by_key[test_df_id["movie_key"].to_numpy()]
    n_items_total = len(mf_sorted["movieId"].unique())

    # evaluate
    dqn_metrics = eval_prcp(
        train_df_id=train_df_id,
        test_df_id=test_df_id,
        n_items_total=n_items_total,
        recommend_func=recommender,
        item_features=movieid_to_features,
        N=10,
    )

    # baselines
    cf_recommender = make_cf_recommender(train_df_id)
    cf_metrics = eval_prcp(
        train_df_id=train_df_id,
        test_df_id=test_df_id,
        n_items_total=n_items_total,
        recommend_func=cf_recommender,
        item_features=movieid_to_features,
        N=10,
    )

    pop_series = train_df_id.groupby("movieId")["userId"].nunique().sort_values(ascending=False)
    top_pop = list(pop_series.index[:10])
    pop_rec = lambda u, n: top_pop[:n]
    pop_metrics = eval_prcp(
        train_df_id=train_df_id,
        test_df_id=test_df_id,
        n_items_total=n_items_total,
        recommend_func=pop_rec,
        item_features=movieid_to_features,
        N=10,
    )

    print("\n=== Candidate Re-ranking Results ===")
    print("DQN re-rank:", dqn_metrics)
    print("CF baseline:", cf_metrics)
    print("Top-popular:", pop_metrics)


if __name__ == "__main__":
    main()
