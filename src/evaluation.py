from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from data_loader import MovieLensLoader
from data_processor import DatasetPrep
from transitions import _item_matrix
from agent.dqn_agent import DQN
from utils.collaborative import collaborative_filtering_recommend

#Calculates the performance metrics by comparing the agent's recommendations against the movies the user actually watched in the test set
def eval_prcp(  train_df_id: pd.DataFrame, test_df_id: pd.DataFrame, n_items_total: int, recommend_func, N: int = 10) -> dict[str, float]:
    item_pop = train_df_id.groupby("movieId")["userId"].nunique()
    log_pop = np.log1p(item_pop)  # log(1+pop)

    hit = 0
    rec_cnt = 0
    test_cnt = 0

    all_rec_items = []
    pop_sum = 0.0

    users = test_df_id["userId"].unique()

    for u in users:
        test_items = set(test_df_id.loc[test_df_id["userId"] == u, "movieId"])
        if not test_items:
            continue

        rec_items = recommend_func(u, N)
        if not rec_items:
            continue

        rec_items = list(dict.fromkeys(rec_items))  

        inter = test_items.intersection(rec_items)
        hit += len(inter)
        rec_cnt += len(rec_items)
        test_cnt += len(test_items)

        all_rec_items.extend(rec_items)
        for i in rec_items:
            if i in log_pop:
                pop_sum += float(log_pop[i])

    if rec_cnt == 0 or test_cnt == 0:
        return {  f"Precision@{N}": 0.0, f"Recall@{N}": 0.0,   "Coverage": 0.0,  "Popularity": 0.0, }

    precision = hit / rec_cnt
    recall = hit / test_cnt

    unique_recs = len(set(all_rec_items))
    coverage = unique_recs / float(n_items_total)

    popularity = pop_sum / rec_cnt

    return { f"Precision@{N}": precision,  f"Recall@{N}": recall,  "Coverage": coverage,  "Popularity": popularity,}

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


##########
def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    data_dir = PROJECT_ROOT / "data" / "ml-latest-small"
    model_path = PROJECT_ROOT / "models" / "dqn_movielens.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = MovieLensLoader(str(data_dir)).load_all()
    prep = DatasetPrep(loader)
    movie_features = prep.encode_movies()
    ratings = prep.encode_ratings()  # [user_key, movie_key, rating, timestamp]
    train_df, test_df = prep.temporal_split(ratings, val_ratio=0.1)

    # movie_key -> movieId 
    mf_sorted = movie_features.sort_values("movie_key")
    movie_ids_by_key = mf_sorted["movieId"].to_numpy()
    n_items_total = len(mf_sorted["movieId"].unique())

    # userId/movieId TABLE
    train_df_id = train_df.copy()
    test_df_id = test_df.copy()
    train_df_id["userId"] = train_df_id["user_key"]
    test_df_id["userId"] = test_df_id["user_key"]
    train_df_id["movieId"] = movie_ids_by_key[train_df_id["movie_key"].to_numpy()]
    test_df_id["movieId"] = movie_ids_by_key[test_df_id["movie_key"].to_numpy()]

    # DQN 
    item_matrix, _ = _item_matrix(movie_features)
    user_state = build_user_state_vectors(train_df, item_matrix)
    seen_train = defaultdict(set)
    for _, row in train_df.iterrows():
        seen_train[int(row["user_key"])].add(int(row["movie_key"]))

    n_actions = item_matrix.shape[0]
    feat_dim = item_matrix.shape[1]

    dqn = DQN(num_actions=n_actions, feature_size=feat_dim).to(device)
    state = torch.load(model_path, map_location=device)
    dqn.load_state_dict(state)

    dqn_recommender = make_dqn_recommender( dqn_model=dqn, user_state=user_state, seen_train=seen_train, 
                                           movie_ids_by_key=movie_ids_by_key, n_actions=n_actions,device=device)
    cf_recommender = make_cf_recommender(train_df_id)
    N = 10

    dqn_metrics = eval_prcp(train_df_id=train_df_id,test_df_id=test_df_id,n_items_total=n_items_total, 
                            recommend_func=dqn_recommender,  N=N)

    cf_metrics = eval_prcp( train_df_id=train_df_id,test_df_id=test_df_id,n_items_total=n_items_total,
        recommend_func=cf_recommender, N=N)

    print(f"\n=== Top-{N} Evaluation (Precision / Recall / Coverage / Popularity) ===")
    print("DQN:")
    for k, v in dqn_metrics.items():
        print(f"  {k}: {v*100:.2f}%" if "Precision" in k or "Recall" in k else f"  {k}: {v:.4f}")

    print("\nCF (User-based collaborative):")
    for k, v in cf_metrics.items():
        print(f"  {k}: {v*100:.2f}%" if "Precision" in k or "Recall" in k else f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()