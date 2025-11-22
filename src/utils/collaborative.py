# baselines/collaborative.py
import pandas as pd
import numpy as np


def collaborative_filtering_recommend( df: pd.DataFrame, user_id: int,n_recs: int = 10, k_neighbors: int = 40, min_overlap: int = 5,) -> pd.DataFrame:
    df = df[["userId", "movieId", "rating"]].copy()

    rating_mat = df.pivot_table( index="userId", columns="movieId", values="rating", aggfunc="mean")

    if user_id not in rating_mat.index:
        raise ValueError(f"user_id {user_id} not in rating matrix")

    target_ratings = rating_mat.loc[user_id]
    user_means = rating_mat.mean(axis=1)
    rating_centered = rating_mat.sub(user_means, axis=0)
    target_centered = rating_centered.loc[user_id]

    sims = []
    for other_user, row in rating_centered.iterrows():
        if other_user == user_id:
            continue
        mask = target_centered.notna() & row.notna()
        if mask.sum() < min_overlap:
            sims.append((other_user, 0.0))
            continue

        v1 = target_centered[mask].values
        v2 = row[mask].values

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            sims.append((other_user, 0.0))
            continue

        sim = float(np.dot(v1, v2) / (norm1 * norm2))
        sims.append((other_user, sim))
    sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
    neighbors = [(u, s) for (u, s) in sims_sorted if s > 0][:k_neighbors]

    if len(neighbors) == 0:  return pd.DataFrame(columns=["movieId", "score"])

    neighbor_ids = [u for (u, s) in neighbors]
    neighbor_sims = np.array([s for (u, s) in neighbors], dtype=np.float32)

    neighbor_centered = rating_centered.loc[neighbor_ids]  # [K, n_items]


    watched_mask = target_ratings.notna()
    candidate_mask = ~watched_mask & (neighbor_centered.notna().sum(axis=0) > 0)
    candidate_items = rating_mat.columns[candidate_mask]

    if len(candidate_items) == 0:
        return pd.DataFrame(columns=["movieId", "score"])

    scores = []

    for m in candidate_items:
        col = neighbor_centered[m]  # Series, index = neighbor_ids
        mask = col.notna()
        if mask.sum() == 0:
            continue

        r_center = col[mask].values          
        s_valid = neighbor_sims[mask.values] 
        denom = np.sum(np.abs(s_valid))
        if denom == 0:
            continue

        pred_centered = np.sum(s_valid * r_center) / denom
        target_mean = float(user_means.loc[user_id])
        pred_score = target_mean + pred_centered

        scores.append((m, pred_score))

    if len(scores) == 0:
        return pd.DataFrame(columns=["movieId", "score"])

    #top-n
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)[:n_recs]
    rec_df = pd.DataFrame(scores_sorted, columns=["movieId", "score"])

    return rec_df