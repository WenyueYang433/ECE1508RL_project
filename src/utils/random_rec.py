# baselines/random_rec.py
import pandas as pd
import numpy as np

def random_recommend(df: pd.DataFrame, user_id: int, n_recs: int = 10, seed: int = 42) -> pd.DataFrame:
    if seed is not None: np.random.seed(seed)


    all_items = df["movieId"].unique()
    user_interacted = df.loc[df["userId"] == user_id, "movieId"].unique()
    #Exclude watched items (Candidates = All - Watched)
    candidates = np.setdiff1d(all_items, user_interacted)

    num_to_sample = min(len(candidates), n_recs)
    
    if num_to_sample == 0: return pd.DataFrame(columns=["movieId", "score"])

    # Randomly select candidates
    recommendations = np.random.choice(candidates, size=num_to_sample, replace=False)

    random_scores = np.random.random(size=num_to_sample)

    rec_df = pd.DataFrame({
        "movieId": recommendations,
        "score": random_scores
    })
    rec_df = rec_df.sort_values("score", ascending=False).reset_index(drop=True)

    return rec_df