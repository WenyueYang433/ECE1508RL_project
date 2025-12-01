"""Build offline DQN transitions from MovieLens data."""

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_loader import MovieLensLoader
from data_processor import DatasetPrep
from collections import deque

'''
state: flattened vector of the last N movies (use Sliding Window method)
action: ID of the recommended movie 
reward: the user's actual rating(normalized), with penalty for recommending seen movie 
'''


def _item_matrix(movie_features: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    keep_cols = [
        col
        for col in movie_features.columns
        if col not in {"movie_key", "movieId", "title"}
    ]
    sorted_df = movie_features.sort_values("movie_key")
    matrix = sorted_df[keep_cols].to_numpy(dtype=np.float32)
    return matrix, keep_cols


def build_offline_transitions(
    ratings_df: pd.DataFrame,
    item_matrix: np.ndarray,
    repeat_penalty: float = 0.0,
    popularity_penalty: float = 0.0,
    history_window: int = 10,
    is_gru: bool = False  
) -> Dict[str, np.ndarray]:
    df = ratings_df.sort_values(["user_key", "timestamp"]).reset_index(drop=True)

    # compute normalized popularity per movie_key (0..1)
    if popularity_penalty:
        counts = ratings_df["movie_key"].value_counts().to_dict()
        pop = np.zeros(item_matrix.shape[0], dtype=np.float32)
        for k, v in counts.items():
            if 0 <= int(k) < item_matrix.shape[0]:
                pop[int(k)] = v
        maxp = float(pop.max()) if pop.max() > 0 else 1.0
        pop_norm = pop / maxp
    else:
        pop_norm = np.zeros(item_matrix.shape[0], dtype=np.float32)

    dim = item_matrix.shape[1]
    empty_feat = np.zeros(dim, dtype=np.float32)
    
    transitions: Dict[str, List] = {
        "state": [],
        "action": [],
        "reward": [],
        "next_state": [],
        "done": [],
    }

    for _, user_hist in df.groupby("user_key"):
        movie_keys = user_hist["movie_key"].to_numpy()
        ratings = user_hist["rating"].to_numpy(dtype=np.float32)

        if len(movie_keys) < 2:
            continue
        
 
        # deque with a fixed maxlen, the oldest item pushed out when append new item
        history = deque([empty_feat] * history_window, maxlen=history_window)
        watched = set()
        
        for idx in range(len(movie_keys) - 1):

            # preserves sequence
            if is_gru:
                # (window, features)
                state = np.array(history, dtype=np.float32)
            else:
                # Flattened (Window * Features) (t-windowSize,....,t-1 )
                state = np.concatenate(history).astype(np.float32)

            action_key = movie_keys[idx]
            rating_val = ratings[idx]

            already_seen = action_key in watched   
            #use negative rewards for 1&2 rating, force the agent to learn from negative feedback(i.e. recommend bad movie)
            #1 star = -2.0 strong Punishment, 3 stars = 0.0 neutral, 5 Stars = +2 (Strong Reward)
            base_reward = (rating_val - 3.0) / 2.0
            # penalty for recommend seen movies
            r_penalty = repeat_penalty if already_seen else 0.0
            reward = base_reward - r_penalty

            # popularity_penalty
            if popularity_penalty and 0 <= action_key < len(pop_norm):
                reward = reward - float(popularity_penalty * pop_norm[int(action_key)])
            
            # Get feature vector for the movie just watched
            if 0 <= action_key < len(item_matrix):
                movie_vec = item_matrix[action_key]
            else:
                movie_vec = empty_feat
                
            # append movie just watched to the history, remove olest one
            history.append(movie_vec)
            
            if is_gru:
                next_state = np.array(history, dtype=np.float32)
            else:
                next_state = np.concatenate(history).astype(np.float32)
            
            watched.add(action_key)
            done = idx == len(movie_keys) - 2

            transitions["state"].append(state)
            transitions["action"].append(int(action_key))
            transitions["reward"].append(float(reward))
            transitions["next_state"].append(next_state)
            transitions["done"].append(done)
            
            # Negative Sampling
            # Use a fake "negative" experience to teach the agent that random movies are bad       
            neg_action = np.random.randint(0, item_matrix.shape[0])  #  random movie ID that is NOT the current action
            # give it a negative reward (-0.5 = a 2-star rating)
            neg_reward = -0.5 
            # state/next_state is the same (we assume user ignored this recommendation)
            # window does not slide for rejected items
            transitions["state"].append(state)
            transitions["action"].append(int(neg_action))
            transitions["reward"].append(float(neg_reward))
            transitions["next_state"].append(state) # Next state doesn't change because they didn't watch it
            transitions["done"].append(False)       # Negatives don't end the episode

    if not transitions["state"]:
        raise ValueError("No transitions built; check input data.")

    out = {
        "state": np.array(transitions["state"], dtype=np.float32),
        "action": np.array(transitions["action"], dtype=np.int32),
        "reward": np.array(transitions["reward"], dtype=np.float32),
        "next_state": np.array(transitions["next_state"], dtype=np.float32),
        "done": np.array(transitions["done"], dtype=np.bool_),
    }
    return out


if __name__ == "__main__":
    loader = MovieLensLoader("data/ml-latest-small").load_all()
    prep = DatasetPrep(loader)
    movie_features = prep.encode_movies()
    ratings = prep.encode_ratings()

    item_matrix, feature_names = _item_matrix(movie_features)
    transitions = build_offline_transitions(ratings, item_matrix, repeat_penalty=0.1)

    print("Item feature dim:", item_matrix.shape)
    print("Transitions:")
    for key, value in transitions.items():
        print(f"  {key}: shape {value.shape}")

