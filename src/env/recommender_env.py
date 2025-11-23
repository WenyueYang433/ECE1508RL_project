# env.py
from __future__ import annotations
from typing import Dict
import numpy as np
from data_loader import MovieLensLoader
from data_processor import DatasetPrep
from transitions import build_offline_transitions, _item_matrix


class RecoEnv:
    def __init__( self,  data_dir: str = "data/ml-latest-small", val_ratio: float = 0.2,  repeat_penalty: float = 0.0, keep_top_n: int = 1000, popularity_penalty: float = 0.0) -> None:
        loader = MovieLensLoader(data_dir).load_all()
        prep = DatasetPrep(loader)

        # Encoded film features & Scoring table
        movie_features = prep.encode_movies(keep_top_n=keep_top_n)
        rating_table = prep.encode_ratings()

        # Time division train/val
        train_df, val_df = prep.temporal_split(rating_table, val_ratio=val_ratio)
        item_matrix, feature_names = _item_matrix(movie_features)
        self.item_matrix = item_matrix  # [n_movies, feat_dim]
        self.feature_names = feature_names

        # offline transitions
        self.transitions_train = build_offline_transitions(
            train_df, item_matrix, repeat_penalty=repeat_penalty, popularity_penalty=popularity_penalty
        )
        self.transitions_val = build_offline_transitions(
            val_df, item_matrix, repeat_penalty=repeat_penalty, popularity_penalty=popularity_penalty
        )


        self.state_dim: int = self.transitions_train["state"].shape[1]
        self.n_actions: int = item_matrix.shape[0]

        print(
            f"[RecoEnv] state_dim={self.state_dim}, n_actions={self.n_actions}, repeat_penalty={repeat_penalty}, popularity_penalty={popularity_penalty} "
            f"\ntrain_transitions={self.transitions_train['state'].shape[0]}, "
            f"val_transitions={self.transitions_val['state'].shape[0]}"
        )

        r_train = self.transitions_train["reward"]
        print(
            f"[RecoEnv] reward(train): min={r_train.min():.3f}, "
            f"max={r_train.max():.3f}, mean={r_train.mean():.3f}"
        )
        print("Sample train rewards:", r_train[:20])
        
    def _get_split(self, split: str) -> Dict[str, np.ndarray]:
        if split == "train":
            return self.transitions_train
        elif split == "val":
            return self.transitions_val
        else:
            raise ValueError(f"Unknown split: {split}")

    def sample_batch( self, batch_size: int, split: str = "train",) -> Dict[str, np.ndarray]:
        trans = self._get_split(split)
        n = trans["state"].shape[0]
        idx = np.random.randint(0, n, size=batch_size)

        batch = {k: v[idx] for k, v in trans.items()}
        return batch

    def get_all_transitions(
        self, split: str = "train"
    ) -> Dict[str, np.ndarray]:
        trans = self._get_split(split)
        return {k: v.copy() for k, v in trans.items()}

