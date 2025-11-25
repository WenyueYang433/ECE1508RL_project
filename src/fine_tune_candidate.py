"""Fine-tune the DQN using candidate-level negatives (pairwise hinge loss).

Usage:
  python -m src.fine_tune_candidate --n_steps 2000 --neg_per_pos 5

This script:
- Loads preprocessing and the saved DQN (if present) or a fresh DQN.
- Builds candidate pools (global-popular + CF per-user).
- Samples (user, positive, negatives) and applies a hinge loss
  loss = max(0, margin - (Q(u,pos) - Q(u,neg))).
- Saves fine-tuned model to `models/dqn_movielens_finetuned.pt` and prints evaluation.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import random
from collections import defaultdict
import time
from collections import Counter
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import MovieLensLoader
from data_processor import DatasetPrep
from transitions import _item_matrix
from agent.dqn_agent import DQN
from evaluation import build_user_state_vectors, make_dqn_recommender, make_cf_recommender, eval_prcp
from utils.collaborative import collaborative_filtering_recommend


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "dqn_movielens.pt"
OUT_MODEL = PROJECT_ROOT / "models" / "dqn_movielens_finetuned.pt"


def build_global_popular(train_df, K=200):
    pop = train_df.groupby("movie_key")["user_key"].nunique().sort_values(ascending=False)
    return list(pop.index[:K])


def build_cf_candidates(train_df_id, K=200):
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


def map_movieid_to_key(mf_sorted):
    # mf_sorted: dataframe sorted by movie_key
    movieid_to_key = {int(mid): int(k) for k, mid in zip(mf_sorted["movie_key"].to_numpy(), mf_sorted["movieId"].to_numpy())}
    return movieid_to_key


def fine_tune(args):
    data_dir = PROJECT_ROOT / "data" / "ml-latest-small"
    loader = MovieLensLoader(str(data_dir)).load_all()
    prep = DatasetPrep(loader)

    movie_features = prep.encode_movies(keep_top_n=args.keep_top_n)
    ratings = prep.encode_ratings()
    train_df, test_df = prep.temporal_split(ratings, val_ratio=0.1)

    mf_sorted = movie_features.sort_values("movie_key")
    movie_ids_by_key = mf_sorted["movieId"].to_numpy()
    movieid_to_key = map_movieid_to_key(mf_sorted)
    # safe mapping: movie_key -> movieId
    key_to_movieid = dict(zip(mf_sorted["movie_key"].to_numpy(), mf_sorted["movieId"].to_numpy()))

    # train_df_id for CF (map movie_key -> movieId safely)
    train_df_id = train_df.copy()
    train_df_id["userId"] = train_df_id["user_key"]
    train_df_id["movieId"] = train_df_id["movie_key"].map(key_to_movieid)
    # drop rows where we couldn't map to a movieId
    train_df_id = train_df_id.dropna(subset=["movieId"]).copy()
    train_df_id["movieId"] = train_df_id["movieId"].astype(int)

    # build candidates (keys)
    K = args.candidate_k
    global_pop_keys = build_global_popular(train_df, K=K)
    cf_cand_movieids = build_cf_candidates(train_df_id, K=K)

    candidates_by_user = {"global": global_pop_keys}
    for u, mids in cf_cand_movieids.items():
        keys = [movieid_to_key[m] for m in mids if m in movieid_to_key]
        if keys:
            candidates_by_user[int(u)] = keys

    # build user -> list of positive movie_keys (from train)
    user_pos = defaultdict(list)
    for _, row in train_df.iterrows():
        user_pos[int(row["user_key"])].append(int(row["movie_key"]))

    # user_state and seen
    item_matrix, _ = _item_matrix(movie_features)
    movieid_to_features: Dict[int, np.ndarray] = {}
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
    n_actions = item_matrix.shape[0]
    feat_dim = item_matrix.shape[1]

    # load or init dqn
    dqn = DQN(num_actions=n_actions, feature_size=feat_dim).to(device)
    if MODEL_PATH.exists():
        state = torch.load(MODEL_PATH, map_location=device)
        dqn.load_state_dict(state)
        print(f"Loaded base model from {MODEL_PATH}")
    else:
        print("No base model found; training from scratch")

    optimizer = optim.Adam(dqn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    margin = args.margin

    # prepare list of eligible users
    eligible_users = [u for u, poses in user_pos.items() if len(poses) > 0]
    if not eligible_users:
        print("No eligible users with train positives — aborting.")
        return

    # evaluation before fine-tuning
    dqn.eval()
    dqn_recommender = make_dqn_recommender(dqn_model=dqn, user_state=user_state, seen_train=seen_train,
                                           movie_ids_by_key=movie_ids_by_key, n_actions=n_actions, device=device)
    # use the already-prepared `train_df_id` (has movieId/userId); prepare `test_df_id` similarly
    train_df_id_for_eval = train_df_id.copy()
    test_df_id = test_df.copy()
    test_df_id["userId"] = test_df_id["user_key"]
    test_df_id["movieId"] = test_df_id["movie_key"].map(key_to_movieid)
    test_df_id = test_df_id.dropna(subset=["movieId"]).copy()
    test_df_id["movieId"] = test_df_id["movieId"].astype(int)
    n_items_total = len(mf_sorted["movieId"].unique())
    before_metrics = eval_prcp(
        train_df_id=train_df_id_for_eval,
        test_df_id=test_df_id,
        n_items_total=n_items_total,
        recommend_func=dqn_recommender,
        item_features=movieid_to_features,
        N=10,
    )
    print("Before fine-tune:", before_metrics)

    if args.debug:
        # top-20 recommended items across users
        all_recs = []
        for u in list(user_state.keys())[:500]:
            recs = dqn_recommender(u, 20)
            all_recs.extend(recs)
        top20 = Counter(all_recs).most_common(20)
        print("Top-20 recommended movieIds (before):", top20)

        # quick Q-stat diagnostics on a small sample of user positives vs negatives
        sample_users = random.sample(list(user_pos.keys()), min(200, len(user_pos)))
        q_pos_vals = []
        q_neg_vals = []
        for u in sample_users:
            poses = user_pos[u]
            pos = random.choice(poses)
            state_vec = user_state.get(u, np.zeros(feat_dim, dtype=np.float32))
            st = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = dqn(st).cpu().numpy().reshape(-1)
            q_pos_vals.append(float(q[pos]))
            # sample one random neg
            neg = random.choice([k for k in range(n_actions) if k != pos])
            q_neg_vals.append(float(q[neg]))
        print(f"Q pos mean:{np.mean(q_pos_vals):.4f} | Q neg mean:{np.mean(q_neg_vals):.4f}")

    # training loop
    dqn.train()
    total_steps = args.n_steps
    batch_size = args.batch_size
    neg_per_pos = args.neg_per_pos

    users = eligible_users
    best_prec = -1.0
    best_state = None
    eval_interval = args.eval_interval

    for step in range(1, total_steps + 1):
        batch_users = random.choices(users, k=batch_size)
        states = []
        pos_idx = []
        neg_idx = []
        for u in batch_users:
            poses = user_pos[u]
            pos = random.choice(poses)
            # candidate pool for negatives (mix candidates + random to avoid narrow bias)
            pool = candidates_by_user.get(u, candidates_by_user["global"]) if args.use_candidates else list(range(n_actions))
            # remove positives and seen
            pool = [k for k in pool if k != pos and k not in seen_train.get(u, set())]
            negs = []
            # sample some negatives from pool (if requested) and some randoms
            n_cand = int(round(neg_per_pos * args.frac_candidate)) if args.use_candidates else 0
            n_rand = neg_per_pos - n_cand
            if n_cand > 0 and pool:
                if len(pool) <= n_cand:
                    negs.extend(random.choices(pool, k=n_cand))
                else:
                    negs.extend(random.sample(pool, n_cand))
            # sample remaining random negatives from all actions excluding pos
            rand_pool = [k for k in range(n_actions) if k != pos and k not in seen_train.get(u, set())]
            if len(rand_pool) == 0:
                rand_pool = [k for k in range(n_actions) if k != pos]
            if n_rand > 0:
                if len(rand_pool) <= n_rand:
                    negs.extend(random.choices(rand_pool, k=n_rand))
                else:
                    negs.extend(random.sample(rand_pool, n_rand))

            states.append(user_state.get(u, np.zeros(feat_dim, dtype=np.float32)))
            pos_idx.append(pos)
            neg_idx.append(negs)

        states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
        q_all = dqn(states_t)  # [B, n_actions]

        pos_tensor = torch.tensor(pos_idx, dtype=torch.long, device=device).unsqueeze(1)  # [B,1]
        q_pos = q_all.gather(1, pos_tensor)  # [B,1]

        # expand negatives
        q_negs = []
        for i in range(batch_size):
            negs = neg_idx[i]
            neg_tensor = torch.tensor(negs, dtype=torch.long, device=device).unsqueeze(0)  # [1, neg_per]
            qn = q_all[i].gather(0, neg_tensor.view(-1)).view(1, -1)  # [1, neg_per]
            q_negs.append(qn)
        q_negs_t = torch.cat(q_negs, dim=0)  # [B, neg_per]

        # hinge loss per negative: max(0, margin - (q_pos - q_neg))
        q_pos_exp = q_pos.expand_as(q_negs_t)  # [B, neg_per]
        loss_mat = torch.clamp(margin - (q_pos_exp - q_negs_t), min=0.0)
        loss = loss_mat.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=1.0)
        optimizer.step()

        if step % args.log_interval == 0 or step == 1:
            print(f"Step {step}/{total_steps} | loss={loss.item():.4f}")

        # periodic evaluation & checkpointing
        if eval_interval > 0 and (step % eval_interval == 0 or step == total_steps):
            dqn.eval()
            dqn_recommender = make_dqn_recommender(dqn_model=dqn, user_state=user_state, seen_train=seen_train,
                                                   movie_ids_by_key=movie_ids_by_key, n_actions=n_actions, device=device)
            metrics = eval_prcp(
                train_df_id=train_df_id_for_eval,
                test_df_id=test_df_id,
                n_items_total=n_items_total,
                recommend_func=dqn_recommender,
                item_features=movieid_to_features,
                N=10,
            )
            prec = metrics.get("Precision@10", 0.0)
            coverage = metrics.get("Coverage@Catalog", metrics.get("Coverage", 0.0))
            popularity = metrics.get("Popularity", 0.0)
            print(
                f"  Eval @ step {step}: Precision@10={prec:.4f}, "
                f"Coverage@Catalog={coverage:.4f}, Popularity={popularity:.4f}"
            )
            if prec > best_prec:
                best_prec = prec
                best_state = {k: v.cpu().clone() for k, v in dqn.state_dict().items()}
                torch.save(best_state, OUT_MODEL)
                print(f"  New best Precision@10={prec:.4f} — checkpoint saved to {OUT_MODEL}")
            dqn.train()

    # save finetuned model
    torch.save(dqn.state_dict(), OUT_MODEL)
    print(f"Saved fine-tuned model to {OUT_MODEL}")

    # evaluation after
    dqn.eval()
    dqn_recommender = make_dqn_recommender(dqn_model=dqn, user_state=user_state, seen_train=seen_train,
                                           movie_ids_by_key=movie_ids_by_key, n_actions=n_actions, device=device)
    after_metrics = eval_prcp(
        train_df_id=train_df_id_for_eval,
        test_df_id=test_df_id,
        n_items_total=n_items_total,
        recommend_func=dqn_recommender,
        item_features=movieid_to_features,
        N=10,
    )
    print("After fine-tune:", after_metrics)

    if args.debug:
        # top-20 recommended items across users
        all_recs = []
        for u in list(user_state.keys())[:500]:
            recs = dqn_recommender(u, 20)
            all_recs.extend(recs)
        top20 = Counter(all_recs).most_common(20)
        print("Top-20 recommended movieIds (after):", top20)

        # Q-stat diagnostics after
        sample_users = random.sample(list(user_pos.keys()), min(200, len(user_pos)))
        q_pos_vals = []
        q_neg_vals = []
        for u in sample_users:
            poses = user_pos[u]
            pos = random.choice(poses)
            state_vec = user_state.get(u, np.zeros(feat_dim, dtype=np.float32))
            st = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = dqn(st).cpu().numpy().reshape(-1)
            q_pos_vals.append(float(q[pos]))
            neg = random.choice([k for k in range(n_actions) if k != pos])
            q_neg_vals.append(float(q[neg]))
        print(f"Q pos mean:{np.mean(q_pos_vals):.4f} | Q neg mean:{np.mean(q_neg_vals):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--neg-per-pos", type=int, default=3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--keep-top-n", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--frac-candidate", type=float, default=0.67, help="Fraction of negatives sampled from candidate pool (rest random)")
    parser.add_argument("--use-candidates", action="store_true", help="Sample negatives from candidate pool (recommended)")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    args = parser.parse_args()
    fine_tune(args)
