"""Fine-tune the DQN using candidate-level negatives (pairwise hinge loss).

This script:
- Loads the Base Model (trained via MSE Loss) using settings from Hyperparameters.
- Switches to Hinge Loss to teach the agent Ranking (A > B).
- Uses "Hard Negatives" (Popular items the user skipped) defined in Hyperparameters.
- Saves fine-tuned model to out_model_path and prints evaluation.
"""
from __future__ import annotations
from pathlib import Path
import random
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.optim as optim

from evaluation import prepare_evaluation_data, eval_prcp, make_dqn_recommender
from utils.collaborative import collaborative_filtering_recommend
from agent.dqn_agent import DQN
from agent.ddqn_dueling_model import DuelingDQN

from utils.hyperparameters import Hyperparameters

def build_global_popular(train_df, K=200):
    """Returns top-K most popular movie_keys."""
    pop = train_df.groupby("movie_key")["user_key"].nunique().sort_values(ascending=False)
    return list(pop.index[:K])

def build_cf_candidates(train_df_id, K=200):
    """Returns CF candidates (External MovieIDs) for Hard Negative Mining."""
    candidates = {}
    users = train_df_id["userId"].unique()
    print(f"Building CF candidates for {len(users)} users (this may take a minute)...")
    
    for u in users:    
        try:
            recs = collaborative_filtering_recommend(train_df_id, user_id=int(u), n_recs=K)
        except Exception:
            recs = None
        if recs is None or recs.empty:
            continue
        candidates[int(u)] = list(recs["movieId"].values)
    return candidates

def fine_tune():

    hp = Hyperparameters()
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    
    data_dir = PROJECT_ROOT / hp.data_rel_path
    base_model_path = PROJECT_ROOT / hp.model_base
    out_model_path = PROJECT_ROOT / hp.model_finetuned

    print(f"--- Starting Fine-Tuning Pipeline ---")
    print(f"Config: Steps={hp.ft_n_steps}, LR={hp.ft_lr}, Margin={hp.margin}")

    # LOAD DATA
    data = prepare_evaluation_data(
        data_dir, 
        val_ratio=hp.val_ratio,          
        keep_top_n=hp.keep_top_n, 
        min_ratings=hp.min_ratings,
        history_window=hp.history_window            
    )
    
    # Unpack for convenience
    train_df = data['train_df']
    train_df_id = data['train_df_id']
    user_state = data['user_state']
    seen_train = data['seen_train']
    movie_ids_by_key = data['movie_ids_by_key']
    n_actions = data['n_actions']
    feat_dim = data['feat_dim']
    
    state_dim = feat_dim * hp.history_window
    
    # Inverse map for CF candidate conversion (ID -> Key)
    movieid_to_key = {mid: k for k, mid in enumerate(movie_ids_by_key)}

    # BUILD CANDIDATES (Hard Negatives)
    candidates_by_user = {}
    if hp.use_candidates:
        print("Generating Hard Negatives...")
        global_pop_keys = build_global_popular(train_df, K=hp.candidate_k)
        cf_cand_movieids = build_cf_candidates(train_df_id, K=hp.candidate_k)

        candidates_by_user = {"global": global_pop_keys}
        for u, mids in cf_cand_movieids.items():
            keys = [movieid_to_key[m] for m in mids if m in movieid_to_key]
            if keys:
                candidates_by_user[int(u)] = keys
    else:
        print("Skipping Hard Negative generation (using Random Negatives only).")

    # Build User -> Positive List
    user_pos = defaultdict(list)
    for _, row in train_df.iterrows():
        user_pos[int(row["user_key"])].append(int(row["movie_key"]))

    eligible_users = [u for u, poses in user_pos.items() if len(poses) > 0]
    
    # SETUP MODEL
    device = torch.device(hp.device if torch.cuda.is_available() else "cpu")
    if getattr(hp, "use_dueling", False): Net = DuelingDQN
    else: Net = DQN
    dqn = Net(num_actions=n_actions, feature_size=state_dim).to(device)
    
    if base_model_path.exists():
        state = torch.load(base_model_path, map_location=device)
        dqn.load_state_dict(state)
        print(f"Loaded Base Model from {base_model_path}")
    else:
        print("WARNING: No base model found; training from scratch (Not Recommended).")

    # Use Fine-Tuning specific LR and Decay
    optimizer = optim.Adam(dqn.parameters(), lr=hp.ft_lr, weight_decay=hp.ft_weight_decay)
    
    # BASELINE EVALUATION
    print("Evaluating Base Model Performance...")
    dqn.eval()
    dqn_recommender = make_dqn_recommender(
        dqn_model=dqn, user_state=user_state, seen_train=seen_train,
        movie_ids_by_key=movie_ids_by_key, n_actions=n_actions, device=device
    )
    
    before_metrics = eval_prcp(
        data['train_df_id'],
        data['test_df_id'],
        n_items_total=len(movie_ids_by_key),
        recommend_func=dqn_recommender,
        item_features=data['movieid_to_features'],
        N=10,
    )
    print(f"Base Precision@10: {before_metrics['Precision@10']:.4f} | Base NDCG@10: {before_metrics['NDCG@10']:.4f}")

    # TRAINING LOOP
    dqn.train()
    best_prec = 0.0
    
    print(f"Starting Fine-Tuning ({hp.ft_n_steps} steps)...")
    
    for step in range(1, hp.ft_n_steps + 1):
        batch_users = random.choices(eligible_users, k=hp.ft_batch_size)
        states = []
        pos_idx = []
        neg_idx = []
        
        for u in batch_users:
            # Positive Sample
            pos = random.choice(user_pos[u])
            
            # Negative Sample Strategy
            pool = candidates_by_user.get(u, candidates_by_user.get("global", [])) if hp.use_candidates else []
            pool = [k for k in pool if k != pos and k not in seen_train.get(u, set())]
            negs = []
            
            n_cand = int(round(hp.neg_per_pos * hp.frac_candidate)) if hp.use_candidates else 0
            n_rand = hp.neg_per_pos - n_cand
            
            # Hard Negatives
            if n_cand > 0 and pool:
                if len(pool) <= n_cand:
                    negs.extend(random.choices(pool, k=n_cand))
                else:
                    negs.extend(random.sample(pool, n_cand))
            else:
                n_rand += n_cand 

            # Random Negatives
            while len(negs) < hp.neg_per_pos:
                n = random.randint(0, n_actions - 1)
                if n != pos and n not in seen_train.get(u, set()) and n not in negs:
                    negs.append(n)

            states.append(user_state.get(u, np.zeros(state_dim, dtype=np.float32)))
            pos_idx.append(pos)
            neg_idx.append(negs)

        # Tensorize
        states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
        pos_t = torch.tensor(pos_idx, dtype=torch.long, device=device).unsqueeze(1) 

        # Forward
        q_all = dqn(states_t)
        q_pos = q_all.gather(1, pos_t)

        # Hinge Loss Calculation
        loss_val = 0
        for i in range(hp.ft_batch_size):
            negs_t = torch.tensor(neg_idx[i], dtype=torch.long, device=device) 
            q_neg_samples = q_all[i][negs_t] 
            
            # Q_pos > Q_neg + margin
            gap = q_pos[i] - q_neg_samples
            current_loss = torch.clamp(hp.margin - gap, min=0.0).mean()
            loss_val += current_loss
        
        loss_val /= hp.ft_batch_size

        optimizer.zero_grad()
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step}/{hp.ft_n_steps} | Hinge Loss: {loss_val.item():.4f}")

        # Periodic Eval (every 200 steps)
        if step % 200 == 0:
            dqn.eval()
            rec_func = make_dqn_recommender(dqn, user_state, seen_train, movie_ids_by_key, n_actions, device)
            
            # Fast Eval (Precision@10 only)
            m = eval_prcp(data['train_df_id'], data['test_df_id'], len(movie_ids_by_key), rec_func, data['movieid_to_features'], N=10)
            prec = m.get("Precision@10", 0.0)
            ndcg = m.get("NDCG@10", 0.0)
            
            print(f"  > Eval: Precision@10={prec:.4f} | NDCG@10={ndcg:.4f}")
            
            if prec > best_prec:
                best_prec = prec
                torch.save(dqn.state_dict(), out_model_path)
                print(f"  > New Best Model Saved!")
            
            dqn.train()

    print(f"Fine-tuning complete. Best model saved to {out_model_path}")

    # FINAL FULL REPORT
    print("\nLoading best fine-tuned model for final report...")
    dqn.load_state_dict(torch.load(out_model_path, map_location=device))
    dqn.eval()
    
    final_rec = make_dqn_recommender(dqn, user_state, seen_train, movie_ids_by_key, n_actions, device)
    
    print("Running full validation on Best Model...")
    after_metrics = eval_prcp(
        data['train_df_id'], 
        data['test_df_id'], 
        len(movie_ids_by_key), 
        final_rec, 
        data['movieid_to_features'], 
        N=10
    )
    
    print("\n=== Final Fine-Tuned Results ===")
    for k, v in after_metrics.items():
        print(f"{k}: {v:.4f}")

    # Debug: top recommendations to see diversity
    all_recs = []
    for u in list(user_state.keys())[:500]:
        all_recs.extend(final_rec(u, 10))
    top20 = Counter(all_recs).most_common(20)
    print("\nTop-20 Recommended MovieIDs (After):", top20)


if __name__ == "__main__":
    fine_tune()