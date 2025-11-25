class Hyperparameters:
    def __init__(self):
        # --- GLOBAL SETTINGS ---
        self.device = "cuda" # or "cpu"
        self.seed = 42
        self.keep_top_n = 1000  # Number of movies to consider for action space 
        self.min_ratings = 5    # User filter (Cold Start Fix)
        self.val_ratio = 0.2    # Train/Test split
        
        # --- PATHS ---
        self.data_rel_path = "data/ml-latest-small" 
        self.model_base = "models/dqn_movielens.pt"
        self.model_finetuned = "models/dqn_movielens_finetuned.pt"
        self.plot_summary = "reports/figures/training_summary.png"
        
        # --- ALGORITHM FLAGS ---
        self.use_ddqn = True  # Set False: standard DQN, True: Double DQN
        
        # --- NN ARCHITECTURE ---
        self.hidden_dim = 128     # width first layer 
        self.dropout_rate = 0.2 
        
        # --- PHASE 1: BASE TRAINING ---
        self.buffer_size = 100_000
        self.batch_size = 512
        self.learning_rate = 1e-4      
        self.gamma = 0.5 
        self.target_update = 2000
        self.base_n_updates = 5000
        self.log_interval = 500
        self.weight_decay = 1e-5
        
        # --- PHASE 2: FINE-TUNING (Hinge Loss) ---
        self.ft_n_steps = 2000
        self.ft_batch_size = 32     # Smaller batch for ranking
        self.ft_lr = 1e-5           # Very small to preserve knowledge
        self.ft_weight_decay = 1e-5
        
        # Ranking Logic
        self.margin = 0.5           # Hinge margin
        self.neg_per_pos = 5        # Negatives per positive
        self.use_candidates = True  # Hard Negative Mining
        self.candidate_k = 200      # Size of candidate pool
        self.frac_candidate = 0.5   # 50% Hard Negatives, 50% Random