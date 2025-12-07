class Hyperparameters:
    def __init__(self):
        self.device = "cuda"
        self.seed = 42
        self.keep_top_n = 1000
        self.min_ratings = 5
        self.val_ratio = 0.2

        self.data_rel_path = "data/ml-latest-small"
        self.model_base = "models/dqn_movielens.pt"
        self.model_finetuned = "models/dqn_movielens_finetuned.pt"
        self.plot_summary = "reports/figures/training_summary.png"

        self.model_arch = "MLP"  # MLP, Dueling, GRU

        self.use_double_q = True

        self.history_window = 10

        self.hidden_dim = 256
        self.dropout_rate = 0.2

        self.buffer_size = 100_000
        self.batch_size = 256
        self.learning_rate = 3e-4
        self.gamma = 0.9
        self.target_update = 1000
        self.base_n_updates = 10000
        self.log_interval = 500
        self.weight_decay = 1e-5
        self.repeat_penalty = 1
        self.popularity_penalty = 0.2

        self.ft_n_steps = 2000
        self.ft_batch_size = 32
        self.ft_lr = 1e-5
        self.ft_weight_decay = 1e-5

        self.margin = 0.5
        self.neg_per_pos = 5
        self.use_candidates = True
        self.candidate_k = 200
        self.frac_candidate = 0.5
