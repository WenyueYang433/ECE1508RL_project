class Hyperparameters:
    def __init__(
        self,
        buffer_size=100_000,
        batch_size=512,
        learning_rate=1e-3,
        gamma=0.99,
        target_update=2000,
        log_interval=1000,
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.target_update = target_update
        self.log_interval = log_interval
        self.epsilon_decay = 0.999
