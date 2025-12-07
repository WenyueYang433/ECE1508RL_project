import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from env.recommender_env import RecoEnv
from agent.dqn_agent import Agent
from utils.hyperparameters import Hyperparameters
from evaluation import prepare_evaluation_data
from utils.logger import Logger


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    hp = Hyperparameters()
    is_gru = hp.model_arch == "GRU"
    algo_type = "DDQN" if hp.use_double_q else "DQN"

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # keep artifacts grouped by run

    data_dir = PROJECT_ROOT / hp.data_rel_path
    model_name_str = f"{hp.model_arch}_{algo_type}"
    model_filename = f"{model_name_str}_{run_timestamp}.pt"
    model_save_path = PROJECT_ROOT / "models" / model_filename
    plot_filename = f"{model_name_str}_{run_timestamp}.png"
    plot_save_path = PROJECT_ROOT / "reports" / "figures" / plot_filename
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_save_path.parent.mkdir(parents=True, exist_ok=True)

    log_dir = PROJECT_ROOT / "reports"
    log_dir.mkdir(exist_ok=True)
    log_filename = f"training_log_{datetime.now():%Y%m%d_%H%M%S}.log"
    sys.stdout = Logger(log_dir / log_filename)
    sys.stderr = sys.stdout

    print(f"--- Starting Training: {model_name_str} {run_timestamp} ---")
    print("--- Hyperparameters ---")
    for key, value in vars(hp).items():
        print(f"{key}: {value}")
    print(f"---- Architecture: {hp.model_arch} | Algorithm: {algo_type} ------")

    env = RecoEnv(
        data_dir=str(data_dir),
        val_ratio=hp.val_ratio,
        keep_top_n=hp.keep_top_n,
        repeat_penalty=hp.repeat_penalty,
        popularity_penalty=hp.popularity_penalty,
        history_window=hp.history_window,
        is_gru=is_gru,
    )

    agent = Agent(env, hp)

    print("Loading Train/Validation Data...")
    val_data = prepare_evaluation_data(
        data_dir,
        val_ratio=hp.val_ratio,
        keep_top_n=hp.keep_top_n,
        min_ratings=hp.min_ratings,
        history_window=hp.history_window,
        is_gru=is_gru,
    )

    stats = agent.train(
        n_updates=hp.base_n_updates, val_data=val_data, save_path=model_save_path
    )

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.set_title(f"Training: {model_name_str}")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss", color="red")
    ax1.plot(stats["loss"], color="red", alpha=0.3, label="Loss")
    ax1.legend(loc="upper left")

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel("NDCG@10", color="blue")
    ax1_twin.plot(stats["steps"], stats["ndcg"], color="blue", marker=".", label="NDCG")
    ax1_twin.legend(loc="upper right")

    ax2.set_title("Q-Value Stability Check")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Avg Predicted Reward")
    ax2.plot(stats["steps"], stats["q_val"], color="green", label="Avg Q")
    ax2.axhline(0, linestyle="--", color="black", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plot_save_path)
    print(f"Training Complete!")
    print(f" > Model saved to: {model_save_path}")
    print(f" > Plot saved to:  {plot_save_path}")
    print(f" > Log saved to:   {log_dir / log_filename}")


if __name__ == "__main__":
    main()
