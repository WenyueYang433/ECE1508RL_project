import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import matplotlib.pyplot as plt

from env.recommender_env import RecoEnv        
from agent.dqn_agent import Agent
from utils.hyperparameters import Hyperparameters
from evaluation import prepare_evaluation_data

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    
    # Load Config
    hp = Hyperparameters()
    
    data_dir = PROJECT_ROOT / hp.data_rel_path
    model_save_path = PROJECT_ROOT / hp.model_base
    plot_save_path = PROJECT_ROOT / hp.plot_summary
    
    env = RecoEnv(
        data_dir=str(data_dir), 
        val_ratio=hp.val_ratio, 
        keep_top_n=hp.keep_top_n, 
        repeat_penalty=0.1,      
        popularity_penalty=0.01
    )
    
    agent = Agent(env, hp)

    print("Loading Validation Data for Live Monitoring...")
    val_data = prepare_evaluation_data(
        data_dir, 
        val_ratio=hp.val_ratio, 
        keep_top_n=hp.keep_top_n, 
        min_ratings=hp.min_ratings
    )
    
    # Run Training
    stats = agent.train(
        n_updates=hp.base_n_updates, 
        val_data=val_data, 
        save_path=model_save_path
    )

    ##Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left Plot: Loss vs NDCG
    ax1.set_title(f"Training Dynamics (LR={hp.learning_rate})")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss (MSE)", color='red')
    ax1.plot(stats["loss"], color='red', alpha=0.3, label='Loss')
    ax1.legend(loc='upper left')
    
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel("NDCG@10", color='blue')
    ax1_twin.plot(stats["steps"], stats["ndcg"], color='blue', marker='.', label='NDCG')
    ax1_twin.legend(loc='upper right')
    
    # Right Plot: Q-Value Stability
    ax2.set_title("Q-Value Stability Check")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Avg Predicted Reward")
    ax2.plot(stats["steps"], stats["q_val"], color='green', label='Avg Q')
    ax2.axhline(0, linestyle='--', color='black', alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(plot_save_path)
    print(f"Training Complete!")
    print(f" > Model saved to: {model_save_path}")
    print(f" > Plot saved to:  {plot_save_path}")

if __name__ == "__main__":
    main()