import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import matplotlib.pyplot as plt
from env.recommender_env import RecoEnv        
from agent.dqn_agent import Agent
from utils.hyperparameters import Hyperparameters
import torch


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    data_dir = PROJECT_ROOT / "data" / "ml-latest-small"
    env = RecoEnv(
        data_dir=str(data_dir),
        val_ratio=0.2,
        repeat_penalty=0.1,
        keep_top_n=1000,
        popularity_penalty=0.01,
    )
    hp = Hyperparameters()

    agent = Agent(env, hp)
    # Increased updates to give the agent more training steps
    losses = agent.train(n_updates=2000)



    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    #save png
    steps = range(1, len(losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, marker="", linewidth=1)
    plt.xlabel("Update step")
    plt.ylabel("Loss")
    plt.title("DQN training loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = PROJECT_ROOT / "models" / "dqn_loss.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    print(f"Saved loss figure to {fig_path}")
    plt.show()
    

    #save model
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    model_path = PROJECT_ROOT / "models" / "dqn_movielens.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.onlineDQN.state_dict(), model_path)
    print(f"Saved DQN to {model_path}")