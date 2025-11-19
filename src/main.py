import os
from pathlib import Path
import matplotlib.pyplot as plt
from env.recommender_env import RecoEnv        
from agent.dqn_agent import Agent
from utils.hyperparameters import Hyperparameters
import torch


if __name__ == "__main__":
    env = RecoEnv(
        val_ratio=0.2,
        repeat_penalty=0.1,
        gamma=0.99,
    )
    hp = Hyperparameters()

    agent = Agent(env, hp)
    losses = agent.train(n_updates=4_000)



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
    plt.show()
    fig_path = PROJECT_ROOT / "models" / "dqn_loss.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved loss figure to {fig_path}")

    #save model
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    model_path = PROJECT_ROOT / "models" / "dqn_movielens.pt"
    torch.save(agent.onlineDQN.state_dict(), model_path)
    print(f"Saved DQN to {model_path}")