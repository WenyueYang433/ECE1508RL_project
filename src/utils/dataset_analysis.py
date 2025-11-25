import matplotlib.pyplot as plt
from pathlib import Path
from src.data_loader import MovieLensLoader

def plot_user_averages():
    
    data_dir = Path("data/ml-latest-small")
    loader = MovieLensLoader(str(data_dir)).load_all()

    user_avg_ratings = loader.ratings.groupby('userId')['rating'].mean()

    plt.figure(figsize=(10, 6))
    plt.hist(user_avg_ratings, bins=30, color='#4C72B0', edgecolor='white', alpha=0.8)
    
    # Formatting
    plt.title('Distribution of Average Ratings per User', fontsize=14)
    plt.xlabel('Average Rating', fontsize=12)
    plt.ylabel('Count of Users', fontsize=12)
    plt.axvline(user_avg_ratings.mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {user_avg_ratings.mean():.2f}')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

if __name__ == "__main__":
    plot_user_averages()