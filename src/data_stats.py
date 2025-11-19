from collections import Counter
import matplotlib.pyplot as plt
from data_loader import MovieLensLoader


class MovieLensStats:
    def __init__(self, loader: MovieLensLoader):
        self.loader = loader
        self.loader.validate()

    def dataset_overview(self):
        return {
            "n_users": int(self.loader.ratings["userId"].nunique()),
            "n_movies": int(self.loader.movies["movieId"].nunique()),
            "n_ratings": int(len(self.loader.ratings)),
            "n_tags": int(len(self.loader.tags)),
        }

    def rating_summary(self):
        ratings = self.loader.ratings["rating"]
        return {
            "avg_rating": float(ratings.mean()),
            "min_rating": float(ratings.min()),
            "max_rating": float(ratings.max()),
        }

    def rating_distribution(self):
        return (
            self.loader.ratings["rating"]
            .value_counts()
            .sort_index()
            .astype(int)
        )

    def top_genres(self, top_k=10):
        genre_counter = Counter()
        for genres in self.loader.movies["genres"].dropna():
            for genre in genres.split("|"):
                genre_counter[genre] += 1
        return genre_counter.most_common(top_k)

    def average_tags_per_movie(self):
        tag_counts = self.loader.tags.groupby("movieId")["tag"].count()
        if tag_counts.empty:
            return 0.0
        return float(tag_counts.mean())

def plot_rating_hist(ratings):
    plt.figure(figsize=(6, 4))
    plt.hist(
        ratings,
        bins=[0.5 + 0.5 * i for i in range(11)],
        color="#4C72B0",
        edgecolor="white",
    )
    plt.title("Rating distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()


def plot_top_genres(genre_counts):
    labels = [g for g, _ in genre_counts]
    values = [c for _, c in genre_counts]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values, color="#55A868")
    plt.xticks(rotation=45, ha="right")
    plt.title("Most common genres")
    plt.ylabel("Movies")
    plt.tight_layout()


if __name__ == "__main__":
    loader = MovieLensLoader("data/ml-latest-small").load_all()
    stats = MovieLensStats(loader)

    print("Overview:", stats.dataset_overview())
    print("Rating summary:", stats.rating_summary())
    print("Rating distribution:")
    print(stats.rating_distribution())
    print("Top genres:", stats.top_genres())
    print("Average tags per movie:", stats.average_tags_per_movie())

    plot_rating_hist(loader.ratings["rating"])
    plot_top_genres(stats.top_genres())
    plt.show()

