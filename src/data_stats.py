from collections import Counter

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


if __name__ == "__main__":
    loader = MovieLensLoader("data/ml-latest-small").load_all()
    stats = MovieLensStats(loader)

    print("Overview:", stats.dataset_overview())
    print("Rating summary:", stats.rating_summary())
    print("Rating distribution:")
    print(stats.rating_distribution())
    print("Top genres:", stats.top_genres())
    print("Average tags per movie:", stats.average_tags_per_movie())

