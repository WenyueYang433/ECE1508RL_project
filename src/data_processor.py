from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
from data_loader import MovieLensLoader


class DatasetPrep:
    def __init__(self, loader: MovieLensLoader):
        loader.validate()
        self.movies = loader.movies.copy()
        self.ratings = loader.ratings.copy()
        self._movie_map = None
        self._user_ids = None

    @staticmethod
    def _extract_year(title: str) -> float:
        if not isinstance(title, str):
            return 0.0
        start = title.rfind("(")
        end = title.rfind(")")
        if start == -1 or end == -1:
            return 0.0
        segment = title[start + 1 : end]
        return float(segment) if segment.isdigit() else 0.0

    def _process_years(self, titles: pd.Series) -> pd.Series:
        years = titles.apply(self._extract_year).astype("float32")
        valid_mask = years > 0
        if valid_mask.any():
            median_year = years[valid_mask].median()
            years[~valid_mask] = median_year

        min_y = years.min()
        max_y = years.max()

        if max_y > min_y:
            years = (years - min_y) / (max_y - min_y)
        else:
            years[:] = 0.0

        return years

    def encode_movies(self, keep_top_n: int = 1000) -> pd.DataFrame:
        pop_counts = self.ratings["movieId"].value_counts()
        print(f"Total unique movies with ratings before filtering: {len(pop_counts)}")
        top_ids = pop_counts.nlargest(keep_top_n).index
        self.movies = self.movies[self.movies["movieId"].isin(top_ids)].copy()
        self.ratings = self.ratings[self.ratings["movieId"].isin(top_ids)].copy()

        print(
            f"Filtered to top {keep_top_n} movies. Remaining Ratings: {len(self.ratings)}"
        )

        genres = self.movies["genres"].str.get_dummies("|").astype("float32")

        self.movies["year"] = self._process_years(self.movies["title"])

        codes = pd.Categorical(self.movies["movieId"])
        series = pd.Series(range(len(codes.categories)), index=codes.categories)
        self._movie_map = series

        base = pd.DataFrame(
            {
                "movie_key": self.movies["movieId"].map(series).astype("int32"),
                "movieId": self.movies["movieId"],
                "title": self.movies["title"],
                "year": self.movies["year"],
            }
        )
        return pd.concat([base, genres], axis=1)

    def encode_users(self, min_ratings: int = 5) -> pd.DataFrame:
        user_counts = self.ratings["userId"].value_counts()
        valid_users = user_counts[user_counts >= min_ratings].index
        before_count = len(self.ratings)
        self.ratings = self.ratings[self.ratings["userId"].isin(valid_users)].copy()
        print(
            f"Filtered users with < {min_ratings} ratings. Ratings dropped: {before_count - len(self.ratings)}"
        )

        users = sorted(self.ratings["userId"].unique())
        self._user_ids = users
        return pd.DataFrame(
            {
                "user_key": users,
                "userId": users,
            }
        )

    def encode_ratings(self) -> pd.DataFrame:
        if self._movie_map is None:
            raise RuntimeError("Call encode_users() and encode_movies() first.")

        mapped = self.ratings.copy()
        mapped["user_key"] = mapped["userId"].astype("int32")
        mapped["movie_key"] = mapped["movieId"].map(self._movie_map).astype("int32")
        mapped["timestamp"] = pd.to_datetime(mapped["timestamp"], unit="s")
        return mapped[["user_key", "movie_key", "rating", "timestamp"]]

    def temporal_split(
        self, ratings: pd.DataFrame, val_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ordered = ratings.sort_values("timestamp")
        cutoff = int((1 - val_ratio) * len(ordered))
        return ordered.iloc[:cutoff], ordered.iloc[cutoff:]

    def feature_sizes(self) -> Dict[str, int]:
        genre_dim = (
            self.movies["genres"].str.get_dummies("|").shape[1]
            if self.movies is not None
            else 0
        )
        return {
            "n_users": len(self._user_ids) if self._user_ids is not None else 0,
            "n_movies": len(self._movie_map) if self._movie_map is not None else 0,
            "n_genres": genre_dim,
        }


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    data_dir = PROJECT_ROOT / "data" / "ml-latest-small"
    loader = MovieLensLoader(data_dir).load_all()
    prep = DatasetPrep(loader)
    movie_features = prep.encode_movies()
    user_features = prep.encode_users()
    rating_table = prep.encode_ratings()
    train_df, val_df = prep.temporal_split(rating_table)
    print("Movies:", movie_features.shape)
    print("Users:", user_features.shape)
    print("Ratings:", rating_table.shape)
    print("Train/Val:", len(train_df), len(val_df))
    print("Feature dims:", prep.feature_sizes())
    print("\nMovie sample:")
    print(movie_features.head(3))
    print("\nUser sample:")
    print(user_features.head(3))
    print("\nRating sample:")
    print(rating_table.head(3))
