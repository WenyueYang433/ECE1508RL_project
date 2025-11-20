from __future__ import annotations
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
    
    #get years, normalizes to 0,1, to avoid over large feature issue (compare to genre 0,1)
    def _process_years(self, titles: pd.Series) -> pd.Series:
        years = titles.apply(self._extract_year).astype("float32")

        #impute missing years(0.0) with the median of valid years
        #to avoid gradients of 0.0 movies be massive compared to normal movies
        valid_mask = years > 0
        if valid_mask.any():
            median_year = years[valid_mask].median()
            years[~valid_mask] = median_year

        #normalize to 0, 1
        min_y = years.min()
        max_y = years.max()
        
        if max_y > min_y:
            years = (years - min_y) / (max_y - min_y)
        else:
            years[:] = 0.0 # corner case: all movies from same year
            
        return years
        

    def encode_movies(self) -> pd.DataFrame:
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

    def encode_users(self) -> pd.DataFrame:
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
    loader = MovieLensLoader("data/ml-latest-small").load_all()
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

