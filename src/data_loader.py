from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "ratings.csv": ["userId", "movieId", "rating", "timestamp"],
    "movies.csv": ["movieId", "title", "genres"],
    "tags.csv": ["userId", "movieId", "tag", "timestamp"],
    "links.csv": ["movieId", "imdbId", "tmdbId"],
}


class MovieLensLoader:
    def __init__(self, data_dir: str):
        self.root = Path(data_dir)
        self.ratings = self.movies = self.tags = self.links = None

    def load_all(self):
        self.ratings = self._load("ratings.csv")
        self.movies = self._load("movies.csv")
        self.tags = self._load("tags.csv")
        self.links = self._load("links.csv")
        return self

    def _load(self, name: str) -> pd.DataFrame:
        path = self.root / name
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        columns = REQUIRED_COLUMNS.get(name, [])
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

        df = df.dropna(subset=columns or None)
        return df.drop_duplicates().reset_index(drop=True)

    def validate(self):
        for name, df in {
            "ratings": self.ratings,
            "movies": self.movies,
            "tags": self.tags,
            "links": self.links,
        }.items():
            if df is None or df.empty:
                raise ValueError(f"{name} not ready")


if __name__ == "__main__":
    MovieLensLoader("data/ml-latest-small").load_all().validate()
    print("MovieLens data loaded.")
