"""
============================================================
Dataset Utilities – MovieLens Data Loader & Preprocessor
============================================================
This module handles loading the MovieLens dataset and
preparing it for the recommendation engine.

MovieLens 100K Dataset Structure:
  u.data   – userId, movieId, rating, timestamp
  u.item   – movieId, title, release_date, genres...
  u.genre  – genre list

Download from: https://grouplens.org/datasets/movielens/100k/
Place files in: /dataset/ml-100k/
============================================================
"""

import pandas as pd
import numpy  as np
import os
import re


class MovieLensLoader:
    """
    Loads and preprocesses the MovieLens 100K dataset.

    Usage:
        loader = MovieLensLoader("dataset/ml-100k")
        movies, ratings = loader.load()
        X_train, X_test = loader.train_test_split(ratings)
    """

    GENRE_COLS = [
        "unknown","Action","Adventure","Animation","Children",
        "Comedy","Crime","Documentary","Drama","Fantasy",
        "Film-Noir","Horror","Musical","Mystery","Romance",
        "Sci-Fi","Thriller","War","Western"
    ]

    def __init__(self, data_dir: str = "dataset/ml-100k"):
        self.data_dir = data_dir

    def load_ratings(self) -> pd.DataFrame:
        """
        Load the u.data ratings file.

        Columns: user_id, content_id (movie_id), rating, timestamp
        Rating scale: 1–5
        """
        path = os.path.join(self.data_dir, "u.data")
        df   = pd.read_csv(path, sep="\t",
                           names=["user_id","content_id","rating","timestamp"])

        # Drop timestamp – not needed for our models
        df = df.drop(columns=["timestamp"])

        print(f"✅ Loaded {len(df):,} ratings from {df['user_id'].nunique()} users")
        return df

    def load_movies(self) -> pd.DataFrame:
        """
        Load the u.item movie metadata file.

        Columns: content_id, title, year, genre (combined), genre_flags...
        """
        path = os.path.join(self.data_dir, "u.item")
        cols = ["content_id","title","release_date","video_date","imdb_url"] + self.GENRE_COLS

        df = pd.read_csv(path, sep="|", names=cols,
                         encoding="latin-1", on_bad_lines="skip")

        # ── Extract year from title e.g. "Toy Story (1995)" → 1995 ──
        df["year"] = df["title"].str.extract(r"\((\d{4})\)").astype(float)

        # ── Build genre string from binary columns ────────────────
        def build_genre_str(row):
            genres = [g for g in self.GENRE_COLS if row.get(g, 0) == 1]
            return ",".join(genres) if genres else "Other"

        df["genre"] = df.apply(build_genre_str, axis=1)

        # ── Clean title (remove year) ─────────────────────────────
        df["title_clean"] = df["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)

        # Keep only needed columns
        df = df[["content_id","title_clean","year","genre","imdb_url"]].copy()
        df = df.rename(columns={"title_clean": "title"})

        print(f"✅ Loaded {len(df):,} movies with {df['genre'].nunique()} unique genre combos")
        return df

    def load(self):
        """
        Load and merge movies + ratings.
        Returns: (movies_df, ratings_df)
        """
        movies  = self.load_movies()
        ratings = self.load_ratings()
        return movies, ratings

    def train_test_split(self, ratings_df: pd.DataFrame,
                          test_ratio: float = 0.2,
                          random_state: int = 42):
        """
        Split ratings into train and test sets.
        Uses a per-user split: each user's most recent ratings → test set.

        Args:
            ratings_df:   Full ratings DataFrame
            test_ratio:   Fraction of ratings per user to hold out
            random_state: Seed for reproducibility

        Returns:
            (train_df, test_df)
        """
        np.random.seed(random_state)

        train_parts = []
        test_parts  = []

        for user_id, group in ratings_df.groupby("user_id"):
            n_test = max(1, int(len(group) * test_ratio))

            # Randomly sample test ratings
            test_idx  = np.random.choice(group.index, n_test, replace=False)
            train_idx = [i for i in group.index if i not in set(test_idx)]

            train_parts.append(group.loc[train_idx])
            test_parts.append(group.loc[test_idx])

        train_df = pd.concat(train_parts).reset_index(drop=True)
        test_df  = pd.concat(test_parts).reset_index(drop=True)

        print(f"✅ Train: {len(train_df):,} ratings | Test: {len(test_df):,} ratings")
        return train_df, test_df

    @staticmethod
    def basic_stats(ratings_df: pd.DataFrame):
        """Print summary statistics about the ratings dataset."""
        print("\n📊 Dataset Statistics")
        print(f"  Total ratings    : {len(ratings_df):,}")
        print(f"  Unique users     : {ratings_df['user_id'].nunique():,}")
        print(f"  Unique movies    : {ratings_df['content_id'].nunique():,}")
        print(f"  Rating range     : {ratings_df['rating'].min()} – {ratings_df['rating'].max()}")
        print(f"  Average rating   : {ratings_df['rating'].mean():.2f}")
        print(f"  Ratings/user     : {ratings_df.groupby('user_id')['rating'].count().mean():.1f} avg")
        print(f"  Sparsity         : {(1 - len(ratings_df)/(ratings_df['user_id'].nunique()*ratings_df['content_id'].nunique()))*100:.1f}%")


def preprocess_for_engine(movies_df: pd.DataFrame) -> list[dict]:
    """
    Convert a movies DataFrame into the list-of-dicts format
    expected by the ContentBasedFilter model.

    Args:
        movies_df: DataFrame with columns: content_id, title, genre, year

    Returns:
        List of content dicts with 'id', 'title', 'genre', 'year', 'description'
    """
    content_list = []
    for _, row in movies_df.iterrows():
        content_list.append({
            "id":          int(row["content_id"]),
            "title":       str(row["title"]),
            "genre":       str(row.get("genre", "")),
            "year":        int(row["year"]) if not pd.isna(row.get("year")) else 0,
            "description": str(row.get("description", "")),
            "avg_rating":  float(row.get("avg_rating", 0)),
        })
    return content_list


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    loader  = MovieLensLoader()
    try:
        movies, ratings = loader.load()
        MovieLensLoader.basic_stats(ratings)
        train, test = loader.train_test_split(ratings)
    except FileNotFoundError:
        print("⚠️  MovieLens dataset not found in dataset/ml-100k/")
        print("   Download from: https://grouplens.org/datasets/movielens/100k/")
        print("   The system will use built-in sample data instead.")
