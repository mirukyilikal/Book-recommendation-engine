
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

BOOKS_PATH = DATA_DIR / "BX_Books.csv"
USERS_PATH = DATA_DIR / "BX-Users.csv"
RATINGS_PATH = DATA_DIR / "BX-Book-Ratings.csv"

HYBRID_ALPHA = 0.25

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Book Recommendation Engine")
    print("--------------------------")
    books = pd.read_csv(
        BOOKS_PATH,
        sep=";",
        encoding="latin-1",
        on_bad_lines="skip"
    )

    users = pd.read_csv(
        USERS_PATH,
        sep=";",
        encoding="latin-1",
        on_bad_lines="skip"
    )

    ratings = pd.read_csv(
        RATINGS_PATH,
        sep=";",
        encoding="latin-1",
        on_bad_lines="skip"
    )

    print("Books:", books.shape)
    print("Users:", users.shape)
    print("Ratings:", ratings.shape)

    print(
        "Datasets loaded. Training pipeline still needs "
        "the notebook's preprocessing and model-building steps."
    )


if __name__ == "__main__":
    main()
