# Book Recommendation Engine

A personalized book recommendation system built with Python and machine learning. The project explores multiple recommendation techniques and combines collaborative filtering with content-based filtering in a hybrid recommendation model.

## Project Overview

The system uses the Book-Crossing dataset to recommend books based on user rating patterns and book metadata.

### Recommendation Methods

- **Popularity-based filtering:** Ranks books using rating statistics.
- **Singular Value Decomposition (SVD):** Learns latent user and book preferences.
- **Item-based collaborative filtering:** Uses similarity between book-rating patterns.
- **Content-based filtering:** Uses book titles, authors, and publishers.
- **Hybrid recommendation:** Combines collaborative and content-based ranking scores.

## Dataset

The project uses the Book-Crossing dataset, including book metadata, user information, and book ratings.

The original dataset contains:
- 271,379 book records
- 278,858 user records
- 1,149,780 rating records

Explicit ratings range from 1 to 10. Zero ratings are excluded from the explicit-rating recommendation pipeline.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- SciPy
- Joblib
- Streamlit

## Evaluation

Models are evaluated using recommendation-ranking metrics:
- Precision@10
- Recall@10
- NDCG@10

The current preliminary experiment tested a hybrid weight of 0.25 for collaborative filtering and 0.75 for content-based filtering. It achieved NDCG@10 of 0.1254 on a 30-user evaluation sample. These results are preliminary and should be revalidated with consistent candidate sets and a larger evaluation sample.

## Project Structure

```text
book-recommendation-engine/
├── data/
├── models/
├── notebooks/
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Setup

Create and activate a Python virtual environment, then install the required packages:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

After the training script and application have been completed:

```bash
python train_model.py
streamlit run app.py
```

## Current Status

The recommendation models have been developed and evaluated in a notebook. The standalone training pipeline and Streamlit interface are being prepared.

## Future Improvements

- Standardize candidate sets across all model evaluations.
- Evaluate on more users and test different hybrid weights.
- Improve recommendation diversity.
- Handle new users and books with limited data.
- Build an interactive book-search and recommendation interface.

## Data Source

Book-Crossing dataset, originally collected by Cai-Nicolas Ziegler and collaborators. Please review the dataset's original distribution terms before redistributing the data.