from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BookWise | Book Recommendation Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
HYBRID_ALPHA_DEFAULT = 0.25
DEFAULT_TOP_N = 10


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 45%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        color: white;
        background: linear-gradient(120deg, #14213d 0%, #244b78 65%, #3978a8 100%);
        margin-bottom: 1.4rem;
    }
    .hero h1 {
        color: white;
        font-size: 2.45rem;
        margin-bottom: 0.4rem;
    }
    .hero p {
        color: #e7eef8;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    .metric-card {
        background: white;
        border: 1px solid #e5eaf1;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        min-height: 105px;
        box-shadow: 0 3px 12px rgba(20, 33, 61, 0.04);
    }
    .metric-label {
        color: #64748b;
        font-size: 0.88rem;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        color: #14213d;
        font-size: 1.55rem;
        font-weight: 700;
    }
    .book-card {
        border: 1px solid #e5eaf1;
        background: #ffffff;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
    }
    .book-card h4 {
        color: #14213d;
        margin: 0 0 0.4rem 0;
        font-size: 1.05rem;
    }
    .book-meta {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .score {
        color: #176b55;
        font-weight: 700;
        font-size: 0.88rem;
    }
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def normalize_isbn(value) -> str:
    """Normalize ISBN values for reliable matching across saved artifacts."""
    return str(value).strip().upper()


def normalize_title(title) -> str:
    """Normalize a title to help avoid recommending another edition of it."""
    title = str(title).lower().strip()
    title = re.sub(r"\([^)]*\)", " ", title)
    title = re.sub(r"\[[^\]]*\]", " ", title)
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


@st.cache_resource(show_spinner="Loading recommendation models...")
def load_artifacts(model_dir_string: str):
    """Load model artifacts once per Streamlit session."""
    model_dir = Path(model_dir_string)

    required_files = {
        "metadata": "book_metadata.joblib",
        "tfidf_matrix": "tfidf_matrix.joblib",
        "cf_similarity": "cf_similarity_matrix.joblib",
        "cf_isbns": "cf_book_isbns.joblib",
        "cf_position": "cf_position.joblib",
        "content_position": "content_position.joblib",
        "common_isbns": "common_isbns.joblib",
        "alpha": "hybrid_alpha.joblib",
    }

    missing = [
        filename
        for filename in required_files.values()
        if not (model_dir / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts: "
            + ", ".join(missing)
            + ". Run the artifact-saving notebook cell first."
        )

    loaded = {
        key: joblib.load(model_dir / filename)
        for key, filename in required_files.items()
    }

    metadata = loaded["metadata"].copy()
    metadata["ISBN"] = metadata["ISBN"].map(normalize_isbn)
    metadata = metadata.drop_duplicates(subset="ISBN").reset_index(drop=True)

    # Saved mappings should match the matrices' row order.
    cf_position = {
        normalize_isbn(isbn): int(pos)
        for isbn, pos in loaded["cf_position"].items()
    }
    content_position = {
        normalize_isbn(isbn): int(pos)
        for isbn, pos in loaded["content_position"].items()
    }

    common_isbns = [
        normalize_isbn(isbn)
        for isbn in loaded["common_isbns"]
        if normalize_isbn(isbn) in cf_position
        and normalize_isbn(isbn) in content_position
    ]

    metadata["Normalized-Title"] = metadata["Book-Title"].fillna("").map(
        normalize_title
    )

    return {
        "metadata": metadata,
        "tfidf_matrix": loaded["tfidf_matrix"],
        "cf_similarity": np.asarray(loaded["cf_similarity"]),
        "cf_isbns": [normalize_isbn(x) for x in loaded["cf_isbns"]],
        "cf_position": cf_position,
        "content_position": content_position,
        "common_isbns": common_isbns,
        "alpha": float(loaded["alpha"]),
    }


def percentile_scores(values: np.ndarray) -> np.ndarray:
    """Convert a score vector to percentile ranks in the range 0..1."""
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)


def recommend_similar_books(
    selected_isbn: str,
    artifacts: dict,
    top_n: int = 10,
    alpha: float | None = None,
) -> pd.DataFrame:
    """
    Recommend books similar to a selected book.

    This is an item-to-item hybrid: it combines collaborative similarity
    and content similarity for the selected seed book. It does not require
    a signed-in user or a user-rating history.
    """
    alpha = artifacts["alpha"] if alpha is None else float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    selected_isbn = normalize_isbn(selected_isbn)
    cf_position = artifacts["cf_position"]
    content_position = artifacts["content_position"]
    common_isbns = artifacts["common_isbns"]
    metadata = artifacts["metadata"]
    cf_similarity = artifacts["cf_similarity"]
    tfidf_matrix = artifacts["tfidf_matrix"]

    if selected_isbn not in cf_position or selected_isbn not in content_position:
        return pd.DataFrame()

    candidate_isbns = [
        isbn for isbn in common_isbns
        if isbn != selected_isbn
    ]
    if not candidate_isbns:
        return pd.DataFrame()

    seed_cf_pos = cf_position[selected_isbn]
    candidate_cf_pos = np.array(
        [cf_position[isbn] for isbn in candidate_isbns],
        dtype=int,
    )
    cf_scores = cf_similarity[seed_cf_pos, candidate_cf_pos]

    seed_content_pos = content_position[selected_isbn]
    candidate_content_pos = np.array(
        [content_position[isbn] for isbn in candidate_isbns],
        dtype=int,
    )

    # One seed-book vector compared with every candidate-book vector.
    content_scores = (
        tfidf_matrix[candidate_content_pos]
        @ tfidf_matrix[seed_content_pos].T
    )
    content_scores = np.asarray(content_scores).ravel()

    cf_rank = percentile_scores(np.asarray(cf_scores).ravel())
    content_rank = percentile_scores(content_scores)
    hybrid_scores = alpha * cf_rank + (1.0 - alpha) * content_rank

    results = pd.DataFrame(
        {
            "ISBN": candidate_isbns,
            "CF_Rank_Score": cf_rank,
            "Content_Rank_Score": content_rank,
            "Hybrid_Score": hybrid_scores,
        }
    )

    display_columns = [
        "ISBN",
        "Book-Title",
        "Book-Author",
        "Publisher",
        "Year-Of-Publication",
        "CF_Rank_Score",
        "Content_Rank_Score",
        "Hybrid_Score",
        "Normalized-Title",
    ]

    metadata_columns = [
        col for col in display_columns
        if col in metadata.columns
    ]
    results = results.merge(
        metadata[metadata_columns].drop_duplicates("ISBN"),
        on="ISBN",
        how="left",
    )

    seed_row = metadata[metadata["ISBN"] == selected_isbn]
    seed_title = (
        seed_row.iloc[0]["Normalized-Title"]
        if not seed_row.empty
        else ""
    )

    results["Normalized-Title"] = results["Normalized-Title"].fillna("")
    results = results[
        (results["Normalized-Title"] != "")
        & (results["Normalized-Title"] != seed_title)
    ]

    # Avoid recommending multiple editions of the same normalized title.
    results = (
        results.sort_values("Hybrid_Score", ascending=False)
        .drop_duplicates("Normalized-Title")
        .head(top_n)
        .reset_index(drop=True)
    )

    return results


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 📚 BookWise")
    st.caption("Hybrid Book Recommendation Engine")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Discover Books", "About the Project"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Recommendation settings**")
    top_n = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=DEFAULT_TOP_N,
        step=5,
    )
    st.caption(
        "The model combines book-rating similarity with title and author metadata."
    )


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

try:
    artifacts = load_artifacts(str(MODEL_DIR))
except Exception as exc:
    st.markdown(
        """
        <div class="hero">
            <h1>BookWise</h1>
            <p>Discover your next great read with hybrid book recommendations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error("The recommendation models could not be loaded.")
    st.code(str(exc))
    st.markdown(
        """
        **Check the following before running the app:**

        1. Your project contains a `models` folder.
        2. The folder contains the required `.joblib` artifacts.
        3. The artifacts were saved from the notebook using the filenames
           listed in the error above.
        4. Run the app from your project folder with `streamlit run app.py`.
        """
    )
    st.stop()


# ============================================================
# PAGE: DISCOVER BOOKS
# ============================================================

if page == "Discover Books":
    st.markdown(
        """
        <div class="hero">
            <h1>Find your next great read.</h1>
            <p>
                Choose a book you enjoy and discover recommendations
                based on book similarity and reader-rating patterns.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metadata = artifacts["metadata"].copy()
    metadata["Book-Title"] = metadata["Book-Title"].fillna("Untitled")
    metadata["Book-Author"] = metadata["Book-Author"].fillna("Unknown author")
    metadata["Publisher"] = metadata["Publisher"].fillna("Unknown publisher")

    metric1, metric2, metric3 = st.columns(3)
    with metric1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Books in metadata</div>'
            f'<div class="metric-value">{len(metadata):,}</div></div>',
            unsafe_allow_html=True,
        )
    with metric2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Books supported by both models</div>'
            f'<div class="metric-value">{len(artifacts["common_isbns"]):,}</div></div>',
            unsafe_allow_html=True,
        )
    with metric3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Content model weight</div>'
            f'<div class="metric-value">{(1 - artifacts["alpha"]) * 100:.0f}%</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("Choose a book")
    st.write("Search by title or author, then select a book to get recommendations.")

    search_query = st.text_input(
        "Search books",
        placeholder="For example: Pride and Prejudice, Jane Austen...",
    ).strip().lower()

    supported_isbns = set(artifacts["common_isbns"])
    selectable_books = metadata[
        metadata["ISBN"].isin(supported_isbns)
    ].copy()

    if search_query:
        mask = (
            selectable_books["Book-Title"].str.lower().str.contains(
                re.escape(search_query), na=False
            )
            | selectable_books["Book-Author"].str.lower().str.contains(
                re.escape(search_query), na=False
            )
        )
        search_results = selectable_books[mask].head(300)
    else:
        search_results = selectable_books.sort_values("Book-Title").head(300)

    if search_results.empty:
        st.warning(
            "No supported books matched that search. Try a shorter title or author name."
        )
    else:
        def format_book(row):
            return f'{row["Book-Title"]} — {row["Book-Author"]} [{row["ISBN"]}]'

        book_options = search_results["ISBN"].tolist()
        option_labels = {
            row["ISBN"]: format_book(row)
            for _, row in search_results.iterrows()
        }

        selected_isbn = st.selectbox(
            "Available books",
            options=book_options,
            format_func=lambda isbn: option_labels.get(isbn, isbn),
        )

        selected_rows = metadata[metadata["ISBN"] == selected_isbn]
        if not selected_rows.empty:
            selected = selected_rows.iloc[0]
            with st.container(border=True):
                left, right = st.columns([3, 1])
                with left:
                    st.markdown("### " + str(selected["Book-Title"]))
                    st.write(f'**Author:** {selected["Book-Author"]}')
                    st.write(f'**Publisher:** {selected["Publisher"]}')
                    year = selected.get("Year-Of-Publication", "")
                    st.write(f"**Publication year:** {year}")
                    st.caption(f"ISBN: {selected_isbn}")
                with right:
                    st.metric("Hybrid weight", f"{artifacts['alpha']:.2f} CF")

        if st.button("Recommend books", type="primary", use_container_width=True):
            with st.spinner("Finding books you may enjoy..."):
                recommendations = recommend_similar_books(
                    selected_isbn=selected_isbn,
                    artifacts=artifacts,
                    top_n=top_n,
                )
            st.session_state["bookwise_recommendations"] = recommendations
            st.session_state["bookwise_seed_isbn"] = selected_isbn

        recommendations = st.session_state.get("bookwise_recommendations")
        saved_seed = st.session_state.get("bookwise_seed_isbn")

        if recommendations is not None and saved_seed == selected_isbn:
            st.divider()
            st.subheader("Recommended for you")

            if recommendations.empty:
                st.info(
                    "No recommendations could be generated for this book. "
                    "Try selecting another title."
                )
            else:
                st.caption(
                    "Rank scores compare candidates within this recommendation run; "
                    "they are not probabilities that you will like a book."
                )
                for start in range(0, len(recommendations), 2):
                    columns = st.columns(2)
                    for column, (_, row) in zip(
                        columns,
                        recommendations.iloc[start:start + 2].iterrows(),
                    ):
                        with column:
                            title = str(row.get("Book-Title", "Untitled"))
                            author = str(row.get("Book-Author", "Unknown author"))
                            publisher = str(row.get("Publisher", "Unknown publisher"))
                            year = row.get("Year-Of-Publication", "")
                            score = float(row["Hybrid_Score"])

                            st.markdown(
                                f"""
                                <div class="book-card">
                                    <h4>{title}</h4>
                                    <div class="book-meta">by {author}</div>
                                    <div class="book-meta">{publisher} · {year}</div>
                                    <div class="score">Hybrid rank score: {score:.4f}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            with st.expander("View score details"):
                                st.write(
                                    f"Collaborative rank: "
                                    f'{row["CF_Rank_Score"]:.4f}'
                                )
                                st.write(
                                    f"Content rank: "
                                    f'{row["Content_Rank_Score"]:.4f}'
                                )
                                st.write(f"ISBN: {row['ISBN']}")


# ============================================================
# PAGE: ABOUT THE PROJECT
# ============================================================

else:
    st.markdown(
        """
        <div class="hero">
            <h1>About BookWise</h1>
            <p>A machine-learning project for exploring book recommendations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("How it works")
    st.write(
        "BookWise combines two recommendation signals. Collaborative filtering "
        "uses patterns in book-rating data, while content-based filtering uses "
        "book metadata such as title and author. Their percentile-ranked scores "
        "are combined to create a final ranking."
    )

    st.subheader("Hybrid scoring")
    alpha = artifacts["alpha"]
    st.latex(
        rf"S_{{hybrid}} = {alpha:.2f}S_{{CF}} + {1-alpha:.2f}S_{{content}}"
    )
    st.write(
        f"The current configuration uses {alpha:.0%} collaborative-filtering "
        f"weight and {(1-alpha):.0%} content-based weight. This setting was "
        "selected from a preliminary experiment on a 30-user sample and should "
        "be validated on a larger, consistently evaluated sample."
    )

    st.subheader("Dataset")
    st.write(
        "The project uses the Book-Crossing dataset, including book metadata, "
        "users, and explicit book ratings. Zero ratings represent implicit or "
        "unspecified ratings in the original data and were excluded from the "
        "explicit-rating model-development pipeline."
    )

    st.subheader("Limitations")
    st.markdown(
        """
        - Recommendations are based on the available metadata and rating data.
        - New or unsupported books may not be eligible for hybrid recommendations.
        - The current app recommends books similar to a selected book; it does
          not yet create a personalized profile from a logged-in user's full history.
        - Evaluation results are preliminary and should not be interpreted as
          production-level performance.
        """
    )

    st.subheader("Technology")
    st.write("Python · Pandas · NumPy · Scikit-learn · SciPy · Joblib · Streamlit")
