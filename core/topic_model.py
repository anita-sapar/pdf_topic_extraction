

# from bertopic import BERTopic
# from sklearn.feature_extraction.text import CountVectorizer
# from hdbscan import HDBSCAN

# def build_topic_model():
#     hdbscan_model=HDBSCAN(
#         min_cluster_size=10,
#         min_samples=5,
#         metric="euclidean",
#         cluster_selection_method="eom",
#         prediction_data=False
#     )


#     vectorizer_model=CountVectorizer(
#         stop_words='english',
#         ngram_range=(1,2),
#         min_df=2
#     )

#     topic_model= BERTopic(
#         hdbscan_model=hdbscan_model,
#         vectorizer_model=vectorizer_model,
#         calculate_probabilities=False,
#         verbose=True
#     )
#     return topic_model

# def extract_topics(topic_model,texts,embeddings):
#     """
#     Fits the topic model and returns:
#     - topic ids for each documnet
#     -topic info dataframe
#     -trained topic model

#     """

#     topics, _ = topic_model.fit_transform(
#         texts,
#         embeddings
#     )

#     topic_info = topic_model.get_topic_info()

#     return topics, topic_info,topic_model

#---------------------------------------------------------------


"""


Improved BERTopic pipeline with:
- Text cleaning (reduce noise)
- UMAP (cosine) for better separation on sentence embeddings
- HDBSCAN tuning to reduce outliers
- Better vectorizer for more meaningful topic labels (n-grams 1..3)
- Optional representation models (KeyBERT/MMR) with safe fallback
- Outlier reduction after training (convert -1 to nearest topics)
- Human-friendly topic labeling for UI sidebar


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import re
import numpy as np

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# Optional deps: umap-learn and hdbscan should be installed in your environment
from umap import UMAP
from hdbscan import HDBSCAN


# ----------------------------
# Configuration
# ----------------------------

@dataclass
class TopicModelConfig:
    # UMAP
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    random_state: int = 42

    # HDBSCAN
    min_cluster_size: int = 8
    min_samples: int = 2
    cluster_selection_method: str = "eom"

    # Vectorizer
    stop_words: str = "english"
    ngram_range: Tuple[int, int] = (1, 3)   # (1,3) gives better labels
    min_df: int = 2
    max_df: float = 0.9

    # BERTopic
    top_n_words: int = 10
    verbose: bool = True
    calculate_probabilities: bool = False

    # Cleaning
    min_text_len: int = 40          # drop too-short chunks (very noisy)
    drop_empty: bool = True

    # Outlier reduction strategy: "c-tf-idf" or "embeddings"
    reduce_outliers: bool = True
    outlier_strategy: str = "c-tf-idf"


# ----------------------------
# Text cleaning
# ----------------------------

_NOISE_PATTERNS = [
    r"\bpage\s*\d+\b",
    r"\bconfidential\b",
    r"\bcopyright\b",
    r"\ball rights reserved\b",
]

def clean_text(text: str) -> str:
    """Basic cleanup to remove boilerplate and normalize whitespace."""
    if not text:
        return ""

    t = text.strip()
    # Remove common noise patterns (headers/footers)
    for pat in _NOISE_PATTERNS:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def preprocess_texts(
    texts: List[str],
    min_len: int = 40,
    drop_empty: bool = True
) -> Tuple[List[str], List[int]]:
    """
    Cleans and filters texts. Returns:
    - cleaned_texts
    - kept_indices (mapping back to original positions)
    """
    cleaned = []
    kept_idx = []

    for i, t in enumerate(texts):
        ct = clean_text(t)
        if drop_empty and not ct:
            continue
        if len(ct) < min_len:
            continue
        cleaned.append(ct)
        kept_idx.append(i)

    return cleaned, kept_idx


# ----------------------------
# Build model
# ----------------------------

# def build_topic_model(config: Optional[TopicModelConfig] = None) -> BERTopic:
#     """
#     Build and return an improved BERTopic model.
#     """
#     if config is None:
#         config = TopicModelConfig()

#     umap_model = UMAP(
#         n_neighbors=config.umap_n_neighbors,
#         n_components=config.umap_n_components,
#         min_dist=config.umap_min_dist,
#         metric=config.umap_metric,
#         random_state=config.random_state,
#     )

#     # HDBSCAN is applied on UMAP-reduced space; euclidean is fine there.
#     hdbscan_model = HDBSCAN(
#         min_cluster_size=config.min_cluster_size,
#         min_samples=config.min_samples,
#         metric="euclidean",
#         cluster_selection_method=config.cluster_selection_method,
#         prediction_data=True,  # helpful for reduce_outliers + later transforms
#     )

#     vectorizer_model = CountVectorizer(
#         stop_words=config.stop_words,
#         ngram_range=config.ngram_range,
#         min_df=config.min_df,
#         max_df=config.max_df,
#     )

#     # Optional representation improvements (safe fallback)
#     representation_model = None
#     try:
#         # These imports exist in recent BERTopic versions
#         from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
#         representation_model = {
#             "KeyBERT": KeyBERTInspired(),
#             "MMR": MaximalMarginalRelevance(diversity=0.3),
#         }
#     except Exception:
#         # Not available in older versions; model still works fine without it.
#         representation_model = None

#     topic_model = BERTopic(
#         umap_model=umap_model,
#         hdbscan_model=hdbscan_model,
#         vectorizer_model=vectorizer_model,
#         representation_model=representation_model,
#         top_n_words=config.top_n_words,
#         verbose=config.verbose,
#         calculate_probabilities=config.calculate_probabilities,
#     )
#     return topic_model


# from bertopic import BERTopic
# from sklearn.feature_extraction.text import CountVectorizer
# from hdbscan import HDBSCAN
# from umap import UMAP

def build_topic_model(embedding_model=None):
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=8,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9
    )

    representation_model = None
    if embedding_model is not None:
        try:
            from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
            representation_model = {
                "KeyBERT": KeyBERTInspired(),
                "MMR": MaximalMarginalRelevance(diversity=0.3)
            }
        except Exception:
            representation_model = None

    topic_model = BERTopic(
        embedding_model=embedding_model,         # can be None
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        calculate_probabilities=False,
        verbose=True,
        top_n_words=10
    )

    return topic_model


# ----------------------------
# Training / Extraction
# ----------------------------

def extract_topics(
    topic_model: BERTopic,
    texts: List[str],
    embeddings: Optional[np.ndarray] = None,
    config: Optional[TopicModelConfig] = None
) -> Tuple[List[int], Any, BERTopic, Dict[str, Any]]:
    """
    Fit the topic model and return:
    - topics: List[int] topic id per text (aligned to input texts)
    - topic_info: DataFrame from topic_model.get_topic_info()
    - trained topic_model
    - debug: dict with stats (outliers, topic counts)

    Notes:
    - If preprocessing filters texts, we map topics back to original length.
    - embeddings must match original texts length if provided.
    """
    if config is None:
        config = TopicModelConfig()

    if embeddings is not None and len(texts) != len(embeddings):
        raise ValueError(
            f"Length mismatch: texts={len(texts)} vs embeddings={len(embeddings)}. "
            "Make sure embeddings align with texts in the same order."
        )

    # Clean and filter texts
    cleaned_texts, kept_idx = preprocess_texts(
        texts,
        min_len=config.min_text_len,
        drop_empty=config.drop_empty,
    )

    if not cleaned_texts:
        # Nothing to model
        empty_topics = [-1] * len(texts)
        debug = {"error": "No texts left after preprocessing", "n_texts": len(texts)}
        return empty_topics, None, topic_model, debug

    # Filter embeddings to match kept texts
    kept_embeddings = None
    if embeddings is not None:
        kept_embeddings = embeddings[kept_idx]

    # Fit model
    topics, _ = topic_model.fit_transform(cleaned_texts, kept_embeddings)

    # Reduce outliers to improve topic coverage
    if config.reduce_outliers:
        try:
            topics = topic_model.reduce_outliers(
                cleaned_texts,
                topics,
                strategy=config.outlier_strategy
            )
        except Exception:
            # If reduce_outliers fails due to version mismatch or missing config,
            # continue without crashing.
            pass

    # Build debug stats
    counts = Counter(topics)
    n = len(topics)
    outliers = counts.get(-1, 0)
    debug = {
        "n_input_texts": len(texts),
        "n_used_texts": len(cleaned_texts),
        "outliers_count": outliers,
        "outliers_ratio": (outliers / n) if n else None,
        "topic_counts_top10": counts.most_common(10),
    }

    topic_info = topic_model.get_topic_info()

    # Map back to original length (so you can store topic per original chunk)
    full_topics = [-1] * len(texts)
    for idx, t in zip(kept_idx, topics):
        full_topics[idx] = int(t)

    return full_topics, topic_info, topic_model, debug


# ----------------------------
# Label helpers for UI / Metadata
# ----------------------------

def topic_label(topic_model: BERTopic, topic_id: int, top_k: int = 3) -> str:
    """
    Convert a topic id into a human-friendly label.
    Uses top words/phrases from BERTopic.
    """
    if topic_id == -1:
        return "Miscellaneous"

    words = topic_model.get_topic(topic_id)
    if not words:
        return f"Topic {topic_id}"

    # words is a list of tuples (term, score)
    top_terms = [term for term, _ in words[:top_k]]
    label = " / ".join(top_terms).strip()
    return label.title() if label else f"Topic {topic_id}"


def build_topic_labels(topic_model: BERTopic, topics: List[int], top_k: int = 3) -> Dict[int, str]:
    """
    Create a dict mapping topic_id -> readable topic_name.
    """
    unique_ids = sorted(set(topics))
    return {tid: topic_label(topic_model, tid, top_k=top_k) for tid in unique_ids}


def attach_topic_metadata(
    metadatas: List[dict],
    topics: List[int],
    topic_labels: Dict[int, str]
) -> List[dict]:
    """
    Attach topic_id and topic_name into each metadata dict.
    Returns updated metadata list.
    """
    if len(metadatas) != len(topics):
        raise ValueError(f"Length mismatch: metadatas={len(metadatas)} vs topics={len(topics)}")

    updated = []
    for meta, tid in zip(metadatas, topics):
        new_meta = dict(meta) if meta else {}
        new_meta["topic_id"] = int(tid)
        new_meta["topic_name"] = topic_labels.get(int(tid), "Miscellaneous")
        updated.append(new_meta)

    return updated