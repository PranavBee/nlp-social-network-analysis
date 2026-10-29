"""Descriptive TF-IDF keyword extraction for posts and existing topic labels.

This module ranks terms in the supplied text. It does not discover semantic
topics or perform topic modelling; grouped summaries use the dataset's
existing ``topic`` labels.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .text_processor import normalize_text


def analyze_keywords(
    documents: Iterable[Any],
    topic_labels: Iterable[Any] | None = None,
    *,
    per_document_top_n: int = 5,
    top_overall_n: int = 10,
    per_topic_top_n: int = 10,
) -> dict[str, Any]:
    """Calculate TF-IDF terms for each document and descriptive summaries.

    Returns one keyword list per input document, overall terms ranked by
    average TF-IDF weight, and summaries grouped by the provided topic label.
    Empty documents and empty vocabularies produce empty lists safely.
    """
    texts = [normalize_text(document) for document in documents]
    has_labels = topic_labels is not None
    labels = list(topic_labels) if has_labels else []
    if has_labels and len(labels) != len(texts):
        raise ValueError("topic_labels must have the same length as documents")

    post_keywords: list[list[str]] = [[] for _ in texts]
    overall_terms: list[dict[str, float | str]] = []
    topic_summaries: dict[str, list[dict[str, float | str]]] = {}

    non_empty_indices = [index for index, text in enumerate(texts) if text]
    if not non_empty_indices:
        return {
            "post_keywords": post_keywords,
            "top_terms": overall_terms,
            "topic_summaries": topic_summaries,
        }

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b\w+\b",
        strip_accents="unicode",
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError as error:
        # sklearn raises for a corpus made only of ignorable tokens.
        if "empty vocabulary" not in str(error).lower():
            raise
        return {
            "post_keywords": post_keywords,
            "top_terms": overall_terms,
            "topic_summaries": topic_summaries,
        }

    terms = vectorizer.get_feature_names_out()

    def ranked_terms(weights: np.ndarray, limit: int) -> list[dict[str, float | str]]:
        order = sorted(range(len(terms)), key=lambda index: (-float(weights[index]), str(terms[index])))
        return [
            {"term": str(terms[index]), "score": float(weights[index])}
            for index in order[: max(0, limit)]
            if weights[index] > 0
        ]

    for row_index in range(len(texts)):
        row = matrix.getrow(row_index)
        weights = np.zeros(len(terms), dtype=float)
        weights[row.indices] = row.data
        post_keywords[row_index] = [
            str(item["term"]) for item in ranked_terms(weights, per_document_top_n)
        ]

    mean_weights = np.asarray(matrix.mean(axis=0)).ravel()
    overall_terms = ranked_terms(mean_weights, top_overall_n)

    # Only group rows when topic labels are supplied; preserve the labels as
    # written and skip missing labels rather than inventing a topic.
    if has_labels:
        grouped_indices: dict[str, list[int]] = {}
        for index, label in enumerate(labels):
            if pd.isna(label):
                continue
            key = str(label)
            grouped_indices.setdefault(key, []).append(index)

        for label, indices in grouped_indices.items():
            group_mean = np.asarray(matrix[indices].mean(axis=0)).ravel()
            topic_summaries[label] = ranked_terms(group_mean, per_topic_top_n)

    return {
        "post_keywords": post_keywords,
        "top_terms": overall_terms,
        "topic_summaries": topic_summaries,
    }
