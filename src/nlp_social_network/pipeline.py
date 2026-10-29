"""Run preprocessing, VADER sentiment, and descriptive TF-IDF analysis."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .sentiment_analyzer import analyze_sentiments
from .text_processor import process_text
from .topic_analysis import analyze_keywords


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "sample_posts.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "processed_posts.csv"
REQUIRED_COLUMNS = (
    "post_id",
    "user_id",
    "username",
    "text",
    "timestamp",
    "mentions",
    "reply_to_user",
    "topic",
)


def run_pipeline(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    analyzer: Any | None = None,
) -> pd.DataFrame:
    """Load, enrich, and save posts while retaining all original columns.

    ``analyzer`` is an optional VADER-compatible analyzer, primarily useful
    for tests. Normal command-line use loads the real NLTK VADER analyzer.
    """
    source_path = Path(input_path)
    destination_path = Path(output_path)
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    if not destination_path.is_absolute():
        destination_path = PROJECT_ROOT / destination_path

    posts = pd.read_csv(source_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in posts.columns]
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing_columns)}")

    processed_text = [process_text(text) for text in posts["text"]]
    cleaned_texts = [item["cleaned_text"] for item in processed_text]
    sentiment_rows = analyze_sentiments(posts["text"], analyzer=analyzer)
    keyword_results = analyze_keywords(cleaned_texts, topic_labels=posts["topic"])

    enriched = posts.copy()
    enriched["cleaned_text"] = cleaned_texts
    enriched["extracted_mentions"] = [json.dumps(item["extracted_mentions"], ensure_ascii=False) for item in processed_text]
    enriched["hashtags"] = [json.dumps(item["hashtags"], ensure_ascii=False) for item in processed_text]
    for column in (
        "sentiment_compound",
        "sentiment_positive",
        "sentiment_negative",
        "sentiment_neutral",
        "sentiment_label",
    ):
        enriched[column] = [row[column] for row in sentiment_rows]
    enriched["keywords"] = [json.dumps(words, ensure_ascii=False) for words in keyword_results["post_keywords"]]

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(destination_path, index=False, encoding="utf-8")

    sentiment_counts = enriched["sentiment_label"].value_counts().reindex(
        ["positive", "negative", "neutral"], fill_value=0
    )
    top_keywords = [item["term"] for item in keyword_results["top_terms"]]
    print(f"Rows processed: {len(enriched)}")
    print("Sentiment distribution: " + ", ".join(f"{label}={count}" for label, count in sentiment_counts.items()))
    print("Top keywords: " + (", ".join(top_keywords) if top_keywords else "(none)"))
    print(f"Saved enriched dataset: {destination_path}")
    return enriched


def main() -> None:
    """Run the project pipeline using paths rooted at the project folder."""
    run_pipeline()


if __name__ == "__main__":
    main()
