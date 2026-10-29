"""Sentiment scoring with NLTK's VADER sentiment analyzer."""

from collections.abc import Iterable
from typing import Any

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from .text_processor import normalize_text


VADER_RESOURCE = "vader_lexicon"


def get_vader_analyzer() -> SentimentIntensityAnalyzer:
    """Return VADER, downloading its lexicon only if NLTK cannot find it."""
    try:
        return SentimentIntensityAnalyzer()
    except LookupError as initial_error:
        try:
            downloaded = nltk.download(VADER_RESOURCE, quiet=True, raise_on_error=True)
            if not downloaded:
                raise RuntimeError("NLTK reported that the VADER lexicon download failed.")
            return SentimentIntensityAnalyzer()
        except Exception as download_error:
            raise RuntimeError(
                "NLTK VADER lexicon is unavailable. Check your internet connection, "
                "then run `python -m nltk.downloader vader_lexicon` and retry. "
                f"Original lookup error: {initial_error}. Download/setup error: {download_error}"
            ) from download_error


def label_sentiment(compound: float) -> str:
    """Map a VADER compound score to the project's three sentiment labels."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def analyze_sentiment(text: Any, analyzer: SentimentIntensityAnalyzer | None = None) -> dict[str, Any]:
    """Return VADER's four scores and the threshold-based label for one post.

    Empty text is scored by VADER itself (normally neutral); no scores are
    fabricated or substituted when the VADER lexicon cannot be loaded.
    """
    vader = analyzer if analyzer is not None else get_vader_analyzer()
    scores = vader.polarity_scores(normalize_text(text))
    compound = float(scores["compound"])
    return {
        "sentiment_compound": compound,
        "sentiment_positive": float(scores["pos"]),
        "sentiment_negative": float(scores["neg"]),
        "sentiment_neutral": float(scores["neu"]),
        "sentiment_label": label_sentiment(compound),
    }


def analyze_sentiments(
    texts: Iterable[Any], analyzer: SentimentIntensityAnalyzer | None = None
) -> list[dict[str, Any]]:
    """Score many posts while constructing the VADER analyzer only once."""
    vader = analyzer if analyzer is not None else get_vader_analyzer()
    return [analyze_sentiment(text, analyzer=vader) for text in texts]
