"""Small, reusable text-cleaning helpers for the post-processing pipeline."""

import re
from typing import Any

import pandas as pd


MENTION_PATTERN = re.compile(r"(?<![\w@])@([A-Za-z0-9_]+)")
HASHTAG_PATTERN = re.compile(r"(?<!\w)#([A-Za-z0-9_]+)")


def normalize_text(text: Any) -> str:
    """Convert a scalar text value to a clean string with single spaces.

    Missing pandas/NumPy values, ``None``, and whitespace-only values become
    an empty string. Other scalar values are safely converted to strings.
    """
    if text is None:
        return ""
    try:
        missing = pd.isna(text)
        if isinstance(missing, bool) and missing:
            return ""
        # NumPy boolean values are not instances of bool on every version.
        if getattr(missing, "item", None) and bool(missing.item()):
            return ""
    except (TypeError, ValueError):
        pass

    return re.sub(r"\s+", " ", str(text)).strip()


def extract_mentions(text: Any) -> list[str]:
    """Return mentioned usernames without the leading ``@`` symbol."""
    return MENTION_PATTERN.findall(normalize_text(text))


def extract_hashtags(text: Any) -> list[str]:
    """Return hashtags with their leading ``#`` symbol preserved."""
    return [f"#{tag}" for tag in HASHTAG_PATTERN.findall(normalize_text(text))]


def clean_text(text: Any) -> str:
    """Prepare text for TF-IDF while preserving the original input elsewhere.

    Mentions are removed so usernames do not become keywords. Hashtag words
    are retained without ``#``. Punctuation is removed only from this derived
    column; sentiment analysis uses the original text so its punctuation cues
    (such as exclamation marks) remain available to VADER.
    """
    normalized = normalize_text(text)
    without_mentions = MENTION_PATTERN.sub(" ", normalized)
    without_hash_marks = HASHTAG_PATTERN.sub(lambda match: match.group(1), without_mentions)
    letters_and_words = re.sub(r"[^\w\s']", " ", without_hash_marks, flags=re.UNICODE)
    return re.sub(r"\s+", " ", letters_and_words).strip().lower()


def process_text(text: Any) -> dict[str, Any]:
    """Return cleaned text and extracted social tokens for one post."""
    normalized = normalize_text(text)
    return {
        "cleaned_text": clean_text(normalized),
        "extracted_mentions": extract_mentions(normalized),
        "hashtags": extract_hashtags(normalized),
    }
