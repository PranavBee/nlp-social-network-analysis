"""Unit tests for cleaning post text and extracting social tokens."""

import numpy as np
import pandas as pd

from src.nlp_social_network.text_processor import (
    clean_text,
    extract_hashtags,
    extract_mentions,
    normalize_text,
    process_text,
)


def test_normalize_text_collapses_whitespace_and_preserves_punctuation():
    assert normalize_text("  Wow!!!\n this   is great. ") == "Wow!!! this is great."


def test_missing_and_empty_text_are_safe():
    for value in (None, np.nan, pd.NA, "", "   \t"):
        assert normalize_text(value) == ""
        assert clean_text(value) == ""
        assert extract_mentions(value) == []
        assert extract_hashtags(value) == []


def test_extracts_multiple_mentions_and_hashtags():
    text = "Hello @maplebyte and @bluejaystudio! #GoodNews #Learning_Together"
    assert extract_mentions(text) == ["maplebyte", "bluejaystudio"]
    assert extract_hashtags(text) == ["#GoodNews", "#Learning_Together"]


def test_clean_text_removes_mentions_but_keeps_hashtag_words():
    assert clean_text("Love this! @maplebyte #GreenNeighbors") == "love this greenneighbors"


def test_process_text_returns_derived_fields_without_mutating_input():
    original = "A wonderful day! @maplebyte #GameDay"
    result = process_text(original)
    assert result["cleaned_text"] == "a wonderful day gameday"
    assert result["extracted_mentions"] == ["maplebyte"]
    assert result["hashtags"] == ["#GameDay"]
    assert original == "A wonderful day! @maplebyte #GameDay"
