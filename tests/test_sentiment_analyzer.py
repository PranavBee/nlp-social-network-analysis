"""Tests for VADER score mapping and sentiment labels."""

import pytest

from src.nlp_social_network.sentiment_analyzer import (
    analyze_sentiment,
    label_sentiment,
)


class StubAnalyzer:
    """A small VADER-compatible test double for predictable unit tests."""

    def polarity_scores(self, text):
        if "love" in text.lower():
            return {"compound": 0.8, "pos": 0.8, "neg": 0.0, "neu": 0.2}
        if "awful" in text.lower():
            return {"compound": -0.8, "pos": 0.0, "neg": 0.8, "neu": 0.2}
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}


def test_sentiment_threshold_boundaries():
    assert label_sentiment(0.05) == "positive"
    assert label_sentiment(-0.05) == "negative"
    assert label_sentiment(0.0499) == "neutral"
    assert label_sentiment(-0.0499) == "neutral"


def test_scores_and_label_are_returned_from_vader_compatible_analyzer():
    result = analyze_sentiment("I love this!", analyzer=StubAnalyzer())
    assert result == {
        "sentiment_compound": 0.8,
        "sentiment_positive": 0.8,
        "sentiment_negative": 0.0,
        "sentiment_neutral": 0.2,
        "sentiment_label": "positive",
    }


def test_empty_text_is_scored_without_error():
    result = analyze_sentiment("   ", analyzer=StubAnalyzer())
    assert result["sentiment_label"] == "neutral"
    assert result["sentiment_neutral"] == 1.0


def test_real_vader_scores_text_when_lexicon_is_available():
    # get_vader_analyzer handles a missing lexicon by attempting the official
    # NLTK download and raises a clear RuntimeError if setup cannot complete.
    from src.nlp_social_network.sentiment_analyzer import get_vader_analyzer

    result = analyze_sentiment("I absolutely love this wonderful day!", analyzer=get_vader_analyzer())
    assert result["sentiment_compound"] > 0
    assert result["sentiment_label"] == "positive"


def test_unavailable_vader_lexicon_raises_actionable_error(monkeypatch):
    import src.nlp_social_network.sentiment_analyzer as sentiment_module

    def missing_lexicon():
        raise LookupError("test VADER lexicon is unavailable")

    monkeypatch.setattr(sentiment_module, "SentimentIntensityAnalyzer", missing_lexicon)
    monkeypatch.setattr(sentiment_module.nltk, "download", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError, match="VADER lexicon is unavailable"):
        sentiment_module.get_vader_analyzer()
