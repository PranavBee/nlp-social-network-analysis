"""Tests for descriptive TF-IDF keyword extraction."""

from src.nlp_social_network.topic_analysis import analyze_keywords


def test_calculates_keywords_for_each_document_and_overall_terms():
    result = analyze_keywords(
        ["green garden grows", "green energy helps", "team scores goal"],
        topic_labels=["environment", "environment", "sports"],
    )
    assert len(result["post_keywords"]) == 3
    assert "green" in result["post_keywords"][0]
    assert result["top_terms"]
    assert {item["term"] for item in result["top_terms"]}.issubset(
        {"green", "garden", "grows", "energy", "helps", "team", "scores", "goal"}
    )


def test_topic_summaries_use_the_supplied_topic_labels():
    result = analyze_keywords(
        ["green garden grows", "green energy helps", "team scores goal"],
        topic_labels=["environment", "environment", "sports"],
    )
    assert set(result["topic_summaries"]) == {"environment", "sports"}
    assert "green" in {item["term"] for item in result["topic_summaries"]["environment"]}


def test_empty_documents_and_empty_vocabulary_are_safe():
    result = analyze_keywords(["", None, "   "], topic_labels=["x", "x", "y"])
    assert result == {"post_keywords": [[], [], []], "top_terms": [], "topic_summaries": {}}
    result = analyze_keywords(["!!!", "??"])
    assert result["post_keywords"] == [[], []]
    assert result["top_terms"] == []


def test_single_document_and_label_length_validation():
    result = analyze_keywords(["one document"], topic_labels=["technology"])
    assert result["post_keywords"][0]
    assert "technology" in result["topic_summaries"]
    try:
        analyze_keywords(["one", "two"], topic_labels=["technology"])
    except ValueError as error:
        assert "same length" in str(error)
    else:
        raise AssertionError("Mismatched topic labels should raise ValueError")
