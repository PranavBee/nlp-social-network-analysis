"""Integration-level tests for the NLP pipeline using tiny temporary CSVs."""

import json

import pandas as pd
import pytest

from src.nlp_social_network.pipeline import REQUIRED_COLUMNS, run_pipeline


class StubAnalyzer:
    def polarity_scores(self, text):
        if "great" in text.lower():
            return {"compound": 0.7, "pos": 0.7, "neg": 0.0, "neu": 0.3}
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}


def sample_posts():
    return pd.DataFrame(
        [
            {
                "post_id": "post_test_1",
                "user_id": "user_test_1",
                "username": "maplebyte",
                "text": "This is great! @bluejaystudio #GoodNews",
                "timestamp": "2025-01-01T10:00:00Z",
                "mentions": '["bluejaystudio"]',
                "reply_to_user": "bluejaystudio",
                "topic": "technology",
            },
            {
                "post_id": "post_test_2",
                "user_id": "user_test_2",
                "username": "bluejaystudio",
                "text": "A quiet update about learning",
                "timestamp": "2025-01-01T10:23:00Z",
                "mentions": "[]",
                "reply_to_user": "",
                "topic": "education",
            },
        ]
    )


def test_pipeline_preserves_original_columns_rows_and_adds_expected_fields(tmp_path, capsys):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "processed.csv"
    original = sample_posts()
    original.to_csv(input_path, index=False)

    enriched = run_pipeline(input_path, output_path, analyzer=StubAnalyzer())
    loaded = pd.read_csv(output_path)
    expected_new_columns = {
        "cleaned_text",
        "extracted_mentions",
        "hashtags",
        "sentiment_compound",
        "sentiment_positive",
        "sentiment_negative",
        "sentiment_neutral",
        "sentiment_label",
        "keywords",
    }

    assert output_path.exists()
    assert len(enriched) == len(original) == len(loaded)
    assert set(REQUIRED_COLUMNS).issubset(enriched.columns)
    assert expected_new_columns.issubset(enriched.columns)
    source_as_loaded = pd.read_csv(input_path)
    pd.testing.assert_frame_equal(
        enriched[list(REQUIRED_COLUMNS)], source_as_loaded[list(REQUIRED_COLUMNS)]
    )
    assert json.loads(loaded.loc[0, "extracted_mentions"]) == ["bluejaystudio"]
    assert json.loads(loaded.loc[0, "hashtags"]) == ["#GoodNews"]
    assert "Rows processed: 2" in capsys.readouterr().out


def test_pipeline_reports_missing_required_columns(tmp_path):
    input_path = tmp_path / "missing.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(input_path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        run_pipeline(input_path, tmp_path / "never_written.csv", analyzer=StubAnalyzer())
