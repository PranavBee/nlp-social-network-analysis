"""Non-browser tests for dashboard data validation and network rendering."""

import json

import networkx as nx
import pandas as pd
import pytest

from app import (
    CENTRALITY_COLUMNS,
    COMMUNITY_COLUMNS,
    MEMBERSHIP_COLUMNS,
    POST_COLUMNS,
    DashboardDataError,
    build_network_figure,
    load_dashboard_data,
    parse_json_list,
    token_frequencies,
    validate_columns,
)


def write_dashboard_outputs(folder):
    posts = pd.DataFrame(
        [
            {
                "post_id": "p1",
                "user_id": "u1",
                "username": "alpha",
                "text": "Hello #Tag",
                "topic": "technology",
                "cleaned_text": "hello tag",
                "extracted_mentions": '["beta"]',
                "hashtags": '["#Tag"]',
                "keywords": '["hello"]',
                "sentiment_compound": 0.2,
                "sentiment_label": "positive",
                "reply_to_user": "",
            }
        ]
    )
    centrality = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "username": "alpha",
                "degree_centrality": 0.5,
                "closeness_centrality": 0.0,
                "betweenness_centrality": 0.0,
                "eigenvector_centrality": 1.0,
                "pagerank": 1.0,
                "post_count": 1,
                "degree_rank": 1,
                "closeness_rank": 1,
                "betweenness_rank": 1,
                "eigenvector_rank": 1,
                "pagerank_rank": 1,
            }
        ]
    )
    membership = pd.DataFrame(
        [{"user_id": "u1", "username": "alpha", "community_id": 0, "post_count": 1,
          "degree_centrality": 0.5, "betweenness_centrality": 0.0, "pagerank": 1.0}]
    )
    community_summary = pd.DataFrame(
        [{"community_id": 0, "num_users": 1, "num_posts": 1, "topic_distribution": '{"technology": 1}',
          "positive_post_count": 1, "negative_post_count": 0, "neutral_post_count": 0,
          "positive_sentiment_proportion": 1.0, "negative_sentiment_proportion": 0.0,
          "neutral_sentiment_proportion": 0.0, "mean_compound_sentiment": 0.2,
          "top_hashtags": '[{"hashtag":"#Tag","post_count":1}]',
          "top_keywords": '[{"term":"hello","post_count":1}]'}]
    )
    posts.to_csv(folder / "processed_posts.csv", index=False)
    centrality.to_csv(folder / "centrality_results.csv", index=False)
    membership.to_csv(folder / "community_membership.csv", index=False)
    community_summary.to_csv(folder / "community_summary.csv", index=False)
    (folder / "graph_summary.json").write_text(json.dumps({"num_edges": 0}), encoding="utf-8")
    (folder / "community_detection_summary.json").write_text(json.dumps({"modularity": None}), encoding="utf-8")


def test_file_and_column_validation_has_actionable_messages(tmp_path):
    with pytest.raises(DashboardDataError, match="Run the Phase 2 NLP pipeline"):
        load_dashboard_data(tmp_path)

    with pytest.raises(DashboardDataError, match="missing required columns: text"):
        validate_columns(pd.DataFrame({"post_id": ["p1"]}), ("post_id", "text"), "processed_posts.csv")


def test_loads_existing_artifacts_and_validates_all_required_schemas(tmp_path):
    write_dashboard_outputs(tmp_path)
    data = load_dashboard_data(tmp_path)
    assert list(data["posts"].columns) == list(POST_COLUMNS)
    assert list(data["centrality"].columns) == list(CENTRALITY_COLUMNS)
    assert list(data["membership"].columns) == list(MEMBERSHIP_COLUMNS)
    assert list(data["community_summary"].columns) == list(COMMUNITY_COLUMNS)
    assert data["graph_summary"]["num_edges"] == 0


def test_load_fails_for_missing_schema_and_user_mismatches(tmp_path):
    write_dashboard_outputs(tmp_path)
    pd.DataFrame({"post_id": ["p1"]}).to_csv(tmp_path / "processed_posts.csv", index=False)
    with pytest.raises(DashboardDataError, match="processed_posts.csv.*missing required columns"):
        load_dashboard_data(tmp_path)

    write_dashboard_outputs(tmp_path)
    pd.DataFrame([{"user_id": "different"}]).to_csv(tmp_path / "community_membership.csv", index=False)
    with pytest.raises(DashboardDataError, match="missing required columns"):
        load_dashboard_data(tmp_path)


def test_json_token_helpers_count_each_item_once_per_post():
    assert parse_json_list('["#Tag", "#Tag"]') == ["#Tag", "#Tag"]
    frequencies = token_frequencies(pd.DataFrame({"hashtags": ['["#Tag", "#Tag"]', '["#Tag", "#Other"]']}), "hashtags")
    assert frequencies.iloc[0].to_dict() == {"term": "#Tag", "post_count": 2}


def test_network_figure_keeps_isolate_and_uses_community_colors_and_size_metric():
    graph = nx.DiGraph()
    graph.add_node("u1", username="alpha", post_count=2, interactions_sent=1, interactions_received=0)
    graph.add_node("u2", username="beta", post_count=1, interactions_sent=0, interactions_received=1)
    graph.add_node("u3", username="quiet", post_count=1, interactions_sent=0, interactions_received=0)
    graph.add_edge("u1", "u2", weight=1, mention_count=1, reply_count=0)
    centrality = pd.DataFrame(
        [
            {"user_id": "u1", "degree_centrality": 0.5, "betweenness_centrality": 0.2,
             "eigenvector_centrality": 0.8, "pagerank": 0.6},
            {"user_id": "u2", "degree_centrality": 0.5, "betweenness_centrality": 0.0,
             "eigenvector_centrality": 0.6, "pagerank": 0.3},
            {"user_id": "u3", "degree_centrality": 0.0, "betweenness_centrality": 0.0,
             "eigenvector_centrality": 0.0, "pagerank": 0.1},
        ]
    )
    membership = pd.DataFrame(
        [
            {"user_id": "u1", "community_id": 0},
            {"user_id": "u2", "community_id": 0},
            {"user_id": "u3", "community_id": 1},
        ]
    )
    positions = {"u1": (0.0, 0.0), "u2": (1.0, 0.0), "u3": (0.0, 1.0)}
    figure = build_network_figure(graph, centrality, membership, "pagerank", positions)

    assert len(figure.layout.annotations) == 1
    community_traces = [trace for trace in figure.data if trace.name and trace.name.startswith("Community")]
    assert len(community_traces) == 2
    all_customdata = [item for trace in community_traces for item in trace.customdata]
    assert {item[1] for item in all_customdata} == {"u1", "u2", "u3"}
    quiet = next(item for item in all_customdata if item[1] == "u3")
    assert quiet[0] == "quiet"

