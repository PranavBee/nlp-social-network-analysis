"""Tests for community partitioning and community-level NLP summaries."""

import json

import networkx as nx
import pandas as pd
import pytest

from src.nlp_social_network.community_detection import (
    COMMUNITY_SUMMARY_COLUMNS,
    MEMBERSHIP_COLUMNS,
    build_membership_dataframe,
    create_undirected_projection,
    detect_communities,
    run_community_analysis,
    summarize_community_posts,
)


def directed_test_graph():
    graph = nx.DiGraph()
    for user_id, username, post_count in [
        ("u1", "alice", 2),
        ("u2", "bob", 1),
        ("u3", "cara", 1),
        ("u4", "dave", 1),
        ("u5", "quiet", 1),
    ]:
        graph.add_node(user_id, username=username, post_count=post_count)
    graph.add_edge("u1", "u2", weight=2)
    graph.add_edge("u2", "u1", weight=1)
    graph.add_edge("u2", "u3", weight=1)
    graph.add_edge("u3", "u4", weight=3)
    # u5 is intentionally isolated.
    return graph


def community_posts():
    return pd.DataFrame(
        [
            {
                "post_id": "p1",
                "user_id": "u1",
                "topic": "technology",
                "sentiment_label": "positive",
                "sentiment_compound": 0.6,
                "hashtags": json.dumps(["#Code", "#Code"]),
                "keywords": json.dumps(["coding", "project"]),
            },
            {
                "post_id": "p2",
                "user_id": "u1",
                "topic": "sports",
                "sentiment_label": "negative",
                "sentiment_compound": -0.6,
                "hashtags": json.dumps(["#GameDay"]),
                "keywords": json.dumps(["team", "coding"]),
            },
            {
                "post_id": "p3",
                "user_id": "u2",
                "topic": "technology",
                "sentiment_label": "neutral",
                "sentiment_compound": 0.0,
                "hashtags": json.dumps(["#Code", "#Tools"]),
                "keywords": json.dumps(["coding", "tools"]),
            },
            {
                "post_id": "p4",
                "user_id": "u3",
                "topic": "environment",
                "sentiment_label": "positive",
                "sentiment_compound": 0.8,
                "hashtags": "[]",
                "keywords": json.dumps(["garden", "green"]),
            },
            {
                "post_id": "p5",
                "user_id": "unknown_user",
                "topic": "education",
                "sentiment_label": "negative",
                "sentiment_compound": -0.8,
                "hashtags": json.dumps(["#Unknown"]),
                "keywords": json.dumps(["unknown"]),
            },
        ]
    )


def manual_membership():
    return pd.DataFrame(
        [
            {"user_id": "u1", "community_id": 0},
            {"user_id": "u2", "community_id": 0},
            {"user_id": "u3", "community_id": 1},
            {"user_id": "u4", "community_id": 1},
        ]
    )


def test_undirected_projection_sums_reciprocal_directional_weights():
    projection = create_undirected_projection(directed_test_graph())
    assert not projection.is_directed()
    assert set(projection.nodes) == set(directed_test_graph().nodes)
    assert projection["u1"]["u2"]["weight"] == 3
    assert projection["u2"]["u3"]["weight"] == 1
    assert projection.degree("u5") == 0


def test_membership_contains_all_nodes_and_ids_are_deterministic():
    graph = directed_test_graph()
    first = detect_communities(graph, seed=123)
    second = detect_communities(graph, seed=123)
    membership = build_membership_dataframe(graph, first)

    assert set(first.assignments) == set(graph.nodes)
    assert first.assignments == second.assignments
    assert set(membership["user_id"]) == set(graph.nodes)
    assert membership.loc[membership.user_id == "u5", "community_id"].notna().all()
    assert all(isinstance(value, int) and value >= 0 for value in first.assignments.values())
    assert list(membership.columns) == list(MEMBERSHIP_COLUMNS)
    assert first.modularity is not None
    assert -1 <= first.modularity <= 1


def test_isolated_users_are_retained_in_singleton_communities():
    result = detect_communities(directed_test_graph())
    isolate_community = result.assignments["u5"]
    assert "u5" in result.communities[isolate_community]
    assert result.communities[isolate_community] == ("u5",)


def test_empty_graph_and_single_user_are_handled_without_modularity():
    empty = detect_communities(nx.DiGraph())
    assert empty.assignments == {}
    assert empty.communities == ()
    assert empty.modularity is None

    one_user = nx.DiGraph()
    one_user.add_node("solo", username="solo", post_count=1)
    single = detect_communities(one_user)
    assert single.assignments == {"solo": 0}
    assert single.communities == (("solo",),)
    assert single.modularity is None


def test_summary_counts_topics_sentiment_hashtags_and_keywords_by_joined_user():
    summary, unmatched = summarize_community_posts(community_posts(), manual_membership())
    community_zero = summary.set_index("community_id").loc[0]
    community_one = summary.set_index("community_id").loc[1]

    assert list(summary.columns) == list(COMMUNITY_SUMMARY_COLUMNS)
    assert community_zero["num_users"] == 2
    assert community_zero["num_posts"] == 3
    assert json.loads(community_zero["topic_distribution"]) == {"sports": 1, "technology": 2}
    assert community_zero["positive_post_count"] == 1
    assert community_zero["negative_post_count"] == 1
    assert community_zero["neutral_post_count"] == 1
    assert community_zero["positive_sentiment_proportion"] == pytest.approx(1 / 3)
    assert community_zero["negative_sentiment_proportion"] == pytest.approx(1 / 3)
    assert community_zero["neutral_sentiment_proportion"] == pytest.approx(1 / 3)
    assert community_zero["mean_compound_sentiment"] == pytest.approx(0.0)
    assert json.loads(community_zero["top_hashtags"])[0] == {"hashtag": "#Code", "post_count": 2}
    assert json.loads(community_zero["top_keywords"])[0] == {"term": "coding", "post_count": 3}

    assert community_one["num_users"] == 2
    assert community_one["num_posts"] == 1
    assert json.loads(community_one["topic_distribution"]) == {"environment": 1}
    assert community_one["positive_post_count"] == 1
    assert len(unmatched) == 1
    assert unmatched.iloc[0]["user_id"] == "unknown_user"


def test_membership_reuses_centrality_results_without_recalculating():
    graph = directed_test_graph()
    detection = detect_communities(graph)
    centrality = pd.DataFrame(
        [
            {"user_id": "u1", "degree_centrality": 0.5, "betweenness_centrality": 0.2, "pagerank": 0.3},
            {"user_id": "u2", "degree_centrality": 0.7, "betweenness_centrality": 0.1, "pagerank": 0.4},
        ]
    )
    membership = build_membership_dataframe(graph, detection, centrality)
    alice = membership.set_index("user_id").loc["u1"]
    quiet = membership.set_index("user_id").loc["u5"]
    assert alice["degree_centrality"] == pytest.approx(0.5)
    assert alice["betweenness_centrality"] == pytest.approx(0.2)
    assert alice["pagerank"] == pytest.approx(0.3)
    assert pd.isna(quiet["pagerank"])


def test_missing_columns_and_invalid_inputs_raise_clear_errors():
    with pytest.raises(TypeError, match="requires the directed"):
        create_undirected_projection(nx.Graph())
    with pytest.raises(ValueError, match="missing required columns"):
        summarize_community_posts(pd.DataFrame({"user_id": ["u1"]}), manual_membership())
    with pytest.raises(ValueError, match="positive and finite"):
        graph = nx.DiGraph()
        graph.add_edge("u1", "u2", weight=0)
        create_undirected_projection(graph)


def test_end_to_end_csv_html_and_chart_outputs_use_temporary_files(tmp_path):
    posts = community_posts().drop(index=4).copy()
    posts["post_id"] = ["p1", "p2", "p3", "p4"]
    posts["username"] = ["alice", "alice", "bob", "cara"]
    posts["extracted_mentions"] = [json.dumps(["bob"]), "[]", "[]", "[]"]
    posts["reply_to_user"] = ["", "", "", ""]
    input_path = tmp_path / "processed_posts.csv"
    posts.to_csv(input_path, index=False)
    centrality_path = tmp_path / "centrality.csv"
    pd.DataFrame(
        [
            {"user_id": user_id, "degree_centrality": 0.1, "betweenness_centrality": 0.0, "pagerank": 0.25}
            for user_id in ["u1", "u2", "u3"]
        ]
    ).to_csv(centrality_path, index=False)

    outcome = run_community_analysis(
        input_path,
        centrality_path=centrality_path,
        membership_path=tmp_path / "membership.csv",
        summary_path=tmp_path / "summary.csv",
        metadata_path=tmp_path / "metadata.json",
        html_path=tmp_path / "community.html",
        unmatched_path=tmp_path / "unmatched.csv",
        figures_dir=tmp_path / "figures",
    )

    assert len(outcome["membership"]) == 3
    assert outcome["metadata"]["num_matched_posts"] == 4
    assert outcome["metadata"]["num_unmatched_posts"] == 0
    assert set(pd.read_csv(tmp_path / "membership.csv").columns) == set(MEMBERSHIP_COLUMNS)
    saved_summary = pd.read_csv(tmp_path / "summary.csv")
    assert set(saved_summary.columns) == set(COMMUNITY_SUMMARY_COLUMNS)
    assert saved_summary["num_posts"].sum() == 4
    assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))["modularity"] is not None
    assert (tmp_path / "community.html").exists()
    assert (tmp_path / "figures" / "community_sizes.html").exists()

