"""Tests for directed graph construction and graph summary statistics."""

import json

import networkx as nx
import pandas as pd
import pytest

from src.nlp_social_network.graph_builder import build_graph, summarize_graph


def small_posts():
    """A tiny fixture with known mentions, replies, duplicates, and isolates."""
    return pd.DataFrame(
        [
            {
                "post_id": "p1",
                "user_id": "u1",
                "username": "alice",
                "extracted_mentions": json.dumps(["bob", "bob", "ghost"]),
                "reply_to_user": "bob",
                "sentiment_compound": 0.8,
            },
            {
                "post_id": "p2",
                "user_id": "u1",
                "username": "alice",
                "extracted_mentions": json.dumps(["bob", "cara"]),
                "reply_to_user": "",
                "sentiment_compound": -0.2,
            },
            {
                "post_id": "p3",
                "user_id": "u2",
                "username": "bob",
                "extracted_mentions": "[]",
                "reply_to_user": "cara",
                "sentiment_compound": 0.2,
            },
            {
                "post_id": "p4",
                "user_id": "u3",
                "username": "cara",
                "extracted_mentions": json.dumps(["ghost", "cara"]),
                "reply_to_user": "ghost",
                "sentiment_compound": 0.0,
            },
            {
                "post_id": "p5",
                "user_id": "u4",
                "username": "dave",
                "extracted_mentions": "[]",
                "reply_to_user": "",
                "sentiment_compound": None,
            },
            {
                "post_id": "p6",
                "user_id": "u5",
                "username": "eve",
                "extracted_mentions": "[]",
                "reply_to_user": "",
                "sentiment_compound": None,
            },
            {
                "post_id": "p7",
                "user_id": "u1",
                "username": "alice",
                "extracted_mentions": json.dumps(["bob"]),
                "reply_to_user": "",
                "sentiment_compound": 0.6,
            },
        ]
    )


def test_builds_all_user_nodes_and_aggregates_interaction_types():
    graph = build_graph(small_posts())

    assert isinstance(graph, nx.DiGraph)
    assert set(graph.nodes) == {"u1", "u2", "u3", "u4", "u5"}
    assert graph.nodes["u1"]["username"] == "alice"
    assert graph.nodes["u1"]["post_count"] == 3
    assert graph.nodes["u1"]["mean_sentiment"] == pytest.approx(0.4)
    assert graph.nodes["u4"]["post_count"] == 1
    assert graph.nodes["u4"]["interactions_sent"] == 0
    assert graph.nodes["u4"]["interactions_received"] == 0
    assert "ghost" not in graph

    alice_bob = graph["u1"]["u2"]
    assert alice_bob == {"weight": 3, "mention_count": 2, "reply_count": 1}
    assert graph["u1"]["u3"] == {"weight": 1, "mention_count": 1, "reply_count": 0}
    assert graph["u2"]["u3"] == {"weight": 1, "mention_count": 0, "reply_count": 1}
    assert not graph.has_edge("u3", "u3")
    assert graph.graph["number_posts"] == 7


def test_resolves_user_ids_in_reference_columns():
    posts = small_posts()
    posts.loc[1, "extracted_mentions"] = json.dumps(["u2"])
    posts.loc[1, "reply_to_user"] = ""
    graph = build_graph(posts)
    assert graph["u1"]["u2"]["weight"] == 3


def test_self_loops_can_be_enabled_explicitly():
    graph = build_graph(small_posts(), include_self_loops=True)
    assert graph.has_edge("u3", "u3")
    assert graph["u3"]["u3"]["mention_count"] == 1


def test_graph_summary_counts_edges_interactions_and_components():
    summary = summarize_graph(build_graph(small_posts()))
    assert summary["num_nodes"] == 5
    assert summary["num_edges"] == 3
    assert summary["num_posts"] == 7
    assert summary["num_mention_interactions"] == 3
    assert summary["num_reply_interactions"] == 2
    assert summary["density"] == pytest.approx(0.15)
    assert summary["average_in_degree"] == pytest.approx(0.6)
    assert summary["average_out_degree"] == pytest.approx(0.6)
    assert summary["num_weakly_connected_components"] == 3
    assert summary["num_isolated_nodes"] == 2
    assert summary["degree_counts_distinct_neighbors"] is True
    assert summary["self_loops_ignored"] is True


def test_invalid_missing_required_columns_raise_clear_error():
    with pytest.raises(ValueError, match="missing required columns"):
        build_graph(pd.DataFrame({"username": ["alice"]}))

    missing_mentions = small_posts().drop(columns=["extracted_mentions"])
    with pytest.raises(ValueError, match="extracted_mentions"):
        build_graph(missing_mentions)


def test_malformed_mentions_raise_clear_error():
    posts = small_posts()
    posts.loc[0, "extracted_mentions"] = "not a JSON array"
    with pytest.raises(ValueError, match="must be a JSON array"):
        build_graph(posts)


def test_conflicting_username_and_user_id_mappings_are_rejected():
    posts = small_posts()
    posts.loc[1, "username"] = "different-name"
    with pytest.raises(ValueError, match="maps to multiple usernames"):
        build_graph(posts)
