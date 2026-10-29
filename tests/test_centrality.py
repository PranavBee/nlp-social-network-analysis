"""Tests for directed centrality definitions, rankings, and output files."""

import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import src.nlp_social_network.centrality as centrality_module
from src.nlp_social_network.centrality import (
    CentralityCalculationError,
    EIGENVECTOR_TOLERANCE,
    METRIC_COLUMNS,
    calculate_centrality,
    save_centrality_outputs,
)


def known_graph():
    """Directed graph with a bridge, a separate component, and an isolate."""
    graph = nx.DiGraph()
    users = {
        "a": ("alpha", 4),
        "b": ("bravo", 3),
        "m": ("middle", 2),
        "c": ("charlie", 2),
        "d": ("delta", 1),
        "x": ("xray", 1),
        "y": ("yankee", 1),
        "z": ("zulu", 1),
    }
    for user_id, (username, post_count) in users.items():
        graph.add_node(user_id, username=username, post_count=post_count)
    graph.add_edge("a", "b", weight=2)
    graph.add_edge("a", "m", weight=3)
    graph.add_edge("b", "m", weight=1)
    graph.add_edge("m", "c", weight=2)
    graph.add_edge("m", "d", weight=1)
    graph.add_edge("c", "d", weight=1)
    graph.add_edge("x", "y", weight=1)
    return graph


def test_degree_and_directional_degree_centrality_match_networkx():
    graph = nx.DiGraph()
    graph.add_nodes_from(["a", "b", "c", "isolated"])
    graph.add_edges_from([("a", "b"), ("b", "c")])
    results = calculate_centrality(graph).set_index("user_id")

    assert results.loc["b", "degree_centrality"] == pytest.approx(nx.degree_centrality(graph)["b"])
    assert results.loc["b", "degree_centrality"] == pytest.approx(2 / 3)
    assert results.loc["b", "in_degree_centrality"] == pytest.approx(1 / 3)
    assert results.loc["b", "out_degree_centrality"] == pytest.approx(1 / 3)
    assert results.loc["a", "in_degree_centrality"] == 0
    assert results.loc["a", "out_degree_centrality"] == pytest.approx(1 / 3)
    assert results.loc["isolated", "degree_centrality"] == 0


def test_closeness_uses_incoming_directed_distances():
    graph = nx.DiGraph()
    graph.add_nodes_from(["a", "b", "c", "isolated"])
    graph.add_edges_from([("a", "b"), ("b", "c")])
    results = calculate_centrality(graph).set_index("user_id")

    expected = nx.closeness_centrality(graph, distance=None, wf_improved=True)
    assert results.loc["b", "closeness_centrality"] == pytest.approx(1 / 3)
    assert results.loc["c", "closeness_centrality"] == pytest.approx(4 / 9)
    assert results.loc["c", "closeness_centrality"] == pytest.approx(expected["c"])
    assert results.loc["a", "closeness_centrality"] == 0
    assert results.loc["isolated", "closeness_centrality"] == 0


def test_directed_betweenness_identifies_bridge_node():
    graph = known_graph()
    results = calculate_centrality(graph).set_index("user_id")
    expected = nx.betweenness_centrality(graph, normalized=True, weight=None)

    assert results.loc["m", "betweenness_centrality"] == pytest.approx(expected["m"])
    assert results.loc["m", "betweenness_centrality"] == pytest.approx(4 / 42)
    assert results.loc["m", "betweenness_centrality"] > results.loc["a", "betweenness_centrality"]
    assert results.loc["z", "betweenness_centrality"] == 0


def test_eigenvector_centrality_uses_incoming_neighbors():
    graph = nx.DiGraph()
    graph.add_nodes_from(["author", "target", "isolated"])
    graph.add_edge("author", "target", weight=2)
    results = calculate_centrality(graph).set_index("user_id")
    expected = nx.eigenvector_centrality(
        graph, weight="weight", max_iter=5000, tol=EIGENVECTOR_TOLERANCE
    )

    assert results.loc["target", "eigenvector_centrality"] == pytest.approx(expected["target"])
    assert results.loc["target", "eigenvector_centrality"] > results.loc["author", "eigenvector_centrality"]


def test_pagerank_matches_weighted_directed_networkx_result():
    graph = known_graph()
    results = calculate_centrality(graph).set_index("user_id")
    expected = nx.pagerank(graph, alpha=0.85, weight="weight", tol=1e-10, max_iter=1000)

    for user_id, score in expected.items():
        assert results.loc[user_id, "pagerank"] == pytest.approx(score)
    assert results["pagerank"].sum() == pytest.approx(1.0)


def test_isolates_and_disconnected_components_remain_in_results():
    graph = known_graph()
    results = calculate_centrality(graph).set_index("user_id")
    assert set(results.index) == set(graph.nodes)
    assert results.loc["z", "degree_centrality"] == 0
    assert results.loc["z", "pagerank"] > 0
    assert np.isfinite(results[list(METRIC_COLUMNS)].to_numpy(dtype=float)).all()


def test_ranks_are_deterministic_and_ties_share_best_rank():
    graph = nx.DiGraph()
    graph.add_node("b", username="bravo", post_count=1)
    graph.add_node("a", username="alpha", post_count=1)
    first = calculate_centrality(graph)
    second = calculate_centrality(graph)

    pd.testing.assert_frame_equal(first, second)
    assert list(first["user_id"]) == ["a", "b"]
    assert list(first["degree_rank"]) == [1, 1]
    assert list(first["pagerank_rank"]) == [1, 1]


def test_results_schema_and_csv_json_and_chart_exports(tmp_path):
    results = calculate_centrality(known_graph())
    csv_path, summary_path, chart_paths = save_centrality_outputs(
        results,
        results_path=tmp_path / "centrality.csv",
        summary_path=tmp_path / "summary.json",
        figures_dir=tmp_path / "figures",
    )

    assert list(results.columns) == [
        "user_id",
        "username",
        *METRIC_COLUMNS,
        "post_count",
        "degree_rank",
        "in_degree_rank",
        "out_degree_rank",
        "closeness_rank",
        "betweenness_rank",
        "eigenvector_rank",
        "pagerank_rank",
    ]
    assert len(pd.read_csv(csv_path)) == len(known_graph())
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["num_users"] == len(known_graph())
    assert set(summary["metrics"]) == set(METRIC_COLUMNS)
    assert len(summary["metrics"]["degree_centrality"]["top_users"]) == len(known_graph())
    assert len(chart_paths) == 4
    assert all(path.exists() and path.stat().st_size > 0 for path in chart_paths)


def test_rejects_undirected_input_without_implicit_conversion():
    with pytest.raises(TypeError, match="requires the directed graph"):
        calculate_centrality(nx.Graph())


def test_eigenvector_nonconvergence_raises_clear_error(monkeypatch):
    def always_fails(*args, **kwargs):
        raise nx.PowerIterationFailedConvergence(kwargs.get("max_iter", 0))

    monkeypatch.setattr(centrality_module.nx, "eigenvector_centrality", always_fails)
    with pytest.raises(CentralityCalculationError, match="did not converge after 5,000 iterations"):
        calculate_centrality(known_graph())
