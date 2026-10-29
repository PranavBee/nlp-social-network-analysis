"""Calculate directed-graph centrality and export rankings and charts.

Graph edges point from the author of a post to the mentioned/replied-to user.
All metrics below use the directed graph from :mod:`graph_builder` without an
undirected projection. The graph's ``weight`` is an interaction frequency,
not a distance, so shortest-path metrics are deliberately unweighted.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .graph_builder import DEFAULT_INPUT_PATH, PROJECT_ROOT, build_graph


DEFAULT_RESULTS_PATH = PROJECT_ROOT / "outputs" / "centrality_results.csv"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "centrality_summary.json"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
EIGENVECTOR_TOLERANCE = 1e-4

METRIC_COLUMNS = (
    "degree_centrality",
    "in_degree_centrality",
    "out_degree_centrality",
    "closeness_centrality",
    "betweenness_centrality",
    "eigenvector_centrality",
    "pagerank",
)

RANK_COLUMNS = {
    "degree_centrality": "degree_rank",
    "in_degree_centrality": "in_degree_rank",
    "out_degree_centrality": "out_degree_rank",
    "closeness_centrality": "closeness_rank",
    "betweenness_centrality": "betweenness_rank",
    "eigenvector_centrality": "eigenvector_rank",
    "pagerank": "pagerank_rank",
}

MAIN_CHART_METRICS = (
    "degree_centrality",
    "closeness_centrality",
    "betweenness_centrality",
    "eigenvector_centrality",
)

METRIC_INTERPRETATIONS = {
    "degree_centrality": (
        "Total distinct incoming plus outgoing neighbors, normalized by n-1. "
        "A user can be both the source and target of an edge with the same neighbor."
    ),
    "in_degree_centrality": (
        "Distinct users who mention or reply to this user, divided by n-1."
    ),
    "out_degree_centrality": (
        "Distinct users this user mentions or replies to, divided by n-1."
    ),
    "closeness_centrality": (
        "Incoming closeness: directed shortest paths from other users to this user. "
        "NetworkX's wf_improved=True adjustment scales by the fraction of users that can reach them."
    ),
    "betweenness_centrality": (
        "Normalized directed shortest paths: the fraction of ordered source-target "
        "shortest paths that pass through this user. Paths are unweighted because edge weight is frequency."
    ),
    "eigenvector_centrality": (
        "Incoming-neighbor convention: a user scores higher when pointed to by users "
        "with high scores. Interaction frequency is used as edge weight. NetworkX power iteration "
        "uses tolerance 1e-4 to handle reducible/disconnected graphs; it retries with up to 5,000 iterations."
    ),
    "pagerank": (
        "Rank flows along directed edges from users who mention/reply to their targets. "
        "Repeated interactions influence transition probability through edge weight; teleportation "
        "keeps disconnected and isolated users in the ranking."
    ),
}


class CentralityCalculationError(RuntimeError):
    """Raised when a centrality algorithm cannot safely converge."""


def _validate_graph(graph: nx.Graph) -> None:
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph")
    if not graph.is_directed():
        raise TypeError("centrality analysis requires the directed graph; undirected input is not converted")


def _with_convergence_retry(
    metric_name: str,
    algorithm: Callable[..., dict[Any, float]],
    graph: nx.DiGraph,
    **kwargs: Any,
) -> dict[Any, float]:
    """Retry a power-iteration metric with more iterations, then fail clearly."""
    last_error: nx.PowerIterationFailedConvergence | None = None
    for max_iter in (1000, 5000):
        try:
            return algorithm(graph, max_iter=max_iter, **kwargs)
        except nx.PowerIterationFailedConvergence as error:
            last_error = error
    raise CentralityCalculationError(
        f"{metric_name} did not converge after 5,000 iterations. "
        "Check graph weights and connectivity before exporting results."
    ) from last_error


def calculate_centrality(graph: nx.DiGraph) -> pd.DataFrame:
    """Return centrality scores and deterministic ranks for each graph node.

    Closeness follows NetworkX's directed default: inward distances to each
    node. Betweenness follows directed shortest paths and ignores frequency
    weights (they are not distances). Eigenvector centrality and PageRank use
    edge ``weight`` as interaction frequency. All users, including isolates,
    remain as rows. Eigenvector power iteration uses tolerance 1e-4 and retries
    with a maximum of 5,000 iterations because reducible graphs can converge
    slowly; an unresolved failure raises :class:`CentralityCalculationError`.
    """
    _validate_graph(graph)
    if graph.number_of_nodes() == 0:
        score_maps: dict[str, dict[Any, float]] = {metric: {} for metric in METRIC_COLUMNS}
    else:
        try:
            eigenvector = _with_convergence_retry(
                "Eigenvector centrality",
                nx.eigenvector_centrality,
                graph,
                weight="weight",
                tol=EIGENVECTOR_TOLERANCE,
            )
        except CentralityCalculationError:
            raise

        pagerank = _with_convergence_retry(
            "PageRank",
            nx.pagerank,
            graph,
            alpha=0.85,
            weight="weight",
            tol=1e-10,
        )
        score_maps = {
            "degree_centrality": nx.degree_centrality(graph),
            "in_degree_centrality": nx.in_degree_centrality(graph),
            "out_degree_centrality": nx.out_degree_centrality(graph),
            # NetworkX directed closeness uses incoming distances by default.
            "closeness_centrality": nx.closeness_centrality(graph, distance=None, wf_improved=True),
            "betweenness_centrality": nx.betweenness_centrality(graph, normalized=True, weight=None),
            "eigenvector_centrality": eigenvector,
            "pagerank": pagerank,
        }

    rows = []
    for user_id, attributes in graph.nodes(data=True):
        row = {
            "user_id": str(user_id),
            "username": str(attributes.get("username", user_id)),
            "post_count": int(attributes.get("post_count", 0)),
        }
        row.update({metric: float(score_maps[metric].get(user_id, 0.0)) for metric in METRIC_COLUMNS})
        rows.append(row)

    columns = ["user_id", "username", *METRIC_COLUMNS, "post_count"]
    results = pd.DataFrame(rows, columns=columns)
    if results.empty:
        for metric in METRIC_COLUMNS:
            results[RANK_COLUMNS[metric]] = pd.Series(dtype="int64")
        return results

    numeric_scores = results[list(METRIC_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(numeric_scores).all():
        raise CentralityCalculationError("Centrality calculation returned a non-finite score")

    # Stable user ID ordering makes output ordering reproducible. Equal scores
    # share the best rank (method='min'); the user ID order resolves display ties.
    results = results.sort_values("user_id", kind="mergesort").reset_index(drop=True)
    for metric in METRIC_COLUMNS:
        results[RANK_COLUMNS[metric]] = results[metric].rank(
            method="min", ascending=False
        ).astype("int64")
    return results


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def _top_metric_rows(results: pd.DataFrame, metric: str, top_n: int) -> pd.DataFrame:
    return results.sort_values([metric, "user_id"], ascending=[False, True], kind="mergesort").head(top_n)


def build_summary(results: pd.DataFrame, *, top_n: int = 10) -> dict[str, Any]:
    """Create a concise JSON-ready metric summary with top users and statistics."""
    metrics: dict[str, Any] = {}
    for metric in METRIC_COLUMNS:
        scores = pd.to_numeric(results[metric], errors="coerce")
        top_users = _top_metric_rows(results, metric, top_n)
        metrics[metric] = {
            "interpretation": METRIC_INTERPRETATIONS[metric],
            "distribution": {
                "count": int(scores.count()),
                "mean": float(scores.mean()) if scores.count() else 0.0,
                "std": float(scores.std(ddof=0)) if scores.count() else 0.0,
                "min": float(scores.min()) if scores.count() else 0.0,
                "median": float(scores.median()) if scores.count() else 0.0,
                "max": float(scores.max()) if scores.count() else 0.0,
            },
            "top_users": [
                {
                    "rank": int(row[RANK_COLUMNS[metric]]),
                    "user_id": str(row["user_id"]),
                    "username": str(row["username"]),
                    "score": float(row[metric]),
                }
                for _, row in top_users.iterrows()
            ],
        }
    return {"num_users": int(len(results)), "metrics": metrics}


def create_top_user_charts(results: pd.DataFrame, figures_dir: str | Path = DEFAULT_FIGURES_DIR) -> list[Path]:
    """Save one self-contained Plotly HTML bar chart per required metric."""
    destination = _resolve_path(figures_dir)
    destination.mkdir(parents=True, exist_ok=True)
    chart_paths: list[Path] = []
    for metric in MAIN_CHART_METRICS:
        top_users = _top_metric_rows(results, metric, 10).sort_values(
            [metric, "user_id"], ascending=[True, False], kind="mergesort"
        )
        figure = go.Figure(
            data=[
                go.Bar(
                    x=top_users[metric],
                    y=top_users["username"],
                    orientation="h",
                    customdata=top_users[["user_id", RANK_COLUMNS[metric]]],
                    hovertemplate=(
                        "Username: %{y}<br>Score: %{x:.6f}<br>"
                        "User ID: %{customdata[0]}<br>Rank: %{customdata[1]}<extra></extra>"
                    ),
                )
            ]
        )
        figure.update_layout(
            title=f"Top 10 users by {metric.replace('_', ' ').title()}",
            xaxis_title=metric.replace("_", " ").title(),
            yaxis_title="Username",
            template="plotly_white",
            margin={"l": 130, "r": 24, "t": 60, "b": 48},
        )
        path = destination / f"{metric}_top10.html"
        figure.write_html(path, include_plotlyjs="inline", full_html=True)
        chart_paths.append(path)
    return chart_paths


def save_centrality_outputs(
    results: pd.DataFrame,
    *,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> tuple[Path, Path, list[Path]]:
    """Write the rankings CSV, JSON summary, and four main-metric charts."""
    csv_path = _resolve_path(results_path)
    json_path = _resolve_path(summary_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(json.dumps(build_summary(results), indent=2), encoding="utf-8")
    chart_paths = create_top_user_charts(results, figures_dir)
    return csv_path, json_path, chart_paths


def run_centrality_analysis(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    *,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> pd.DataFrame:
    """Reuse the graph builder, calculate every user's scores, and save output."""
    source = _resolve_path(input_path)
    posts = pd.read_csv(source)
    graph = build_graph(posts)
    results = calculate_centrality(graph)
    csv_path, json_path, chart_paths = save_centrality_outputs(
        results,
        results_path=results_path,
        summary_path=summary_path,
        figures_dir=figures_dir,
    )

    print(f"Users ranked: {len(results)}")
    for metric in MAIN_CHART_METRICS:
        top = _top_metric_rows(results, metric, 3)
        leaders = ", ".join(f"{row.username} ({row[metric]:.4f})" for _, row in top.iterrows())
        print(f"Top {metric.replace('_', ' ')}: {leaders}")
    print(f"Results CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")
    print(f"Charts: {len(chart_paths)} saved under {_resolve_path(figures_dir)}")
    print("These rankings describe only the project's fictional synthetic dataset.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate directed social network centrality metrics.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Processed post CSV")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH, help="Centrality CSV output")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH, help="JSON summary output")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR, help="Top-user chart directory")
    args = parser.parse_args()
    run_centrality_analysis(
        args.input,
        results_path=args.results,
        summary_path=args.summary,
        figures_dir=args.figures_dir,
    )


if __name__ == "__main__":
    main()
