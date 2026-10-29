"""Detect user communities and summarize their posts and NLP attributes.

Community detection uses a weighted undirected projection of the existing
directed interaction graph. For each unordered pair of users, the projection
edge weight is the sum of the interaction weights in both directions. This is
appropriate for modularity-based algorithms, which optimize undirected
within-group connectivity. The original directed graph is retained for the
interactive visualization. Isolated users remain as singleton communities.

Community IDs are deterministic labels assigned after sorting each partition;
they are arbitrary identifiers, not community rankings.
"""

import argparse
import html
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from pyvis.network import Network

from .centrality import DEFAULT_RESULTS_PATH
from .graph_builder import DEFAULT_INPUT_PATH, PROJECT_ROOT, build_graph


DEFAULT_MEMBERSHIP_PATH = PROJECT_ROOT / "outputs" / "community_membership.csv"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "community_summary.csv"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "outputs" / "community_detection_summary.json"
DEFAULT_HTML_PATH = PROJECT_ROOT / "outputs" / "community_network.html"
DEFAULT_UNMATCHED_PATH = PROJECT_ROOT / "outputs" / "unmatched_community_posts.csv"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
DEFAULT_CENTRALITY_PATH = DEFAULT_RESULTS_PATH

SUMMARY_REQUIRED_COLUMNS = ("user_id", "topic", "sentiment_label", "sentiment_compound")
MEMBERSHIP_COLUMNS = (
    "user_id",
    "username",
    "community_id",
    "post_count",
    "degree_centrality",
    "betweenness_centrality",
    "pagerank",
)
COMMUNITY_SUMMARY_COLUMNS = (
    "community_id",
    "num_users",
    "num_posts",
    "topic_distribution",
    "positive_post_count",
    "negative_post_count",
    "neutral_post_count",
    "sentiment_labeled_post_count",
    "positive_sentiment_proportion",
    "negative_sentiment_proportion",
    "neutral_sentiment_proportion",
    "mean_compound_sentiment",
    "top_hashtags",
    "top_keywords",
)

CENTRALITY_COLUMNS = ("degree_centrality", "betweenness_centrality", "pagerank")
SENTIMENT_LABELS = ("positive", "negative", "neutral")


@dataclass(frozen=True)
class CommunityDetectionResult:
    """A canonicalized community partition and its modularity metadata."""

    assignments: dict[str, int]
    communities: tuple[tuple[str, ...], ...]
    modularity: float | None
    algorithm: str


def create_undirected_projection(directed_graph: nx.DiGraph) -> nx.Graph:
    """Project a directed interaction graph to an undirected weighted graph.

    Reciprocal directed edges are combined by summing their weights. Missing
    edge weights default to one interaction. Invalid, non-finite, or
    non-positive weights raise a clear error rather than being dropped.
    """
    if not isinstance(directed_graph, nx.Graph) or not directed_graph.is_directed():
        raise TypeError("community detection requires the directed NetworkX graph")

    projection = nx.Graph()
    for user_id, attributes in directed_graph.nodes(data=True):
        projection.add_node(user_id, **attributes)

    for source, target, attributes in directed_graph.edges(data=True):
        try:
            weight = float(attributes.get("weight", 1))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid interaction weight on edge {source!r} -> {target!r}") from error
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Interaction weight on edge {source!r} -> {target!r} must be positive and finite")

        if projection.has_edge(source, target):
            projection[source][target]["weight"] += weight
        else:
            projection.add_edge(source, target, weight=weight)

    projection.graph["projection"] = (
        "Undirected modularity projection; each unordered edge weight is the sum "
        "of directed interaction weights in both directions."
    )
    return projection


def _canonicalize_communities(
    communities: list[set[Any]] | list[frozenset[Any]],
    nodes: list[Any],
) -> tuple[tuple[tuple[str, ...], ...], dict[str, int]]:
    """Sort members and groups to assign stable integer community IDs."""
    groups = [set(members) for members in communities if members]
    assigned = set().union(*groups) if groups else set()
    missing = set(nodes) - assigned
    groups.extend({node} for node in missing)

    flattened = [node for group in groups for node in group]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("Community algorithm returned overlapping memberships")
    if set(flattened) != set(nodes):
        raise RuntimeError("Community algorithm did not assign every graph node")

    stable_groups = [tuple(sorted((str(node) for node in group))) for group in groups]
    stable_groups.sort()
    assignment = {
        user_id: community_id
        for community_id, members in enumerate(stable_groups)
        for user_id in members
    }
    return tuple(stable_groups), assignment


def detect_communities(graph: nx.DiGraph, *, seed: int = 42) -> CommunityDetectionResult:
    """Detect communities using Louvain (or NetworkX greedy modularity fallback).

    Community IDs are assigned by lexicographically sorting each community's
    user IDs. Modularity is omitted for graphs without edges because it is not
    defined when total edge weight is zero.
    """
    projection = create_undirected_projection(graph)
    nodes = list(projection.nodes)

    if projection.number_of_nodes() == 0:
        raw_communities: list[set[Any]] = []
        algorithm = "louvain" if hasattr(nx.community, "louvain_communities") else "greedy_modularity"
    elif projection.number_of_edges() == 0:
        raw_communities = [{node} for node in nodes]
        algorithm = "louvain" if hasattr(nx.community, "louvain_communities") else "greedy_modularity"
    elif hasattr(nx.community, "louvain_communities"):
        # Sort graph insertion order before the seeded algorithm for stable ties.
        ordered_projection = nx.Graph()
        for node in sorted(nodes, key=str):
            ordered_projection.add_node(node, **projection.nodes[node])
        for source, target, data in sorted(
            projection.edges(data=True),
            key=lambda edge: (str(min(edge[0], edge[1], key=str)), str(max(edge[0], edge[1], key=str))),
        ):
            ordered_projection.add_edge(source, target, **data)
        raw_communities = list(
            nx.community.louvain_communities(ordered_projection, weight="weight", seed=seed)
        )
        algorithm = "louvain"
    else:
        raw_communities = list(nx.community.greedy_modularity_communities(projection, weight="weight"))
        algorithm = "greedy_modularity"

    communities, assignments = _canonicalize_communities(raw_communities, nodes)
    modularity: float | None = None
    if projection.number_of_edges() > 0 and communities:
        modularity = float(nx.community.modularity(projection, [set(group) for group in communities], weight="weight"))

    return CommunityDetectionResult(
        assignments=assignments,
        communities=communities,
        modularity=modularity,
        algorithm=algorithm,
    )


def build_membership_dataframe(
    graph: nx.DiGraph,
    detection: CommunityDetectionResult,
    centrality_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create one membership row per graph node and reuse centrality output.

    Missing centrality values remain empty; this function does not recalculate
    metrics. A left join preserves users even if a centrality CSV is incomplete.
    """
    rows = []
    for user_id, attributes in graph.nodes(data=True):
        stable_id = str(user_id)
        if stable_id not in detection.assignments:
            raise ValueError(f"No community assignment exists for user_id {stable_id!r}")
        rows.append(
            {
                "user_id": stable_id,
                "username": str(attributes.get("username", stable_id)),
                "community_id": detection.assignments[stable_id],
                "post_count": int(attributes.get("post_count", 0)),
            }
        )
    membership = pd.DataFrame(rows, columns=["user_id", "username", "community_id", "post_count"])

    if centrality_results is not None:
        if "user_id" not in centrality_results.columns:
            raise ValueError("centrality results must include a user_id column")
        centrality = centrality_results.copy()
        centrality["user_id"] = centrality["user_id"].astype(str)
        if centrality["user_id"].duplicated().any():
            raise ValueError("centrality results contain duplicate user_id values")
        available_metrics = [column for column in CENTRALITY_COLUMNS if column in centrality.columns]
        membership = membership.merge(
            centrality[["user_id", *available_metrics]], on="user_id", how="left", validate="one_to_one"
        )

    for column in CENTRALITY_COLUMNS:
        if column not in membership.columns:
            membership[column] = np.nan
    membership = membership.loc[:, MEMBERSHIP_COLUMNS]
    return membership.sort_values("user_id", kind="mergesort").reset_index(drop=True)


def _parse_json_list(value: Any, *, column: str, row_number: int) -> list[str]:
    """Read a Phase 2 JSON-array column from pandas values or CSV strings."""
    if value is None:
        return []
    try:
        missing = pd.isna(value)
        if np.isscalar(missing) and bool(missing):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    elif isinstance(value, str):
        if not value.strip():
            return []
        try:
            raw_values = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{column} at data row {row_number} must be a JSON array") from error
        if not isinstance(raw_values, list):
            raise ValueError(f"{column} at data row {row_number} must be a JSON array")
    else:
        raise ValueError(f"{column} at data row {row_number} must be a JSON array")
    return [str(item).strip() for item in raw_values if not pd.isna(item) and str(item).strip()]


def _join_user_id(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip()
    return normalized or None


def summarize_community_posts(
    posts: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate topics, sentiment, hashtags, and existing TF-IDF terms by community.

    Existing ``topic`` values are treated as predefined labels. Existing
    per-post ``keywords`` are aggregated as descriptive TF-IDF terms; they are
    not presented as discovered semantic topics. Returns both the summary and
    a dataframe of posts whose user IDs had no community membership.
    """
    missing_posts = [column for column in SUMMARY_REQUIRED_COLUMNS if column not in posts.columns]
    if missing_posts:
        raise ValueError(f"Processed posts are missing required columns: {', '.join(missing_posts)}")
    missing_membership = [column for column in ("user_id", "community_id") if column not in membership.columns]
    if missing_membership:
        raise ValueError(f"Community membership is missing required columns: {', '.join(missing_membership)}")
    if membership["user_id"].astype(str).duplicated().any():
        raise ValueError("Community membership must contain one row per user_id")

    post_frame = posts.copy()
    post_frame["_join_user_id"] = post_frame["user_id"].map(_join_user_id)
    membership_frame = membership[["user_id", "community_id"]].copy()
    membership_frame["_join_user_id"] = membership_frame["user_id"].map(_join_user_id)
    community_by_user = dict(zip(membership_frame["_join_user_id"], membership_frame["community_id"], strict=True))
    post_frame["community_id"] = post_frame["_join_user_id"].map(community_by_user)

    unmatched_posts = post_frame[post_frame["community_id"].isna()].drop(
        columns=["_join_user_id", "community_id"]
    )
    matched_posts = post_frame[post_frame["community_id"].notna()].copy()
    matched_posts["community_id"] = matched_posts["community_id"].astype(int)
    compound_scores = pd.to_numeric(matched_posts["sentiment_compound"], errors="coerce")
    matched_posts["_compound"] = compound_scores

    user_counts = membership.groupby("community_id")["user_id"].nunique().to_dict()
    rows: list[dict[str, Any]] = []
    for community_id in sorted(user_counts):
        group_posts = matched_posts[matched_posts["community_id"] == community_id]
        post_total = len(group_posts)

        topic_counts: Counter[str] = Counter()
        for topic in group_posts["topic"]:
            if not pd.isna(topic) and str(topic).strip():
                topic_counts[str(topic)] += 1

        labels = group_posts["sentiment_label"].fillna("").astype(str).str.strip().str.casefold()
        sentiment_counts = {label: int((labels == label).sum()) for label in SENTIMENT_LABELS}
        denominator = post_total
        sentiment_proportions = {
            label: (sentiment_counts[label] / denominator if denominator else 0.0)
            for label in SENTIMENT_LABELS
        }

        hashtag_counts: Counter[str] = Counter()
        keyword_counts: Counter[str] = Counter()
        has_hashtags = "hashtags" in group_posts.columns
        has_keywords = "keywords" in group_posts.columns
        if has_hashtags or has_keywords:
            for row_number, (_, post) in enumerate(group_posts.iterrows(), start=1):
                if has_hashtags:
                    hashtags = _parse_json_list(post["hashtags"], column="hashtags", row_number=row_number)
                    hashtag_counts.update(set(hashtags))  # Count a tag at most once per post.
                if has_keywords:
                    keywords = _parse_json_list(post["keywords"], column="keywords", row_number=row_number)
                    keyword_counts.update(set(keywords))  # Count a term at most once per post.

        top_hashtags = [
            {"hashtag": item, "post_count": count}
            for item, count in sorted(hashtag_counts.items(), key=lambda pair: (-pair[1], pair[0].casefold()))[:top_n]
        ]
        top_keywords = [
            {"term": item, "post_count": count}
            for item, count in sorted(keyword_counts.items(), key=lambda pair: (-pair[1], pair[0].casefold()))[:top_n]
        ]

        rows.append(
            {
                "community_id": int(community_id),
                "num_users": int(user_counts[community_id]),
                "num_posts": int(post_total),
                "topic_distribution": json.dumps(dict(sorted(topic_counts.items())), ensure_ascii=False),
                "positive_post_count": sentiment_counts["positive"],
                "negative_post_count": sentiment_counts["negative"],
                "neutral_post_count": sentiment_counts["neutral"],
                "sentiment_labeled_post_count": sum(sentiment_counts.values()),
                "positive_sentiment_proportion": sentiment_proportions["positive"],
                "negative_sentiment_proportion": sentiment_proportions["negative"],
                "neutral_sentiment_proportion": sentiment_proportions["neutral"],
                "mean_compound_sentiment": float(group_posts["_compound"].mean()) if group_posts["_compound"].notna().any() else np.nan,
                "top_hashtags": json.dumps(top_hashtags, ensure_ascii=False),
                "top_keywords": json.dumps(top_keywords, ensure_ascii=False),
            }
        )

    return pd.DataFrame(rows, columns=COMMUNITY_SUMMARY_COLUMNS), unmatched_posts.reset_index(drop=True)


def _community_colors(community_ids: list[int]) -> dict[int, str]:
    palette = qualitative.Safe
    return {community_id: palette[index % len(palette)] for index, community_id in enumerate(community_ids)}


def create_community_visualization(
    graph: nx.DiGraph,
    detection: CommunityDetectionResult,
    output_path: str | Path = DEFAULT_HTML_PATH,
) -> Path:
    """Save an interactive PyVis visualization of the original directed graph."""
    destination = _resolve_path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    community_ids = sorted(set(detection.assignments.values()))
    colors = _community_colors(community_ids)

    network = Network(
        height="780px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#202124",
        cdn_resources="in_line",
    )
    network.set_options(
        """
        {
          "interaction": {"hover": true, "navigationButtons": true, "keyboard": {"enabled": true}},
          "physics": {"enabled": true, "stabilization": {"iterations": 120}},
          "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 0.65}}, "smooth": {"type": "dynamic"}},
          "nodes": {"font": {"size": 14, "face": "Arial"}, "borderWidth": 1}
        }
        """
    )

    for user_id, attributes in graph.nodes(data=True):
        stable_id = str(user_id)
        community_id = detection.assignments[stable_id]
        username = str(attributes.get("username", stable_id))
        post_count = int(attributes.get("post_count", 0))
        title = (
            f"<b>{html.escape(username)}</b><br>user_id: {html.escape(stable_id)}"
            f"<br>Community ID: {community_id}<br>Posts: {post_count}"
        )
        activity = int(attributes.get("interactions_sent", 0)) + int(attributes.get("interactions_received", 0))
        size = 14 + min(34, 2 * math.sqrt(post_count) + 1.5 * activity)
        network.add_node(
            user_id,
            label=username,
            title=title,
            size=size,
            color=colors[community_id],
        )

    for source, target, attributes in graph.edges(data=True):
        weight = float(attributes.get("weight", 1))
        title = (
            f"{html.escape(str(graph.nodes[source].get('username', source)))} → "
            f"{html.escape(str(graph.nodes[target].get('username', target)))}"
            f"<br>Interaction weight: {weight:g}"
            f"<br>Mentions: {attributes.get('mention_count', 0)}"
            f"<br>Replies: {attributes.get('reply_count', 0)}"
        )
        network.add_edge(source, target, value=weight, title=title, width=1 + min(7, math.log1p(weight)))

    legend_items = "".join(
        f'<span style="display:inline-block;margin:3px 10px 3px 0">'
        f'<i style="display:inline-block;width:12px;height:12px;border-radius:50%;'
        f'background:{colors[community_id]};margin-right:5px"></i>'
        f'Community {community_id}</span>'
        for community_id in community_ids
    )
    legend = (
        '<div style="position:fixed;left:12px;bottom:12px;z-index:10;'
        'background:#fff;border:1px solid #bbb;border-radius:6px;padding:10px 12px;'
        'font:13px Arial;max-width:440px">'
        '<b>Community network</b><br>Each node is a user colored by its detected community. '
        'Arrows preserve the original direction: author → mentioned/replied-to user. '
        'Isolated users remain visible. Hover for usernames, community IDs, and edge counts. '
        'Drag nodes to rearrange; use the mouse wheel to zoom.<br>'
        f'{legend_items}</div>'
    )
    html_document = network.generate_html(notebook=False)
    html_document = html_document.replace("<body>", f"<body>\n{legend}", 1)
    destination.write_text(html_document, encoding="utf-8")
    return destination


def create_community_size_chart(summary: pd.DataFrame, output_dir: str | Path = DEFAULT_FIGURES_DIR) -> Path:
    """Save a Plotly HTML bar chart of detected community sizes."""
    destination = _resolve_path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    figure = go.Figure(
        data=[go.Bar(x=summary["community_id"], y=summary["num_users"], text=summary["num_users"], textposition="auto")]
    )
    figure.update_layout(
        title="Users per detected community",
        xaxis_title="Community ID (arbitrary label)",
        yaxis_title="Number of users",
        template="plotly_white",
    )
    path = destination / "community_sizes.html"
    figure.write_html(path, include_plotlyjs="inline", full_html=True)
    return path


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def run_community_analysis(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    *,
    centrality_path: str | Path = DEFAULT_CENTRALITY_PATH,
    membership_path: str | Path = DEFAULT_MEMBERSHIP_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    html_path: str | Path = DEFAULT_HTML_PATH,
    unmatched_path: str | Path = DEFAULT_UNMATCHED_PATH,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    seed: int = 42,
) -> dict[str, Any]:
    """Run community detection and all community-level output steps."""
    posts_file = _resolve_path(input_path)
    posts = pd.read_csv(posts_file)
    graph = build_graph(posts)
    detection = detect_communities(graph, seed=seed)

    centrality_file = _resolve_path(centrality_path)
    centrality = pd.read_csv(centrality_file) if centrality_file.exists() else None
    membership = build_membership_dataframe(graph, detection, centrality)
    summary, unmatched_posts = summarize_community_posts(posts, membership)

    membership_file = _resolve_path(membership_path)
    summary_file = _resolve_path(summary_path)
    metadata_file = _resolve_path(metadata_path)
    unmatched_file = _resolve_path(unmatched_path)
    for path in (membership_file, summary_file, metadata_file, unmatched_file):
        path.parent.mkdir(parents=True, exist_ok=True)
    membership.to_csv(membership_file, index=False, encoding="utf-8")
    summary.to_csv(summary_file, index=False, encoding="utf-8")
    unmatched_posts.to_csv(unmatched_file, index=False, encoding="utf-8")

    metadata = {
        "algorithm": detection.algorithm,
        "algorithm_seed": seed if detection.algorithm == "louvain" else None,
        "community_ids_are_rankings": False,
        "graph_representation": (
            "Weighted undirected projection for modularity; for each unordered user pair, "
            "sum the directed interaction weights in both directions. The original directed graph is used for display."
        ),
        "modularity": detection.modularity,
        "num_communities": len(detection.communities),
        "community_sizes": {str(row.community_id): int(row.num_users) for _, row in summary.iterrows()},
        "num_input_posts": int(len(posts)),
        "num_matched_posts": int(len(posts) - len(unmatched_posts)),
        "num_unmatched_posts": int(len(unmatched_posts)),
        "unmatched_user_ids": sorted(
            str(value) for value in unmatched_posts["user_id"].dropna().unique()
        ) if "user_id" in unmatched_posts.columns else [],
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    html_file = create_community_visualization(graph, detection, html_path)
    size_chart = create_community_size_chart(summary, figures_dir)

    print(f"Algorithm: {detection.algorithm} on weighted undirected projection")
    print(f"Communities: {len(detection.communities)}; modularity: {detection.modularity}")
    print(f"Community sizes: {metadata['community_sizes']}")
    print(f"Posts accounted for: {metadata['num_matched_posts']}/{len(posts)}")
    print(f"Unmatched posts: {len(unmatched_posts)}; details: {unmatched_file}")
    print(f"Membership CSV: {membership_file}")
    print(f"Community summary CSV: {summary_file}")
    print(f"Community graph: {html_file}")
    print(f"Metadata JSON: {metadata_file}")
    print(f"Community-size chart: {size_chart}")
    print("Community IDs are arbitrary labels, and the source posts are fictional synthetic data.")
    return {
        "graph": graph,
        "detection": detection,
        "membership": membership,
        "summary": summary,
        "unmatched_posts": unmatched_posts,
        "metadata": metadata,
        "outputs": {
            "membership": membership_file,
            "summary": summary_file,
            "metadata": metadata_file,
            "unmatched_posts": unmatched_file,
            "html": html_file,
            "community_size_chart": size_chart,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect communities and summarize community-level NLP results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Processed posts CSV")
    parser.add_argument("--centrality", type=Path, default=DEFAULT_CENTRALITY_PATH, help="Optional existing centrality CSV")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic Louvain detection")
    args = parser.parse_args()
    run_community_analysis(args.input, centrality_path=args.centrality, seed=args.seed)


if __name__ == "__main__":
    main()
