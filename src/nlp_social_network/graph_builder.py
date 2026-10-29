"""Build and visualize a directed, weighted user-interaction graph.

Nodes represent users and are identified by ``user_id``. A directed edge
``A -> B`` represents one post-level interaction from A to B. Each unique
target in a post's mention list counts once. If the same target is also the
post's ``reply_to_user``, that post-target interaction is classified as a
reply only, so it is never counted twice. Repeated interactions across posts
are accumulated as an edge's ``weight``.

Self-loops are ignored by default. Unknown usernames/IDs in interaction
columns are ignored and never create graph nodes. Degree statistics count
distinct neighboring users; weighted interaction totals count repeated
post-level interactions.
"""

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from pyvis.network import Network


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "outputs" / "processed_posts.csv"
DEFAULT_HTML_PATH = PROJECT_ROOT / "outputs" / "social_network.html"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "graph_summary.json"

REQUIRED_COLUMNS = (
    "post_id",
    "user_id",
    "username",
    "extracted_mentions",
    "reply_to_user",
)


def _clean_identifier(value: Any) -> str:
    """Convert a user identifier to a trimmed string, or empty if missing."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _parse_mention_list(value: Any, *, row_number: int) -> list[str]:
    """Parse extracted mentions stored as JSON arrays in the Phase 2 CSV."""
    if value is None or pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_mentions = list(value)
    elif isinstance(value, str):
        if not value.strip():
            return []
        try:
            raw_mentions = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"extracted_mentions at data row {row_number} must be a JSON array"
            ) from error
        if not isinstance(raw_mentions, list):
            raise ValueError(f"extracted_mentions at data row {row_number} must be a JSON array")
    else:
        raise ValueError(f"extracted_mentions at data row {row_number} must be a JSON array")

    mentions: list[str] = []
    for mention in raw_mentions:
        username = _clean_identifier(mention).removeprefix("@").strip()
        if username and username not in mentions:
            mentions.append(username)
    return mentions


def _build_user_maps(posts: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    """Map normalized usernames and IDs to canonical user IDs and names."""
    username_to_id: dict[str, str] = {}
    user_id_to_username: dict[str, str] = {}

    for row_number, (user_id_value, username_value) in enumerate(
        zip(posts["user_id"], posts["username"], strict=True), start=2
    ):
        user_id = _clean_identifier(user_id_value)
        username = _clean_identifier(username_value).removeprefix("@").strip()
        if not user_id or not username:
            raise ValueError(f"user_id and username must be populated (CSV row {row_number})")

        normalized_username = username.casefold()
        existing_username_id = username_to_id.get(normalized_username)
        if existing_username_id is not None and existing_username_id != user_id:
            raise ValueError(
                f"Username {username!r} maps to multiple user_id values "
                f"({existing_username_id!r} and {user_id!r})"
            )
        existing_username = user_id_to_username.get(user_id)
        if existing_username is not None and existing_username.casefold() != normalized_username:
            raise ValueError(
                f"user_id {user_id!r} maps to multiple usernames "
                f"({existing_username!r} and {username!r})"
            )
        username_to_id[normalized_username] = user_id
        user_id_to_username[user_id] = username

    # Both usernames and stable IDs are accepted in the reference columns.
    reference_to_id = dict(username_to_id)
    for user_id in user_id_to_username:
        normalized_id = user_id.casefold()
        existing = reference_to_id.get(normalized_id)
        if existing is not None and existing != user_id:
            raise ValueError(f"Ambiguous user reference {user_id!r}")
        reference_to_id[normalized_id] = user_id
    return reference_to_id, user_id_to_username


def build_graph(posts: pd.DataFrame, *, include_self_loops: bool = False) -> nx.DiGraph:
    """Create the user interaction graph from a Phase 2 posts dataframe.

    ``post_id``, ``user_id``, ``username``, ``extracted_mentions``, and
    ``reply_to_user`` are required. When available, ``sentiment_compound`` is
    averaged per user and included as ``mean_sentiment``.
    """
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in posts.columns]
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {', '.join(missing_columns)}")

    users_by_reference, usernames_by_id = _build_user_maps(posts)
    graph = nx.DiGraph(
        ignore_self_loops=not include_self_loops,
        edge_weight_definition=(
            "One unique source-target interaction per post; reply takes precedence "
            "when the same target is both mentioned and replied to."
        ),
    )

    for user_id, username in usernames_by_id.items():
        graph.add_node(user_id, username=username, post_count=0)

    sentiment_values: dict[str, list[float]] = defaultdict(list)
    if "sentiment_compound" in posts.columns:
        sentiment = pd.to_numeric(posts["sentiment_compound"], errors="coerce")
    else:
        sentiment = pd.Series([float("nan")] * len(posts), index=posts.index)

    post_ids: set[str] = set()
    for position, (index, row) in enumerate(posts.iterrows(), start=1):
        author_id = _clean_identifier(row["user_id"])
        post_id = _clean_identifier(row["post_id"])
        if not post_id:
            raise ValueError(f"post_id must be populated (data row {position})")
        if post_id in post_ids:
            raise ValueError(f"Duplicate post_id {post_id!r}; each input row must represent a unique post")
        post_ids.add(post_id)
        graph.nodes[author_id]["post_count"] += 1

        score = sentiment.iloc[position - 1]
        if pd.notna(score):
            sentiment_values[author_id].append(float(score))

        mentions = _parse_mention_list(row["extracted_mentions"], row_number=position)
        mention_targets: set[str] = set()
        for mention in mentions:
            target_id = users_by_reference.get(mention.casefold())
            if target_id is not None and (include_self_loops or target_id != author_id):
                mention_targets.add(target_id)

        reply_value = _clean_identifier(row["reply_to_user"]).removeprefix("@").strip()
        reply_target = users_by_reference.get(reply_value.casefold()) if reply_value else None
        if reply_target == author_id and not include_self_loops:
            reply_target = None

        # A reply is a single post-target interaction. Remove a matching target
        # from mentions first so the event is classified exactly once as reply.
        if reply_target is not None:
            mention_targets.discard(reply_target)
        for target_id in mention_targets:
            _add_interaction(graph, author_id, target_id, interaction_type="mention")
        if reply_target is not None:
            _add_interaction(graph, author_id, reply_target, interaction_type="reply")

    graph.graph["number_posts"] = len(post_ids)
    for user_id, scores in sentiment_values.items():
        graph.nodes[user_id]["mean_sentiment"] = sum(scores) / len(scores)
        graph.nodes[user_id]["sentiment_post_count"] = len(scores)

    _add_interaction_activity_attributes(graph)
    return graph


def _add_interaction(graph: nx.DiGraph, source: str, target: str, *, interaction_type: str) -> None:
    """Add one post-target event to the aggregated directed edge."""
    if graph.has_edge(source, target):
        edge = graph[source][target]
        edge["weight"] += 1
        edge[f"{interaction_type}_count"] += 1
    else:
        graph.add_edge(
            source,
            target,
            weight=1,
            mention_count=int(interaction_type == "mention"),
            reply_count=int(interaction_type == "reply"),
        )


def _add_interaction_activity_attributes(graph: nx.DiGraph) -> None:
    """Attach count-weighted sent/received interaction totals to every node."""
    for user_id in graph.nodes:
        sent = sum(data.get("weight", 1) for _, _, data in graph.out_edges(user_id, data=True))
        received = sum(data.get("weight", 1) for _, _, data in graph.in_edges(user_id, data=True))
        graph.nodes[user_id]["interactions_sent"] = int(sent)
        graph.nodes[user_id]["interactions_received"] = int(received)


def summarize_graph(graph: nx.DiGraph, *, number_posts: int | None = None) -> dict[str, Any]:
    """Calculate node, edge, weighted-interaction, and connectivity statistics.

    ``num_edges`` counts distinct ordered user pairs. Mention and reply totals
    sum edge attributes, so repeated interactions contribute to those totals
    without increasing the number of edges.
    """
    node_count = graph.number_of_nodes()
    indegrees = [degree for _, degree in graph.in_degree()]
    outdegrees = [degree for _, degree in graph.out_degree()]
    posts_count = number_posts if number_posts is not None else graph.graph.get("number_posts", 0)

    return {
        "num_nodes": node_count,
        "num_edges": graph.number_of_edges(),
        "num_posts": int(posts_count),
        "num_mention_interactions": sum(data.get("mention_count", 0) for _, _, data in graph.edges(data=True)),
        "num_reply_interactions": sum(data.get("reply_count", 0) for _, _, data in graph.edges(data=True)),
        "density": float(nx.density(graph)) if node_count > 1 else 0.0,
        "average_in_degree": sum(indegrees) / node_count if node_count else 0.0,
        "average_out_degree": sum(outdegrees) / node_count if node_count else 0.0,
        "num_weakly_connected_components": nx.number_weakly_connected_components(graph),
        "num_isolated_nodes": sum(1 for _ in nx.isolates(graph)),
        "degree_counts_distinct_neighbors": True,
        "self_loops_ignored": bool(graph.graph.get("ignore_self_loops", True)),
        "edge_weight_definition": graph.graph.get(
            "edge_weight_definition",
            "One unique source-target interaction per post; a reply takes precedence over a matching mention.",
        ),
    }


def create_visualization(graph: nx.DiGraph, output_path: str | Path = DEFAULT_HTML_PATH) -> Path:
    """Write an interactive, self-contained PyVis HTML graph visualization."""
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)

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
        username = str(attributes.get("username", user_id))
        post_count = int(attributes.get("post_count", 0))
        sent = int(attributes.get("interactions_sent", 0))
        received = int(attributes.get("interactions_received", 0))
        mean_sentiment = attributes.get("mean_sentiment")
        sentiment_line = (
            f"Average sentiment: {mean_sentiment:.3f}"
            if mean_sentiment is not None
            else "Average sentiment: unavailable"
        )
        title = (
            f"<b>{html.escape(username)}</b><br>user_id: {html.escape(str(user_id))}"
            f"<br>Posts: {post_count}<br>Interactions sent: {sent}"
            f"<br>Interactions received: {received}<br>{sentiment_line}"
        )
        size = 14 + min(34, 2 * math.sqrt(post_count) + 1.5 * (sent + received))
        network.add_node(
            user_id,
            label=username,
            title=title,
            size=size,
            color="#84b6f4" if sent + received else "#c9ced6",
        )

    for source, target, attributes in graph.edges(data=True):
        weight = int(attributes.get("weight", 1))
        title = (
            f"{html.escape(str(graph.nodes[source].get('username', source)))} → "
            f"{html.escape(str(graph.nodes[target].get('username', target)))}"
            f"<br>Total interactions: {weight}"
            f"<br>Mentions: {attributes.get('mention_count', 0)}"
            f"<br>Replies: {attributes.get('reply_count', 0)}"
        )
        network.add_edge(source, target, value=weight, width=1 + min(7, math.log1p(weight)), title=title)

    legend = (
        '<div style="position:fixed;left:12px;bottom:12px;z-index:10;'
        'background:#fff;border:1px solid #bbb;border-radius:6px;padding:10px 12px;'
        'font:13px Arial;max-width:360px">'
        '<b>Network key</b><br>Each node is a user; node size reflects posts and '
        'sent/received activity. Gray nodes have no interactions. Directed arrows '
        'show mentions or replies; thicker edges represent more post-level interactions. '
        'Hover over a node or edge for details. Drag nodes to rearrange; use the mouse '
        'wheel to zoom.</div>'
    )
    html_document = network.generate_html(notebook=False)
    html_document = html_document.replace("<body>", f"<body>\n{legend}", 1)
    destination.write_text(html_document, encoding="utf-8")
    return destination


def build_graph_from_csv(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    *,
    html_path: str | Path = DEFAULT_HTML_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    include_self_loops: bool = False,
) -> tuple[nx.DiGraph, dict[str, Any]]:
    """Load the processed CSV, build the graph, and save both output artifacts."""
    source = Path(input_path)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    posts = pd.read_csv(source)
    graph = build_graph(posts, include_self_loops=include_self_loops)
    summary = summarize_graph(graph)

    summary_destination = Path(summary_path)
    if not summary_destination.is_absolute():
        summary_destination = PROJECT_ROOT / summary_destination
    summary_destination.parent.mkdir(parents=True, exist_ok=True)
    summary_destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    create_visualization(graph, html_path)

    print(f"Users (nodes): {summary['num_nodes']}")
    print(f"Distinct directed edges: {summary['num_edges']}")
    print(f"Posts represented: {summary['num_posts']}")
    print(
        "Interactions: "
        f"mentions={summary['num_mention_interactions']}, "
        f"replies={summary['num_reply_interactions']}"
    )
    print(
        "Graph structure: "
        f"density={summary['density']:.4f}, "
        f"average in-degree={summary['average_in_degree']:.2f}, "
        f"average out-degree={summary['average_out_degree']:.2f}, "
        f"weak components={summary['num_weakly_connected_components']}, "
        f"isolated users={summary['num_isolated_nodes']}"
    )
    print(f"Visualization: {Path(html_path) if Path(html_path).is_absolute() else PROJECT_ROOT / html_path}")
    print(f"Summary: {summary_destination}")
    return graph, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the social interaction graph from processed posts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Processed post CSV")
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML_PATH, help="PyVis HTML output")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH, help="JSON summary output")
    parser.add_argument(
        "--include-self-loops",
        action="store_true",
        help="Include a user's interactions with themselves (ignored by default)",
    )
    args = parser.parse_args()
    build_graph_from_csv(
        input_path=args.input,
        html_path=args.html,
        summary_path=args.summary,
        include_self_loops=args.include_self_loops,
    )


if __name__ == "__main__":
    main()
