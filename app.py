"""Read-only Streamlit dashboard for the synthetic NLP social-network project."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative

from src.nlp_social_network.graph_builder import PROJECT_ROOT, build_graph


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REQUIRED_FILES = (
    "processed_posts.csv",
    "graph_summary.json",
    "centrality_results.csv",
    "community_membership.csv",
    "community_summary.csv",
    "community_detection_summary.json",
)

POST_COLUMNS = (
    "post_id",
    "user_id",
    "username",
    "text",
    "topic",
    "cleaned_text",
    "extracted_mentions",
    "hashtags",
    "keywords",
    "sentiment_compound",
    "sentiment_label",
    "reply_to_user",
)
CENTRALITY_COLUMNS = (
    "user_id",
    "username",
    "degree_centrality",
    "closeness_centrality",
    "betweenness_centrality",
    "eigenvector_centrality",
    "pagerank",
    "post_count",
    "degree_rank",
    "closeness_rank",
    "betweenness_rank",
    "eigenvector_rank",
    "pagerank_rank",
)
MEMBERSHIP_COLUMNS = (
    "user_id",
    "username",
    "community_id",
    "post_count",
    "degree_centrality",
    "betweenness_centrality",
    "pagerank",
)
COMMUNITY_COLUMNS = (
    "community_id",
    "num_users",
    "num_posts",
    "topic_distribution",
    "positive_post_count",
    "negative_post_count",
    "neutral_post_count",
    "positive_sentiment_proportion",
    "negative_sentiment_proportion",
    "neutral_sentiment_proportion",
    "mean_compound_sentiment",
    "top_hashtags",
    "top_keywords",
)

SENTIMENT_LABELS = ("positive", "negative", "neutral")
SENTIMENT_COLORS = {"positive": "#2e8b57", "negative": "#d9534f", "neutral": "#808080"}
NETWORK_SIZE_METRICS = {
    "Degree centrality": "degree_centrality",
    "Betweenness centrality": "betweenness_centrality",
    "Eigenvector centrality": "eigenvector_centrality",
    "PageRank": "pagerank",
}
RANK_METRICS = {
    "Degree centrality": ("degree_centrality", "degree_rank"),
    "Closeness centrality": ("closeness_centrality", "closeness_rank"),
    "Betweenness centrality": ("betweenness_centrality", "betweenness_rank"),
    "Eigenvector centrality": ("eigenvector_centrality", "eigenvector_rank"),
    "PageRank": ("pagerank", "pagerank_rank"),
}


class DashboardDataError(ValueError):
    """An actionable problem with generated dashboard input files."""


def validate_columns(dataframe: pd.DataFrame, required: tuple[str, ...], filename: str) -> None:
    """Raise a readable error when a generated artifact has an incomplete schema."""
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise DashboardDataError(
            f"`{filename}` is missing required columns: {', '.join(missing)}. "
            "Regenerate that phase's output, then reload the dashboard."
        )


def _file_signature(output_dir: Path) -> tuple[tuple[str, int, int], ...]:
    """Create a cache key that changes when an output file is replaced."""
    signature = []
    for filename in REQUIRED_FILES:
        path = output_dir / filename
        if path.exists():
            stat = path.stat()
            signature.append((filename, stat.st_mtime_ns, stat.st_size))
        else:
            signature.append((filename, -1, -1))
    return tuple(signature)


def load_dashboard_data(output_dir: str | Path = OUTPUTS_DIR) -> dict[str, Any]:
    """Load and validate existing generated artifacts without modifying them."""
    output_dir = Path(output_dir)
    missing = [filename for filename in REQUIRED_FILES if not (output_dir / filename).is_file()]
    if missing:
        expected = ", ".join(f"`outputs/{name}`" for name in missing)
        raise DashboardDataError(
            f"Required dashboard file(s) not found: {expected}. Run the Phase 2 NLP pipeline, "
            "Phase 3 graph builder, Phase 4 centrality analysis, and Phase 5 community detection "
            "from the project root, then reload this page."
        )

    try:
        posts = pd.read_csv(output_dir / "processed_posts.csv")
        graph_summary = json.loads((output_dir / "graph_summary.json").read_text(encoding="utf-8"))
        centrality = pd.read_csv(output_dir / "centrality_results.csv")
        membership = pd.read_csv(output_dir / "community_membership.csv")
        community_summary = pd.read_csv(output_dir / "community_summary.csv")
        community_metadata = json.loads(
            (output_dir / "community_detection_summary.json").read_text(encoding="utf-8")
        )
    except (OSError, pd.errors.ParserError, json.JSONDecodeError) as error:
        raise DashboardDataError(
            f"Could not read a generated dashboard file: {error}. Rerun the phase that creates that file."
        ) from error

    validate_columns(posts, POST_COLUMNS, "processed_posts.csv")
    validate_columns(centrality, CENTRALITY_COLUMNS, "centrality_results.csv")
    validate_columns(membership, MEMBERSHIP_COLUMNS, "community_membership.csv")
    validate_columns(community_summary, COMMUNITY_COLUMNS, "community_summary.csv")

    if "num_edges" not in graph_summary:
        raise DashboardDataError("`graph_summary.json` is missing `num_edges`; rerun the Phase 3 graph builder.")
    if "modularity" not in community_metadata:
        raise DashboardDataError(
            "`community_detection_summary.json` is missing `modularity`; rerun Phase 5 community detection."
        )

    for table_name, frame in (("centrality_results.csv", centrality), ("community_membership.csv", membership)):
        if frame["user_id"].astype(str).duplicated().any():
            raise DashboardDataError(f"`{table_name}` has duplicate user_id rows; rerun its analysis phase.")

    post_users = set(posts["user_id"].dropna().astype(str))
    centrality_users = set(centrality["user_id"].dropna().astype(str))
    membership_users = set(membership["user_id"].dropna().astype(str))
    if post_users != centrality_users:
        raise DashboardDataError(
            "The processed posts and centrality results do not contain the same users. "
            "Rerun Phase 4 centrality analysis after the Phase 2 pipeline."
        )
    if post_users != membership_users:
        raise DashboardDataError(
            "The processed posts and community membership do not contain the same users. "
            "Rerun Phase 5 community detection after the Phase 2 pipeline."
        )

    return {
        "posts": posts,
        "graph_summary": graph_summary,
        "centrality": centrality,
        "membership": membership,
        "community_summary": community_summary,
        "community_metadata": community_metadata,
    }


@st.cache_data(show_spinner=False)
def _cached_load_dashboard_data(output_dir: str, signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    """Cache parsed tables, invalidating the cache when generated files change."""
    del signature  # Its values participate in Streamlit's cache key.
    return load_dashboard_data(output_dir)


@st.cache_data(show_spinner="Preparing the network view…")
def _cached_build_graph(posts: pd.DataFrame) -> nx.DiGraph:
    """Build the graph using the existing, tested Phase 3 business logic."""
    return build_graph(posts)


@st.cache_data(show_spinner=False)
def _cached_layout(
    nodes: tuple[str, ...],
    edges: tuple[tuple[str, str, float], ...],
) -> dict[str, tuple[float, float]]:
    """Cache deterministic spring-layout coordinates for the network explorer."""
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges, weight="weight")
    if not graph:
        return {}
    positions = nx.spring_layout(graph, seed=42, weight="weight")
    return {str(user_id): (float(position[0]), float(position[1])) for user_id, position in positions.items()}


def parse_json_list(value: Any) -> list[str]:
    """Parse a JSON-encoded string list used in existing processed outputs."""
    if value is None:
        return []
    try:
        missing = pd.isna(value)
        if np.isscalar(missing) and bool(missing):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, set)):
        values = list(value)
    elif isinstance(value, str):
        if not value.strip():
            return []
        try:
            values = json.loads(value)
        except json.JSONDecodeError as error:
            raise DashboardDataError("A processed post contains a malformed JSON-list value.") from error
        if not isinstance(values, list):
            raise DashboardDataError("A processed post contains a value that is not a JSON array.")
    else:
        raise DashboardDataError("A processed post contains an unsupported JSON-list value.")
    return [str(item).strip() for item in values if item is not None and str(item).strip()]


def token_frequencies(frame: pd.DataFrame, column: str, *, limit: int = 15) -> pd.DataFrame:
    """Count each hashtag or per-post TF-IDF term at most once per post."""
    counts: Counter[str] = Counter()
    if column not in frame.columns:
        return pd.DataFrame(columns=["term", "post_count"])
    for value in frame[column]:
        counts.update(set(parse_json_list(value)))
    rows = sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold()))[:limit]
    return pd.DataFrame(rows, columns=["term", "post_count"])


def build_network_figure(
    graph: nx.DiGraph,
    centrality: pd.DataFrame,
    membership: pd.DataFrame,
    size_metric: str,
    positions: dict[str, tuple[float, float]] | None = None,
) -> go.Figure:
    """Create a Plotly directed graph with selectable centrality-based node sizes."""
    if positions is None:
        nodes = tuple(sorted((str(node) for node in graph.nodes)))
        edges = tuple(
            (str(source), str(target), float(data.get("weight", 1)))
            for source, target, data in graph.edges(data=True)
        )
        positions = _cached_layout(nodes, edges)

    centrality_by_user = centrality.assign(user_id=centrality["user_id"].astype(str)).set_index("user_id")
    membership_by_user = membership.assign(user_id=membership["user_id"].astype(str)).set_index("user_id")
    metric_values = pd.to_numeric(centrality_by_user[size_metric], errors="coerce").fillna(0.0)
    if len(metric_values) and float(metric_values.max()) > float(metric_values.min()):
        scale = (metric_values - metric_values.min()) / (metric_values.max() - metric_values.min())
        scaled_sizes = 14 + 30 * scale
    else:
        scaled_sizes = pd.Series(22.0, index=metric_values.index)

    figure = go.Figure()
    annotations = []
    for source, target, attributes in graph.edges(data=True):
        source_id, target_id = str(source), str(target)
        x0, y0 = positions[source_id]
        x1, y1 = positions[target_id]
        weight = int(attributes.get("weight", 1))
        source_name = graph.nodes[source].get("username", source_id)
        target_name = graph.nodes[target].get("username", target_id)
        hover = (
            f"{source_name} → {target_name}<br>Interactions: {weight}"
            f"<br>Mentions: {attributes.get('mention_count', 0)}"
            f"<br>Replies: {attributes.get('reply_count', 0)}"
        )
        figure.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={"color": "#9aa0a6", "width": 1 + min(3, weight / 2)},
                hoverinfo="text",
                text=[hover, hover],
                showlegend=False,
            )
        )
        annotations.append(
            {
                "x": x0 + 0.78 * (x1 - x0),
                "y": y0 + 0.78 * (y1 - y0),
                "ax": x0 + 0.58 * (x1 - x0),
                "ay": y0 + 0.58 * (y1 - y0),
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "showarrow": True,
                "arrowhead": 2,
                "arrowsize": 0.8,
                "arrowwidth": 1,
                "arrowcolor": "#7b8188",
                "opacity": 0.65,
            }
        )

    community_ids = sorted(pd.to_numeric(membership["community_id"], errors="coerce").dropna().astype(int).unique())
    palette = qualitative.Safe
    colors = {community_id: palette[index % len(palette)] for index, community_id in enumerate(community_ids)}
    for community_id in community_ids:
        user_ids = [
            str(user_id)
            for user_id in graph.nodes
            if int(membership_by_user.loc[str(user_id), "community_id"]) == community_id
        ]
        x_values, y_values, sizes, customdata = [], [], [], []
        for user_id in user_ids:
            attributes = graph.nodes[user_id]
            username = str(attributes.get("username", user_id))
            centrality_row = centrality_by_user.loc[user_id]
            community_row = membership_by_user.loc[user_id]
            x_values.append(positions[user_id][0])
            y_values.append(positions[user_id][1])
            sizes.append(float(scaled_sizes.get(user_id, 22.0)))
            customdata.append(
                [
                    username,
                    user_id,
                    community_id,
                    int(attributes.get("post_count", 0)),
                    int(attributes.get("interactions_received", 0)),
                    int(attributes.get("interactions_sent", 0)),
                    float(centrality_row["degree_centrality"]),
                    float(centrality_row["betweenness_centrality"]),
                    float(centrality_row["eigenvector_centrality"]),
                    float(centrality_row["pagerank"]),
                ]
            )
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name=f"Community {community_id}",
                marker={"size": sizes, "color": colors[community_id], "line": {"width": 1, "color": "white"}},
                customdata=customdata,
                hovertemplate=(
                    "<b>@%{customdata[0]}</b><br>User ID: %{customdata[1]}"
                    "<br>Community ID: %{customdata[2]}<br>Posts: %{customdata[3]}"
                    "<br>Interactions received: %{customdata[4]}"
                    "<br>Interactions sent: %{customdata[5]}"
                    "<br>Degree centrality: %{customdata[6]:.4f}"
                    "<br>Betweenness: %{customdata[7]:.4f}"
                    "<br>Eigenvector: %{customdata[8]:.4f}"
                    "<br>PageRank: %{customdata[9]:.4f}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=f"Directed user interactions · node size: {size_metric.replace('_', ' ')}",
        annotations=annotations,
        height=660,
        template="plotly_white",
        hovermode="closest",
        xaxis={"visible": False, "showgrid": False, "zeroline": False},
        yaxis={"visible": False, "showgrid": False, "zeroline": False, "scaleanchor": "x", "scaleratio": 1},
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
        legend_title="Detected community",
    )
    return figure


def _sentiment_counts(frame: pd.DataFrame) -> pd.Series:
    labels = frame["sentiment_label"].fillna("").astype(str).str.casefold()
    return labels.value_counts().reindex(SENTIMENT_LABELS, fill_value=0)


def _formatted_tokens(value: Any, *, key: str = "term") -> str:
    try:
        items = json.loads(value) if isinstance(value, str) and value.strip() else []
    except json.JSONDecodeError:
        return ""
    return ", ".join(
        f"{item.get(key, '')} ({item.get('post_count', 0)})" if isinstance(item, dict) else str(item)
        for item in items
    )


def _show_downloads(data: dict[str, Any]) -> None:
    st.subheader("Download project data")
    frames = (
        ("Processed posts", data["posts"], "processed_posts.csv"),
        ("Centrality results", data["centrality"], "centrality_results.csv"),
        ("Community membership", data["membership"], "community_membership.csv"),
        ("Community summaries", data["community_summary"], "community_summary.csv"),
    )
    columns = st.columns(4)
    for column, (label, frame, filename) in zip(columns, frames, strict=True):
        column.download_button(
            label=f"Download {label.lower()}",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            width="stretch",
        )


def _render_overview(data: dict[str, Any], graph: nx.DiGraph) -> None:
    posts = data["posts"]
    counts = _sentiment_counts(posts)
    st.info(
        "This dashboard uses fictional, synthetic posts and usernames. Centrality and community results "
        "describe only this generated dataset and make no claims about real-world people or influence."
    )
    metrics = [
        ("Total posts", len(posts)),
        ("Total users", graph.number_of_nodes()),
        ("Directed edges", graph.number_of_edges()),
        ("Communities", data["community_summary"]["community_id"].nunique()),
    ]
    cols = st.columns(4)
    for column, (label, value) in zip(cols, metrics, strict=True):
        column.metric(label, f"{value:,}")
    sentiment_cols = st.columns(3)
    for column, label in zip(sentiment_cols, SENTIMENT_LABELS, strict=True):
        column.metric(f"{label.title()} posts", f"{int(counts[label]):,}")
    st.caption("Directed edge A → B means a post by A mentions or replies to B. Edge weights aggregate repeated interactions.")


def _render_network(data: dict[str, Any], graph: nx.DiGraph) -> None:
    st.header("Network Explorer")
    st.write("Explore directed interactions. Node colors show detected communities, and isolated users stay visible.")
    label = st.selectbox("Node size metric", list(NETWORK_SIZE_METRICS))
    metric = NETWORK_SIZE_METRICS[label]
    nodes = tuple(sorted(str(node) for node in graph.nodes))
    edges = tuple(
        (str(source), str(target), float(attributes.get("weight", 1)))
        for source, target, attributes in graph.edges(data=True)
    )
    positions = _cached_layout(nodes, edges)
    figure = build_network_figure(graph, data["centrality"], data["membership"], metric, positions)
    st.plotly_chart(figure, width="stretch", config={"scrollZoom": True, "displaylogo": False})
    st.caption(
        "Hover over users for username, ID, community, post count, interactions, and scores. "
        "Hover over an edge for its mention/reply counts. Use the legend to focus communities; zoom and pan to explore."
    )


def _render_rankings(data: dict[str, Any]) -> None:
    st.header("Influencer Rankings")
    st.write("These are mathematical rankings within the fictional network, not measures of real-world influence.")
    selected = st.selectbox("Ranking metric", list(RANK_METRICS))
    metric, rank_column = RANK_METRICS[selected]
    maximum = max(1, len(data["centrality"]))
    top_n = st.slider("Number of top users", min_value=1, max_value=maximum, value=min(10, maximum))
    search = st.text_input("Search usernames", placeholder="Type part of a username")

    ordered = data["centrality"].sort_values([metric, "user_id"], ascending=[False, True], kind="mergesort")
    top_users = ordered.head(top_n).copy()
    chart_data = top_users.sort_values([metric, "user_id"], ascending=[True, False], kind="mergesort")
    chart = px.bar(
        chart_data,
        x=metric,
        y="username",
        orientation="h",
        color=metric,
        color_continuous_scale="Blues",
        hover_data={"user_id": True, rank_column: True, metric: ":.5f"},
        labels={metric: selected, "username": "Username", rank_column: "Rank"},
        title=f"Top {len(top_users)} users by {selected.lower()}",
    )
    chart.update_layout(template="plotly_white", coloraxis_showscale=False, yaxis_title=None)
    st.plotly_chart(chart, width="stretch")

    if search.strip():
        table = ordered[ordered["username"].str.contains(search.strip(), case=False, na=False)].copy()
        st.caption(f"Showing {len(table)} matching users across all ranks; the chart above remains the selected top {top_n}.")
    else:
        table = top_users
    table_columns = ["user_id", "username", rank_column, metric, "post_count"]
    st.dataframe(
        table[table_columns].rename(columns={rank_column: "rank", metric: selected}),
        width="stretch",
        hide_index=True,
    )
    with st.expander("Why do the rankings differ?"):
        st.markdown(
            "- **Degree** counts distinct neighbors; incoming and outgoing scores are separate in the results.\n"
            "- **Closeness** uses directed incoming paths, so it favors users reachable from many others.\n"
            "- **Betweenness** measures how often a user lies on directed shortest paths.\n"
            "- **Eigenvector centrality** rewards incoming links from users who also score highly.\n"
            "- **PageRank** passes rank along directed, interaction-weighted edges and includes teleportation."
        )


def _render_nlp(data: dict[str, Any]) -> None:
    st.header("NLP Analysis")
    posts = data["posts"]
    topics = sorted(posts["topic"].dropna().astype(str).unique())
    sentiments = [label for label in SENTIMENT_LABELS if label in set(posts["sentiment_label"].dropna().astype(str).str.lower())]
    c1, c2 = st.columns(2)
    selected_topics = c1.multiselect("Filter by predefined topic", topics, default=topics)
    selected_sentiments = c2.multiselect("Filter by sentiment", sentiments, default=sentiments)
    labels = posts["sentiment_label"].fillna("").astype(str).str.casefold()
    filtered = posts[posts["topic"].astype(str).isin(selected_topics) & labels.isin(selected_sentiments)].copy()
    st.caption(f"Showing {len(filtered):,} of {len(posts):,} posts. Topic labels are predefined in the synthetic data.")

    counts = _sentiment_counts(filtered)
    sentiment_frame = pd.DataFrame({"sentiment": list(SENTIMENT_LABELS), "posts": [int(counts[x]) for x in SENTIMENT_LABELS]})
    topic_counts = filtered["topic"].value_counts().rename_axis("topic").reset_index(name="posts")
    token_chart_frame = token_frequencies(filtered, "keywords", limit=12)
    hashtag_chart_frame = token_frequencies(filtered, "hashtags", limit=12)

    col1, col2 = st.columns(2)
    sentiment_fig = px.bar(
        sentiment_frame,
        x="sentiment",
        y="posts",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment distribution for filtered posts",
    )
    sentiment_fig.update_layout(template="plotly_white", showlegend=False)
    col1.plotly_chart(sentiment_fig, width="stretch")
    topic_fig = px.bar(topic_counts, x="topic", y="posts", title="Predefined topic distribution")
    topic_fig.update_layout(template="plotly_white", xaxis_title="Existing topic label")
    col2.plotly_chart(topic_fig, width="stretch")

    col3, col4 = st.columns(2)
    keyword_fig = px.bar(token_chart_frame.sort_values("post_count"), x="post_count", y="term", orientation="h", title="Frequent existing TF-IDF keywords")
    keyword_fig.update_layout(template="plotly_white", xaxis_title="Posts containing keyword", yaxis_title=None)
    col3.plotly_chart(keyword_fig, width="stretch")
    hashtag_fig = px.bar(hashtag_chart_frame.sort_values("post_count"), x="post_count", y="term", orientation="h", title="Frequent hashtags")
    hashtag_fig.update_layout(template="plotly_white", xaxis_title="Posts containing hashtag", yaxis_title=None)
    col4.plotly_chart(hashtag_fig, width="stretch")

    st.subheader("Posts")
    display = filtered[["post_id", "timestamp", "username", "topic", "sentiment_label", "sentiment_compound", "text"]].copy()
    display["hashtags"] = filtered["hashtags"].map(lambda value: ", ".join(parse_json_list(value)))
    display["keywords"] = filtered["keywords"].map(lambda value: ", ".join(parse_json_list(value)))
    st.dataframe(display, width="stretch", hide_index=True)


def _render_communities(data: dict[str, Any]) -> None:
    st.header("Community Analysis")
    summary = data["community_summary"].copy().sort_values("community_id")
    metadata = data["community_metadata"]
    modularity = metadata.get("modularity")
    if modularity is None:
        st.info("Modularity is unavailable because this graph has no interaction edges.")
    else:
        st.metric("Weighted modularity", f"{float(modularity):.4f}")
        st.caption("Modularity compares within-community edge weight with a randomized-network expectation. It is a structural summary, not proof of real social groups.")
    st.caption("Community IDs are arbitrary labels. Topic values below are predefined labels; keywords are descriptive TF-IDF terms.")

    col1, col2 = st.columns(2)
    sizes = px.bar(summary, x="community_id", y="num_users", text="num_users", title="Users per community")
    sizes.update_layout(template="plotly_white", xaxis_title="Community ID (arbitrary)", yaxis_title="Users")
    col1.plotly_chart(sizes, width="stretch")
    proportions = summary.melt(
        id_vars="community_id",
        value_vars=["positive_sentiment_proportion", "negative_sentiment_proportion", "neutral_sentiment_proportion"],
        var_name="sentiment",
        value_name="proportion",
    )
    proportions["sentiment"] = proportions["sentiment"].str.replace("_sentiment_proportion", "", regex=False)
    sentiment_fig = px.bar(
        proportions,
        x="community_id",
        y="proportion",
        color="sentiment",
        barmode="group",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment proportions by community",
    )
    sentiment_fig.update_layout(template="plotly_white", yaxis_tickformat=".0%")
    col2.plotly_chart(sentiment_fig, width="stretch")

    topic_rows = []
    for _, row in summary.iterrows():
        try:
            distribution = json.loads(row["topic_distribution"])
        except (json.JSONDecodeError, TypeError):
            distribution = {}
        topic_rows.extend(
            {"community_id": int(row["community_id"]), "topic": topic, "posts": count}
            for topic, count in distribution.items()
        )
    if topic_rows:
        topic_fig = px.bar(
            pd.DataFrame(topic_rows),
            x="community_id",
            y="posts",
            color="topic",
            barmode="stack",
            title="Predefined topic counts by community",
        )
        topic_fig.update_layout(template="plotly_white", xaxis_title="Community ID (arbitrary)", yaxis_title="Posts")
        st.plotly_chart(topic_fig, width="stretch")

    display = summary.copy()
    display["top_hashtags"] = display["top_hashtags"].map(lambda value: _formatted_tokens(value, key="hashtag"))
    display["top_keywords"] = display["top_keywords"].map(_formatted_tokens)
    st.subheader("Community summaries")
    st.dataframe(
        display[
            [
                "community_id",
                "num_users",
                "num_posts",
                "topic_distribution",
                "positive_post_count",
                "negative_post_count",
                "neutral_post_count",
                "positive_sentiment_proportion",
                "negative_sentiment_proportion",
                "neutral_sentiment_proportion",
                "mean_compound_sentiment",
                "top_hashtags",
                "top_keywords",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    st.subheader("Community membership")
    st.dataframe(
        data["membership"].sort_values(["community_id", "username"]),
        width="stretch",
        hide_index=True,
    )


def _render_user_explorer(data: dict[str, Any], graph: nx.DiGraph) -> None:
    st.header("User Explorer")
    membership = data["membership"].copy()
    membership["user_id"] = membership["user_id"].astype(str)
    usernames = membership.set_index("user_id")["username"].to_dict()
    ordered_ids = sorted(usernames, key=lambda user_id: (str(usernames[user_id]).casefold(), user_id))
    user_id = st.selectbox(
        "Select a user",
        ordered_ids,
        format_func=lambda value: f"@{usernames[value]} ({value})",
    )

    posts = data["posts"].copy()
    posts["user_id"] = posts["user_id"].astype(str)
    user_posts = posts[posts["user_id"] == user_id].copy()
    profile = graph.nodes[user_id]
    membership_row = membership[membership["user_id"] == user_id].iloc[0]
    centrality_row = data["centrality"].set_index(data["centrality"]["user_id"].astype(str)).loc[user_id]
    counts = _sentiment_counts(user_posts)
    incoming = list(graph.in_edges(user_id, data=True))
    outgoing = list(graph.out_edges(user_id, data=True))
    incoming_weight = sum(int(edge[2].get("weight", 1)) for edge in incoming)
    outgoing_weight = sum(int(edge[2].get("weight", 1)) for edge in outgoing)

    cols = st.columns(4)
    cols[0].metric("Posts", int(profile.get("post_count", len(user_posts))))
    cols[1].metric("Community", int(membership_row["community_id"]))
    cols[2].metric("Interactions received", incoming_weight, help=f"Across {len(incoming)} distinct incoming users")
    cols[3].metric("Interactions sent", outgoing_weight, help=f"Across {len(outgoing)} distinct outgoing users")

    st.markdown(
        f"**Profile:** @{usernames[user_id]} · `{user_id}` · "
        f"mean compound sentiment {float(user_posts['sentiment_compound'].mean()):.3f}"
        if user_posts["sentiment_compound"].notna().any()
        else f"**Profile:** @{usernames[user_id]} · `{user_id}` · sentiment unavailable"
    )

    centrality_metrics = [
        ("Degree centrality", "degree_centrality", "degree_rank"),
        ("Closeness centrality", "closeness_centrality", "closeness_rank"),
        ("Betweenness centrality", "betweenness_centrality", "betweenness_rank"),
        ("Eigenvector centrality", "eigenvector_centrality", "eigenvector_rank"),
        ("PageRank", "pagerank", "pagerank_rank"),
    ]
    scores = pd.DataFrame(
        [{"metric": label, "score": float(centrality_row[value]), "rank": int(centrality_row[rank])} for label, value, rank in centrality_metrics]
    )
    left, right = st.columns([1, 2])
    left.subheader("Sentiment distribution")
    left.dataframe(
        pd.DataFrame({"sentiment": list(SENTIMENT_LABELS), "posts": [int(counts[label]) for label in SENTIMENT_LABELS]}),
        width="stretch",
        hide_index=True,
    )
    right.subheader("Centrality scores and ranks")
    right.dataframe(scores, width="stretch", hide_index=True)

    st.subheader("Interactions")
    out_col, in_col = st.columns(2)

    def interaction_rows(edges: list[tuple[Any, Any, dict[str, Any]]], *, incoming_edges: bool) -> pd.DataFrame:
        rows = []
        for source, target, attributes in edges:
            other = source if incoming_edges else target
            rows.append(
                {
                    "username": graph.nodes[other].get("username", str(other)),
                    "user_id": str(other),
                    "interaction_weight": int(attributes.get("weight", 1)),
                    "mention_count": int(attributes.get("mention_count", 0)),
                    "reply_count": int(attributes.get("reply_count", 0)),
                }
            )
        return pd.DataFrame(rows, columns=["username", "user_id", "interaction_weight", "mention_count", "reply_count"])

    out_col.caption("Users this account mentions or replies to")
    outgoing_frame = interaction_rows(outgoing, incoming_edges=False)
    if outgoing_frame.empty:
        out_col.info("This user has no outgoing interactions.")
    else:
        out_col.dataframe(outgoing_frame, width="stretch", hide_index=True)
    in_col.caption("Users who mention or reply to this account")
    incoming_frame = interaction_rows(incoming, incoming_edges=True)
    if incoming_frame.empty:
        in_col.info("This user has no incoming interactions.")
    else:
        in_col.dataframe(incoming_frame, width="stretch", hide_index=True)

    st.subheader("Authored posts")
    user_posts["hashtags"] = user_posts["hashtags"].map(lambda value: ", ".join(parse_json_list(value)))
    user_posts["keywords"] = user_posts["keywords"].map(lambda value: ", ".join(parse_json_list(value)))
    st.dataframe(
        user_posts[["post_id", "timestamp", "topic", "sentiment_label", "sentiment_compound", "text", "hashtags", "keywords"]],
        width="stretch",
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title="NLP Social Network Analysis", page_icon="🔎", layout="wide")
    st.title("NLP-Based Social Network Analysis and Influencer Detection")
    st.caption("Explore synthetic posts, directed interactions, network metrics, and detected communities.")

    signature = _file_signature(OUTPUTS_DIR)
    try:
        data = _cached_load_dashboard_data(str(OUTPUTS_DIR), signature)
        graph = _cached_build_graph(data["posts"])
    except DashboardDataError as error:
        st.error(str(error))
        st.stop()
    except (ValueError, KeyError) as error:
        st.error(f"Could not prepare the dashboard graph: {error}. Check the Phase 2 and Phase 3 outputs.")
        st.stop()

    _render_overview(data, graph)
    tabs = st.tabs(["Network Explorer", "Influencer Rankings", "NLP Analysis", "Community Analysis", "User Explorer", "Downloads"])
    with tabs[0]:
        _render_network(data, graph)
    with tabs[1]:
        _render_rankings(data)
    with tabs[2]:
        _render_nlp(data)
    with tabs[3]:
        _render_communities(data)
    with tabs[4]:
        _render_user_explorer(data, graph)
    with tabs[5]:
        st.header("Data Downloads")
        st.caption("Downloads are copies of the existing generated CSV files; opening the dashboard does not modify them.")
        _show_downloads(data)


if __name__ == "__main__":
    main()

