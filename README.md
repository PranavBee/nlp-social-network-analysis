# NLP-Based Social Network Analysis and Influencer Detection

A beginner-friendly Python project for exploring a synthetic social network. It analyzes post sentiment and terms, builds a network graph, calculates centrality metrics, finds communities, and presents the results in an interactive Streamlit dashboard.

## Project status

**Phases 0 through 7 are complete.** Phase 7 adds end-to-end reproducibility checks, a small manually labeled VADER evaluation, and final review documents. The dashboard presents existing graph, centrality, community and NLP outputs without rerunning analysis or modifying generated files.

## Planned phases

1. **Phase 0 — Project initialization:** create the project layout and document setup.
2. **Phase 1 — Synthetic dataset creation:** generate a fictional CSV dataset with a fixed seed.
3. **Phase 2 — NLP analysis:** preprocess post text, score sentiment with NLTK VADER, and analyze terms/topics with TF-IDF. (Complete.)
4. **Phase 3 — Graph construction:** build a directed weighted graph and calculate structural summary statistics. (Complete.)
5. **Phase 4 — Centrality analysis:** calculate directed centrality metrics, rank users, and visualize top users. (Complete.)
6. **Phase 5 — Community detection and community-level NLP analysis:** group users, summarize their posts, and visualize communities. (Complete.)
7. **Phase 6 — Streamlit dashboard:** explore the existing analysis results interactively. (Complete.)
8. **Phase 7 — Final evaluation and project report:** verify reproducibility, document measured results, and prepare project review notes. (Complete.)

No real social media accounts or API keys are needed. Dataset records will be synthetic and fictional.

## Requirements

- Python 3.10 or newer
- pip

## Setup

Run these commands from the project root.

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency list is in `requirements.txt`. It includes pandas, NumPy, NLTK, scikit-learn, NetworkX, PyVis, Plotly, Streamlit, and pytest. These packages are used across the generator's analysis, visualization, dashboard, and test modules; no additional dependency was needed for Phase 7.

## Complete execution guide

From the project root, activate the virtual environment as described above. To recreate the dataset and run all analysis steps in order:

```powershell
python -m src.nlp_social_network.data_generator
python -m src.nlp_social_network.pipeline
python -m src.nlp_social_network.graph_builder
python -m src.nlp_social_network.centrality
python -m src.nlp_social_network.community_detection
```

These commands regenerate their respective files under `data/` and `outputs/`. The generator uses seed `42` by default. Run them when you intend to refresh those artifacts. To inspect the existing results without rerunning analysis, start only the dashboard:

```powershell
python -m streamlit run app.py
```

To run all tests:

```powershell
python -m pytest -v
```

The project report and presentation support notes are in `reports/final_project_report.md` and `reports/project_review_notes.md`. Results are synthetic and do not establish real-world model performance or influence.

## Generate the synthetic dataset

From the project root, activate the virtual environment from the setup steps, then run:

```powershell
python -m src.nlp_social_network.data_generator
```

This writes 180 posts by 30 fictional users to `data/sample_posts.csv`. The default seed is `42`, so rerunning the command recreates the same data. You can choose a different seed or output path:

```powershell
python -m src.nlp_social_network.data_generator --seed 123 --output data/another_sample.csv
```

Post count must be a multiple of 30 so each fictional user receives the same number of posts. For example, use `--posts 300` to generate ten posts per user.

## Dataset columns

The CSV includes `post_id`, `user_id`, `username`, `text`, `timestamp`, `mentions`, `reply_to_user`, and `topic`. `mentions` is a JSON array of usernames. Timestamps are UTC ISO 8601 values. Topics include technology, sports, education, environment, and entertainment. Post text includes positive, negative, and neutral examples, hashtags, mentions, and replies. Some accounts receive frequent mentions, some initiate conversations, a few bridge the topic groups, and one account is intentionally interaction-free.

## Run the dataset tests

With the virtual environment active, run:

```powershell
python -m pytest tests/test_data_generator.py -v
```

## Phase 2 NLP modules

- `src/nlp_social_network/text_processor.py` normalizes text, extracts mentions and hashtags, and creates `cleaned_text` while leaving original post text untouched for sentiment analysis.
- `src/nlp_social_network/sentiment_analyzer.py` uses NLTK VADER to return compound, positive, negative, and neutral scores plus a threshold-based label. It downloads the VADER lexicon only if NLTK cannot find it; if setup fails, it reports an actionable error instead of inventing scores.
- `src/nlp_social_network/topic_analysis.py` uses TF-IDF to extract per-post and overall terms, and summarize terms using the dataset's existing `topic` labels. These summaries are descriptive keyword rankings, not semantic topic discovery or topic modelling.
- `src/nlp_social_network/pipeline.py` validates and loads the Phase 1 CSV, adds the NLP columns, preserves original data, prints a brief summary, and writes `outputs/processed_posts.csv`.

Phase 4's `src/nlp_social_network/centrality.py` reuses the directed graph builder, calculates centrality scores and ranks, then saves `outputs/centrality_results.csv`, `outputs/centrality_summary.json`, and four top-user charts under `outputs/figures/`.

## Phase 3 graph construction

- `src/nlp_social_network/graph_builder.py` builds a directed NetworkX graph from `outputs/processed_posts.csv`. Each `user_id` is a node, and an edge from A to B represents a mention or reply from A to B. It aggregates repeated post-level interactions into `weight`, `mention_count`, and `reply_count` edge attributes.
- Mention and reply data are stored separately in the processed CSV. If a post's reply target is also in its mention list, that single post-target interaction is counted once as a reply. Duplicate mentions in one post are counted once. Unknown references are ignored, and self-loops are excluded unless `--include-self-loops` is requested.
- Node degree and graph edge counts refer to distinct users and ordered pairs. Weighted interaction totals include repeated interactions across posts. Nodes also contain post counts, sent/received interaction totals, and average VADER compound sentiment when available.
- The summary reports nodes, distinct edges, posts, mention/reply interactions, density, average in/out-degree, weakly connected components, and isolated users.

Run the graph builder from the project root:

```powershell
python -m src.nlp_social_network.graph_builder
```

This reads `outputs/processed_posts.csv` and creates `outputs/social_network.html` and `outputs/graph_summary.json`. Open the interactive network in your default browser with:

```powershell
Invoke-Item outputs\social_network.html
```

Use `--include-self-loops` only if you want user-to-self interactions displayed. The PyVis HTML embeds its visualization resources and does not need API keys.

## Phase 4 centrality analysis

Run the analysis from the project root:

```powershell
python -m src.nlp_social_network.centrality
```

`src/nlp_social_network/centrality.py` reuses `build_graph()` from `graph_builder.py`; it does not duplicate graph construction or convert the graph to undirected. Each edge `A → B` means A mentioned or replied to B.

### Metric definitions and direction

- **Degree centrality** is total distinct in-neighbors plus out-neighbors divided by `n - 1`. Because the graph is directed, this score can reach 2. Incoming and outgoing degree centrality are also reported separately, each normalized by `n - 1`.
- **Closeness centrality** uses NetworkX's directed incoming-distance convention: it measures how close users who can reach a given user are. The `wf_improved=True` adjustment accounts for the fraction of the network that can reach that user.
- **Betweenness centrality** counts a user's share of normalized shortest paths between ordered source/target pairs that pass through them. Paths follow edge direction and are unweighted because edge weight is interaction frequency, not distance.
- **Eigenvector centrality** uses NetworkX's incoming-neighbor convention: a user receives a higher score when users with high scores point to them. Edge `weight` is used, so repeated interactions have more influence. Power iteration uses tolerance `1e-4` and retries with up to 5,000 iterations; if it still cannot converge, the analysis raises a clear error instead of silently substituting a score.
- **PageRank** follows directed edges from the user who mentioned/replied to the target. Edge weights affect transition probabilities. Teleportation keeps disconnected and isolated users in the ranking.

Results include a separate rank column for each of the seven scores. Equal scores share the best rank; CSV and chart ties are ordered by `user_id` for reproducibility. The JSON summary contains the top users and distribution statistics for every score. Four self-contained HTML bar charts—degree, closeness, betweenness, and eigenvector centrality—are saved under `outputs/figures/`. Isolated users remain in the CSV and receive valid scores for every metric.

For example, open the degree-centrality chart in PowerShell with:

```powershell
Invoke-Item outputs\figures\degree_centrality_top10.html
```

All user rankings describe only this project's fictional synthetic data; they are not claims about real people or networks.

## Phase 5 community detection

Run community detection and community-level NLP summaries from the project root:

```powershell
python -m src.nlp_social_network.community_detection
```

`src/nlp_social_network/community_detection.py` reuses `build_graph()` and, when present, `outputs/centrality_results.csv`. It does not recalculate centrality. Louvain community detection uses a weighted undirected projection: for each unordered pair of users, the weights from both directed edges are summed. This projection is used because modularity-based Louvain optimizes undirected within-community connections. The original directed graph is retained for display, with arrows pointing from each post author to the user they mentioned or replied to. If Louvain is unavailable in the installed NetworkX version, the module falls back to greedy modularity.

Every graph user is assigned, including isolated users (which remain singleton communities). Community IDs are deterministic labels produced by sorting groups by user ID. They are arbitrary labels, not rankings. Weighted modularity is reported when the projected graph has interaction edges; it is left blank for edgeless graphs where modularity is undefined.

Community summaries join posts to membership by `user_id`. Unmatched posts are reported and written to `outputs/unmatched_community_posts.csv`; they are not silently discarded. The summary includes user/post counts, a distribution of the dataset's **predefined** `topic` labels, sentiment counts and proportions, mean compound sentiment, and frequent hashtags and TF-IDF terms aggregated from Phase 2's per-post keyword lists. These keyword summaries are descriptive and do not claim to discover semantic topics.

The command writes:

- `outputs/community_membership.csv` — one row per user, with community ID, post count, and existing degree, betweenness, and PageRank values when available.
- `outputs/community_summary.csv` — one row per community with topic, sentiment, hashtag, and keyword summaries. Structured values are JSON encoded in their CSV cells.
- `outputs/community_detection_summary.json` — algorithm, projection details, modularity, community sizes, and matched/unmatched post counts.
- `outputs/unmatched_community_posts.csv` — posts whose user IDs were missing from membership (header only when all posts match).
- `outputs/community_network.html` — self-contained interactive PyVis graph colored by community, with directed edges and isolated users.
- `outputs/figures/community_sizes.html` — interactive community-size bar chart.

Open the network in PowerShell with:

```powershell
Invoke-Item outputs\community_network.html
```

Open the community-size chart with:

```powershell
Invoke-Item outputs\figures\community_sizes.html
```

Community assignments and summaries describe only the fictional synthetic dataset.

## Phase 6 Streamlit dashboard

Start the dashboard from the project root:

```powershell
python -m streamlit run app.py
```

Streamlit prints the local URL (normally `http://localhost:8501`). The dashboard reads the generated CSV/JSON outputs and the existing processed posts. It validates the required files and columns before rendering and shows an actionable error when an artifact is missing. It does not regenerate the dataset, rerun NLP processing, or write to the source or output files.

The dashboard has these sections:

- **Overview:** post, user, edge, community and sentiment counts. All results are based on fictional synthetic data and are not real-world influence claims.
- **Network Explorer:** directed author-to-target interactions, colored by community, with a selectable node-size metric and user tooltips. Isolated users are shown.
- **Influencer Rankings:** selectable centrality metric and top-user count, username search, sortable results, and a bar chart. Metrics rank different graph properties, so their rankings can differ.
- **NLP Analysis:** filter posts by existing topic label and sentiment, inspect sentiment distribution, TF-IDF terms, hashtag and topic frequencies, and review matching posts. Topic labels are predefined; TF-IDF terms are descriptive.
- **Community Analysis:** community sizes and membership, topic and sentiment comparisons, mean sentiment, and community keywords and hashtags. Modularity (when available) measures how strongly the weighted undirected projection separates into groups; it is not a measure of real-world influence.
- **User Explorer:** inspect a user's posts, sentiment, community, centrality scores and ranks, incoming/outgoing interactions, and interaction partners. Users with no interactions remain selectable.

CSV download buttons are provided for processed posts, centrality results, community membership and community summaries. To run the complete tests, use `python -m pytest -v`.

## Run the NLP pipeline

With the project virtual environment active, run from the project root:

```powershell
python -m src.nlp_social_network.pipeline
```

The input and output locations are resolved from the project root, so this module command works when launched from the root directory.

## Run all tests

```powershell
python -m pytest -v
```

## Project layout

```text
.
├── data/
│   └── sample_posts.csv  # Generated fictional posts
├── outputs/              # Generated results and exports
├── src/
│   └── nlp_social_network/
│       ├── data_generator.py
│       ├── text_processor.py
│       ├── sentiment_analyzer.py
│       ├── topic_analysis.py
│       ├── pipeline.py
│       ├── graph_builder.py
│       ├── centrality.py
│       └── community_detection.py
├── app.py                 # Read-only Streamlit dashboard
├── reports/
│   ├── final_project_report.md
│   └── project_review_notes.md
├── tests/
│   ├── test_data_generator.py
│   ├── test_text_processor.py
│   ├── test_sentiment_analyzer.py
│   ├── test_topic_analysis.py
│   ├── test_pipeline.py
│   ├── test_graph_builder.py
│   ├── test_centrality.py
│   ├── test_community_detection.py
│   └── test_dashboard.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Privacy and data

This project is designed for a synthetic dataset. Do not add real account identifiers or private social media content to the sample files.

