# Project Review Notes

## 2-minute project explanation

This project demonstrates how natural language processing and social network analysis can be combined to study a small, fictional social network. It does not download social media posts or use real accounts. A fixed-seed generator creates 180 posts by 30 fictional users across five predefined topics. The posts include positive, negative, and neutral wording, hashtags, mentions, replies, conversation initiators, high-incoming-interaction hubs, bridge users, and one isolated account.

The NLP stage preserves the original post text, normalizes it for feature extraction, uses NLTK VADER to calculate sentiment scores and labels, and applies TF-IDF to identify descriptive terms. The existing topic labels are used to group keyword summaries; the TF-IDF step does not discover topics.

The graph stage creates a directed NetworkX graph: an edge from A to B means A mentioned or replied to B. It aggregates repeated interactions and avoids counting a reply target twice when that target is also mentioned in the same post. The project calculates degree, closeness, betweenness, and eigenvector centrality, plus PageRank. Each metric describes a different graph property, so the rankings can differ.

For community detection, the directed graph is converted to an explicit undirected weighted projection, summing interactions in both directions for each pair. Louvain finds six groups on this projection; the original directed graph remains available for visualization. The Streamlit dashboard lets a reviewer explore network structure, rankings, posts, sentiment, topics, communities, and user profiles, and download the generated CSV summaries.

The reproducibility run regenerated a byte-identical CSV, processed 180 rows, constructed 30 nodes and 100 directed edges, and matched all 180 posts to one of six communities. All 53 automated tests passed. These are results about a deliberately designed synthetic dataset, not claims about real people, social platforms, or model performance. A small set of 15 easy manually labeled examples was scored perfectly by VADER, but that is only a smoke check and not a general accuracy estimate.

## Why combine NLP and social network analysis?

NLP summarizes what posts say: sentiment scores, labels, hashtags, and salient terms. Social network analysis summarizes how users interact: incoming and outgoing links, shortest-path roles, recursive prominence, and community structure. Joining the two by `user_id` enables questions such as whether sentiment or vocabulary differs across graph communities, while preserving the distinction between content features and structural scores. Neither view alone answers both kinds of questions. In this project, the combination is exploratory and descriptive.

## Why these tools?

- **NLTK VADER:** a lightweight, rule-and-lexicon sentiment analyzer designed for social-media-like text. It provides interpretable polarity scores without training a project-specific model. It remains sensitive to context, language, and domain.
- **TF-IDF / scikit-learn:** a compact, established way to weight terms by their document frequency and extract descriptive keywords. It is suitable for this small corpus and avoids claiming semantic topic discovery.
- **NetworkX:** a readable Python graph library with directed graph representations, centrality functions, PageRank, graph summaries, and community algorithms. It makes graph assumptions explicit and testable at this project scale.
- **Louvain:** a fast heuristic for modularity optimization. It works on the documented undirected weighted projection here; it is not a unique or definitive partition, and the result depends on graph representation and algorithm parameters.
- **Streamlit / Plotly:** make existing CSV/JSON analysis results interactive in a local browser without a separate front-end build. The dashboard reads, validates, and displays existing outputs.

## 5-minute demonstration sequence

1. **Set context (30 sec):** open the dashboard and point out the synthetic-data notice and the overview cards: 180 posts, 30 users, 100 directed edges, six communities, and sentiment counts.
2. **Network Explorer (60 sec):** explain the arrow `A → B`, select a node-size metric, hover a node for its username, ID, community and graph statistics, and identify the isolated user. Emphasize that the scores characterize the generated graph only.
3. **Influencer Rankings (45 sec):** choose betweenness, show the top list and bar chart, then compare with PageRank. Explain shortest-path brokerage versus weighted random-walk prominence.
4. **NLP Analysis (45 sec):** filter posts by one predefined topic and sentiment, inspect the table and keyword/hashtag charts. Clarify that keywords are TF-IDF terms and topics are supplied labels.
5. **Community Analysis (55 sec):** show the six community sizes, modularity (0.1976), sentiment proportions and topic distributions. Explain that Louvain used an undirected weighted projection while visual edges remain directed.
6. **User Explorer and downloads (35 sec):** select a high-interaction user, show authored posts, scores/ranks, and incoming/outgoing targets; optionally select the isolated account to show graceful handling. Point out CSV downloads.
7. **Close (10 sec):** report 53 passing tests and a byte-identical generator rerun; restate that this validates this synthetic pipeline, not real-world influence or general sentiment performance.

## Ten likely review questions

1. **What does an edge mean?** An edge `A → B` records a post by A that mentions or replies to B. Repeated post-level events increase the edge weight.
2. **Why is the graph directed?** The author and target have different roles. Direction preserves who initiated the recorded interaction and is required for incoming/outgoing degree, directed paths, eigenvector interpretation, and PageRank.
3. **How do you prevent double-counting replies?** If a post's reply target is also in its mention list, the one post-target event is counted once as a reply. Duplicate mentions in one post are also reduced to one.
4. **Why do centrality rankings differ?** Degree counts distinct adjacent users, closeness measures incoming reachability, betweenness measures directed shortest-path brokerage, eigenvector gives recursive credit to incoming neighbors, and PageRank models a weighted random walk with teleportation.
5. **How is closeness oriented?** NetworkX's directed default uses distances into a node. A high value means many nodes can reach that user by short directed paths, with a reachability adjustment for disconnected graphs.
6. **Why convert the graph for Louvain?** The implementation uses a weighted undirected projection because this modularity-based Louvain analysis operates on undirected community structure. Reciprocal edge weights are summed; the original directed graph is retained for visualization and other metrics.
7. **Are the topics discovered by TF-IDF?** No. Topic labels are predefined in the synthetic dataset. TF-IDF extracts terms and summarizes them within those existing labels.
8. **What does the 15/15 VADER check prove?** Only that VADER labels these 15 deliberately clear, hand-written examples as expected. It is too small and unrepresentative to estimate real-world performance.
9. **Is the project reproducible?** The seed-42 generator recreated a CSV byte-for-byte in an isolated copy, and the documented analysis modules ran end to end there. This verifies the current workflow in this environment, not all future dependency versions.
10. **Can you call the top user a real influencer?** No. The dataset is fictional and its hubs are deliberately constructed. The scores illustrate mathematical graph properties only.

## Limitations and future work

The corpus is small, template-generated, English-only, and balanced by posts per user. VADER can misinterpret sarcasm, mixed emotions, context, slang, multilingual text, and domain-specific language. TF-IDF captures term salience rather than semantic meaning. Centrality depends on the edge definition and graph topology; Louvain depends on the undirected projection, seed, and resolution. The manually labeled check is a smoke test, not a representative evaluation. Dashboard tests validate data loading and figure construction but do not automate every browser interaction or download.

Future work could use a larger, independently labeled and appropriately licensed corpus; report per-class metrics and uncertainty; evaluate alternative sentiment methods; run seed/projection/resolution sensitivity analysis; add temporal network analysis; and compare Louvain with Leiden or other community algorithms. No high-impact decisions should be based on these synthetic outputs.

## Commands for the review

Run from the project root with the virtual environment activated:

```powershell
python -m pytest -v
python -m src.nlp_social_network.data_generator
python -m src.nlp_social_network.pipeline
python -m src.nlp_social_network.graph_builder
python -m src.nlp_social_network.centrality
python -m src.nlp_social_network.community_detection
python -m streamlit run app.py
```
