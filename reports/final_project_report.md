# NLP-Based Social Network Analysis and Influencer Detection

## Abstract

This project is a reproducible, end-to-end demonstration of combining text analysis with directed social-network analysis. A seeded generator creates 180 fictional posts from 30 fictional users. The pipeline preprocesses text, scores sentiment with NLTK VADER, and extracts descriptive terms with TF-IDF. It builds a directed interaction graph, calculates centrality measures, detects communities using Louvain on a weighted undirected projection, and presents the existing results in a read-only Streamlit dashboard. The full suite contains 53 tests and passes. An isolated rerun reproduced the source CSV byte-for-byte and regenerated the analysis outputs. A small manually labeled VADER check scored 15/15, but its deliberately simple examples are not evidence of performance on real or representative social-media language.

## Introduction and problem statement

Short social posts contain both language signals and interaction signals. Text analysis can describe sentiment and recurring terms, while graph analysis can describe who interacts with whom and how users connect structurally. This project demonstrates a transparent workflow joining those views while keeping its data entirely synthetic.

The project does not establish who is influential in a real community. Its rankings are mathematical summaries of a constructed graph. They should be read as an educational demonstration of methods and not as evidence about real people.

## Objectives

- Generate a repeatable dataset with fictional accounts, varied topics, sentiment, mentions, replies, hashtags, and interactions.
- Preserve source posts while adding reproducible NLP-derived fields.
- Represent user-to-user interactions in a directed, weighted graph.
- Calculate four required centralities and additional in/out-degree and PageRank scores.
- Detect and summarize communities, including topic and sentiment attributes.
- Make the existing outputs explorable in a local dashboard.
- Test core behavior and document what the results can and cannot establish.

## System architecture

1. `src/nlp_social_network/data_generator.py` writes the seeded CSV at `data/sample_posts.csv`.
2. `text_processor.py`, `sentiment_analyzer.py`, and `topic_analysis.py` support `pipeline.py`, which writes `outputs/processed_posts.csv`.
3. `graph_builder.py` creates the directed NetworkX graph and writes graph HTML and JSON statistics.
4. `centrality.py` reuses that graph builder and writes one row per user to `centrality_results.csv`, a JSON summary, and four HTML charts.
5. `community_detection.py` reuses the directed graph and centrality CSV, writes community membership and summaries, and emits a community-colored graph.
6. Root-level `app.py` validates and reads the existing artifacts for the Streamlit dashboard. It does not rerun NLP or write analysis files.

## Dataset design and limitations

The sample dataset contains 180 posts, six per each of 30 fictional users, and five predefined labels: education (39), technology (37), entertainment (35), environment (35), and sports (34). Post text and interaction structure are generated from templates and seeded choices. Three accounts receive many incoming interactions, several initiate conversations, selected users connect groups, and `user_30` (`sleepykiwi`) is intentionally isolated.

The default seed is 42. Regenerating the data in an isolated project copy produced a CSV byte-for-byte identical to the existing `data/sample_posts.csv`. That verifies repeatability for this version, seed, Python environment, and default parameters; it does not promise byte identity across arbitrary future generator changes or environments.

Synthetic template data is small, English-only, balanced by author post count, and structurally designed. It lacks the variety, ambiguity, temporal dynamics, sampling bias, bot behavior, and context of real social platforms. The predefined `topic` column is not model-discovered. The generator is not a proxy for a real platform or population.

## NLP preprocessing

Text preprocessing safely normalizes whitespace, extracts `@mentions` and hashtags before cleaning, and adds `cleaned_text` without replacing original `text`. Original punctuation and casing remain in the source text used for VADER. Missing and blank values are handled without fabricated content.

## VADER sentiment analysis

NLTK's VADER `SentimentIntensityAnalyzer` scores each original post. The processed table contains compound, positive, negative, and neutral scores; the compound thresholds are positive at `>= 0.05`, negative at `<= -0.05`, and neutral between them. A missing VADER lexicon triggers an install/setup error rather than substitute scores. Hashtags are preserved for reporting but VADER may not interpret hashtag text unless its marker is removed; this pipeline does not alter hashtags for sentiment scoring.

## TF-IDF keyword extraction

Scikit-learn's `TfidfVectorizer` converts post text into term-weight features. The project retains up to five high-scoring terms per post, an overall term ranking, and descriptive term summaries grouped by the existing topic labels. TF-IDF is a sparse weighting method; it does not infer semantic themes, discover the supplied topic labels, or perform topic modeling.

## Directed graph construction and edge semantics

Every unique `user_id` is a node, including users without interactions. An edge `A → B` means author A mentioned or replied to B. Repeated post-level interactions aggregate into edge weights, with separate mention and reply counts. If the reply target also occurs in the mention list of that post, it is counted once as a reply. Duplicate mentions within one post count once. Unknown users are ignored, and self-loops are excluded by default.

The summary distinguishes users (nodes), unique ordered source-target pairs (edges), and repeated post-level interactions (edge weights). For the evaluated data: 30 nodes, 100 directed edges, 35 mention interactions, 113 reply interactions, and 148 weighted interactions total. Density is 0.1149; average in-degree and out-degree are both 3.33; there are two weakly connected components and one isolated node.

## Centrality definitions

Let `n` be the number of users, and let directed paths follow the edge `A → B` semantics.

- **Degree centrality:** `C_D(v) = (d_in(v) + d_out(v)) / (n - 1)`. It counts distinct adjacent users by direction. A user can share the same neighbor in both directions, so directed NetworkX degree centrality can be as high as 2. In-degree and out-degree centrality are also reported separately, normalized by `n - 1`.
- **Closeness centrality:** uses incoming distances. It asks how close the users that can reach `v` are to `v`, not how quickly `v` can reach everyone. With `R_v` the other users with a directed path to `v`, `C_C(v) = (|R_v| / Σ[u∈R_v] d(u,v)) × (|R_v|/(n-1))`; the second factor is NetworkX's `wf_improved=True` adjustment for disconnected reachability.
- **Betweenness centrality:** `C_B(v) = Σ[s≠v≠t] σ_st(v)/σ_st`, normalized across ordered pairs. `σ_st` is the number of shortest directed paths from s to t and `σ_st(v)` counts those passing through v. Paths are unweighted: interaction frequency is not treated as path distance.
- **Eigenvector centrality:** uses incoming neighbors and edge weights: `x_v = (1/λ) Σ[u→v] w_uv x_u`. A user receives a higher score when pointed to by users who themselves score highly. Iteration retries on convergence failure and otherwise raises an error.

**PageRank** is an additional directed, weighted random-surfer score with damping `α = 0.85`: `PR(v) = (1-α)/n + α Σ[u→v] PR(u) w_uv / Σ[x:u→x] w_ux`, with NetworkX's dangling-node handling. It distributes score along outgoing interactions and includes teleportation, so isolated users retain a score.

These measures capture different graph properties—local adjacency, reachability, shortest-path brokerage, recursive incoming prominence, and random-walk visitation. None directly measures expertise, trust, popularity outside the graph, or causal influence.

## Louvain community detection and projection

Louvain optimizes modularity on an undirected weighted graph. The implementation constructs an explicit projection: for an unordered pair, weights of the directed edge in either direction are summed. This avoids silently discarding direction or interaction frequency during the community step. The original directed graph remains available for display. Seed 42 is used, every user is retained, and IDs are canonicalized by sorted membership; community IDs are arbitrary labels rather than ranks. Modularity is calculated on the projection.

The observed modularity was **0.1976**, with six communities. In community-ID order, their user counts were 10, 9, 5, 2, 3, and 1, and their post totals were 60, 54, 30, 12, 18, and 6. All 180 posts matched a community; none were unmatched.

| Community ID | Users | Posts | Largest predefined topic count | Positive | Negative | Neutral | Mean compound |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 10 | 60 | education (15) | 27 | 20 | 13 | 0.0984 |
| 1 | 9 | 54 | technology (15) | 32 | 12 | 10 | 0.2332 |
| 2 | 5 | 30 | entertainment (9) | 14 | 9 | 7 | 0.1366 |
| 3 | 2 | 12 | education (4) | 7 | 4 | 1 | 0.1938 |
| 4 | 3 | 18 | sports (6) | 12 | 2 | 4 | 0.3515 |
| 5 | 1 | 6 | sports (3) | 5 | 1 | 0 | 0.4800 |

The one-user community contains the intentionally isolated account. These topic and sentiment summaries are descriptive; small groups and a single graph realization make them unstable and not generalizable.

## Dashboard functionality

The local Streamlit dashboard contains Overview, Network Explorer, Influencer Rankings, NLP Analysis, Community Analysis, User Explorer, and Downloads sections. It visualizes directed edges with community colors, keeps isolates, supports selectable centrality-based node size, exposes centrality rankings and post filters, and downloads the four existing CSV tables. Required files and columns are validated before rendering. The dashboard reads analysis artifacts without regenerating them. All panels state that the data and rankings are synthetic.

## Actual experimental results

### Sentiment and topics

| Sentiment | Posts | Proportion |
|---|---:|---:|
| Positive | 97 | 53.89% |
| Negative | 48 | 26.67% |
| Neutral | 35 | 19.44% |
| **Total** | **180** | **100%** |

Predefined topic counts were education 39, technology 37, entertainment 35, environment 35, and sports 34.

### Top users by centrality

Scores below are read from the regenerated centrality output; all 30 users were present and all centrality scores were finite.

| Metric | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|
| Degree centrality | maplebyte (0.9310) | bluejaystudio (0.8276) | cedarcloud (0.6897) |
| Closeness centrality | maplebyte (0.8448) | bluejaystudio (0.7724) | cedarcloud (0.6932) |
| Betweenness centrality | maplebyte (0.2374) | cloudpebble (0.2025) | lunarloaf (0.1702) |
| Eigenvector centrality | maplebyte (0.7051) | bluejaystudio (0.5887) | cedarcloud (0.3406) |
| PageRank | maplebyte (0.2631) | bluejaystudio (0.2549) | cedarcloud (0.1452) |

The same synthetic account leads several metrics because the dataset intentionally makes it a high-incoming-interaction hub. This is an expected property of the construction, not a discovered real-world finding.

## Testing and evaluation

The full pytest suite passed: **53 passed, 0 failed**. It includes generator schema/seed/interactions, text processing, sentiment thresholds and lexicon failure, TF-IDF, pipeline preservation, graph semantics and statistics, centrality definitions/ranks/exports, community assignments and summaries, and dashboard validation/figure construction.

An end-to-end run was performed in a disposable copy using the documented module commands. The generator output exactly matched the source sample CSV by SHA-256 comparison. NLP processing preserved 180 rows and reported the expected sentiment counts. Graph creation reported 30 users and 100 directed edges. Centrality output contained 30 users with finite metric values. Community analysis assigned all 30 users, matched all 180 posts, reported zero unmatched posts, and generated the expected outputs. Dashboard artifact loading validated 180 posts, 30 membership rows, and 100 edges. Existing project outputs were not regenerated by this check.

A separate 15-example manually labeled VADER check used five deliberately clear examples per class. It predicted all 15 labels correctly: accuracy 1.00; macro precision 1.00; macro recall 1.00; macro F1 1.00. The confusion matrix in positive/negative/neutral order was `[[5,0,0],[0,5,0],[0,0,5]]`. Because this convenience sample is tiny, intentionally obvious, and not a representative held-out sample, these metrics are only a smoke check of the installed VADER behavior. They must not be interpreted as a model-performance estimate.

### Remaining evaluation gaps

- No representative, independently labeled corpus is available; real-world precision, recall, calibration, and domain transfer remain unknown.
- The manual VADER check omits sarcasm, mixed sentiment, slang, spelling variation, emoji-only posts, multilingual text, and context-dependent meanings.
- Unit tests cover data and algorithm edge cases, but do not automate browser-level clicking through every dashboard tab or completing each CSV download. The dashboard itself was launched and its rendered overview/network explorer inspected.
- The graph and community structure are a single seeded construction; sensitivity to alternate seeds and parameters has not been measured.

## Ethical considerations and limitations

All accounts and posts are fictional. No external API, account, or real-person data is used. The dashboard and reports avoid claims of real-world influence. Centrality can encode data-generation choices and interaction volume, while community detection depends on the chosen projection, objective, resolution, and seed. VADER is lexicon/rule based and may misread context, negation, irony, reclaimed language, and domain-specific wording. TF-IDF represents term salience, not meaning. The sample size is too small for population claims or high-stakes decisions.

## Future enhancements

- Evaluate on a larger, independently labeled, consented dataset with class-specific metrics and uncertainty intervals.
- Add sensitivity analysis over generator seeds, graph projection choices, and Louvain resolution.
- Test additional real-world language phenomena using appropriately licensed and privacy-protective examples.
- Add temporal network views and compare centrality changes over time.
- Add model/data version metadata and an automated reproducibility report.
- Consider Leiden or other methods and document how their partitions differ.

## Conclusion

The project demonstrates a complete and repeatable workflow from fictional post generation through text features, directed graph metrics, community summaries, and an interactive dashboard. The verified outputs are internally consistent for 180 posts and 30 fictional users, and the 53-test suite passes. Its value is educational and methodological: the synthetic evaluation does not validate real-world sentiment accuracy or influence detection.

## References

1. NLTK Project. “`nltk.sentiment.SentimentIntensityAnalyzer`.” [NLTK API documentation](https://www.nltk.org/api/nltk.sentiment.SentimentIntensityAnalyzer.html).
2. Hutto, C. J., & Gilbert, E. (2014). “VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text.” *Proceedings of the International AAAI Conference on Web and Social Media*, 8(1), 216–225. [DOI: 10.1609/icwsm.v8i1.14550](https://doi.org/10.1609/icwsm.v8i1.14550).
3. scikit-learn developers. “`TfidfVectorizer`.” [API documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
4. NetworkX developers. “Centrality.” [NetworkX algorithm documentation](https://networkx.org/documentation/stable/reference/algorithms/centrality.html).
5. NetworkX developers. “`louvain_communities`.” [NetworkX community documentation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html).
6. Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). “Fast unfolding of communities in large networks.” *Journal of Statistical Mechanics: Theory and Experiment*, 2008(10), P10008. [DOI: 10.1088/1742-5468/2008/10/P10008](https://doi.org/10.1088/1742-5468/2008/10/P10008).
7. Streamlit developers. “`st.plotly_chart`.” [API documentation](https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart).
