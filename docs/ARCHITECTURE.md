# Architecture — LearnNet Analyzer

```
                    ┌──────────── React + Vite dashboard (Recharts, react-force-graph) ────────────┐
                    │ Overview · Concept graph · Clusters · Learning path · Educators · Predict · Add │
                    └───────────────────────────────┬────────────────────────────────────────────────┘
                                                    │ JSON over /api
                                         ┌──────────▼──────────┐
                                         │  FastAPI (api/main) │  rebuild lock, validation, error codes
                                         └──────────┬──────────┘
                                                    ▼
 transcripts.json / YouTube captions ──► clean ──► Analysis (learnnet/analyzer.py)
                                                    │
     ┌──────────────┬──────────────┬───────────────┼────────────────┬──────────────┬──────────────┐
     ▼              ▼              ▼               ▼                ▼              ▼              ▼
 embeddings     concepts       clustering     concept graph    prerequisite    difficulty     influence
 (SBERT |       (key-phrases,  (K-means,      (co-occurrence   DAG (RefD +     & levels       (precedence
  TF-IDF/SVD)   TF-IDF +       silhouette,     + similarity,   teaching order) (depth, order,  PageRank …)
                semantic)      PCA)            PageRank)                       rarity)
                                                                    │
                                                                    ▼
                                                   learning paths · level predictor
```

## Pipeline steps

1. **Cleaning:** removes `[Music]`-style tags, timestamps and filler words, and normalises whitespace.
2. **Embeddings:** `all-MiniLM-L6-v2` (384-d, normalised). If the model can't load, falls back to TF-IDF (1–2 grams) + TruncatedSVD (LSA).
3. **Concepts:** candidates are 1–3-word phrases inside one sentence, containing no stopwords, generic words or verbs. Plurals are normalised. A phrase must appear in at least 2 videos and at most 80% of them. Scoring is `max TF-IDF + 0.15·mean TF-IDF`, × (1 + 0.15 per extra word), × semantic closeness to the corpus centroid. Overlapping phrases with similar document frequency are removed.
4. **Clustering:** K-means for k = 2…10. Picks the k with the best cosine silhouette. Clusters are named by their most distinctive concepts (frequency in the cluster ÷ frequency in the corpus).
5. **Concept graph:** an edge is added when two concepts co-occur in at least 2 videos, or when their embedding cosine similarity is at least 0.6. Includes PageRank and greedy-modularity communities.
6. **Prerequisite DAG (RefD):**
   `RefD(A,B) = P(mentions A | video about B) − P(mentions B | video about A)`.
   A video is "about" a concept if it is among the video's 4 most-mentioned concepts. A bonus of `0.1 × (net order votes)/#educators` comes from each educator's publishing order. An edge A→B is added if the score is at least 0.25 and A–B are related in the concept graph. Cycles are broken by removing the weakest edge, then a transitive reduction is applied.
7. **Difficulty:** `0.45·prerequisite depth + 0.35·mean relative position where educators first teach it + 0.20·rarity`, each scaled to 0–1. Video difficulty is `0.8 × mean of its 5 hardest concepts + 0.2 × readability`. Levels are corpus tertiles.
8. **Learning path:** ancestors of the goal in the DAG, sorted topologically (ties broken by difficulty), limited to 12 steps (closest prerequisites first). Each step links to the video that mentions the concept most (ties broken by easier video).
9. **Influence:** see the formula in `learnnet/influence.py`. The *precedence network* has an edge B→A whenever educator A published a concept before educator B covered it. PageRank on this network rewards educators whose teaching came first.
10. **Prediction:** logistic regression (balanced classes) on embeddings → level. Stratified k-fold CV accuracy and the majority baseline are computed at build time. The cluster is the nearest centroid, and similar videos are found by cosine similarity.

## Design decisions

| Decision | Reason |
|---|---|
| In-memory analysis, rebuilt on change | The corpus is small (tens to hundreds of videos). A rebuild takes about 2–6 s on CPU, so no database is needed. User additions persist as JSON. |
| Fallback embedder | The app and tests work offline and on machines without PyTorch. |
| RefD for prerequisites | An established, explainable method. Simple "first mention" order turned out to be inverted, because lessons name their own topic first and then recap. |
| One FastAPI process serving the built React app | One command to run; the dev mode still has hot reload via the Vite proxy. |
| Hash-based routing in React | No router dependency, and deep links still work (e.g. `/#/path`). |

## Error handling

- Invalid transcript records (missing fields, under 20 words) raise `TranscriptError`, returned as HTTP 422.
- YouTube fetch failures (no captions, invalid URL) return 422 with the reason. Duplicate videos return 409.
- An unknown learning-path goal returns 404 with a hint. Analysis that isn't ready returns 503.
- The frontend shows loading, error and retry states on every page, and a clear message when the API is unreachable.
