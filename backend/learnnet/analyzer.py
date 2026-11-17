"""End-to-end analysis: transcripts -> concepts -> embeddings -> clusters -> graphs -> scores -> predictor."""
from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict

import networkx as nx
import numpy as np

from learnnet.clustering import cluster_embeddings
from learnnet.concepts import extract_concepts
from learnnet.difficulty import concept_difficulty, to_levels, video_difficulty
from learnnet.embeddings import Embedder, TfidfSvdEmbedder, make_embedder
from learnnet.graph import build_concept_graph, prerequisite_graph
from learnnet.influence import educator_influence
from learnnet.learning_path import full_curriculum, learning_path, match_concept
from learnnet.prediction import LevelPredictor
from learnnet.transcripts import Video

log = logging.getLogger(__name__)


class Analysis:
    def __init__(self, videos: list[Video], embedder: Embedder | None = None, top_k_concepts: int = 60):
        if len(videos) < 4:
            raise ValueError("Add at least 4 videos to analyse.")
        t0 = time.perf_counter()
        self.videos = videos
        texts = [v.clean_text for v in videos]
        self.embedder = embedder or make_embedder()
        if isinstance(self.embedder, TfidfSvdEmbedder) and self.embedder.svd is None:
            self.embedder.fit(texts)

        self.video_emb = self.embedder.encode(texts)
        self.index = extract_concepts(texts, top_k=top_k_concepts, embedder=self.embedder)
        self.concept_emb = self.embedder.encode(self.index.concepts)
        self.clusters = cluster_embeddings(self.video_emb)

        # each educator's videos in publishing order
        by_edu = defaultdict(list)
        for i, v in enumerate(videos):
            by_edu[v.educator].append(i)
        self.sequences = [sorted(ix, key=lambda i: (videos[i].published or "", videos[i].video_id)) for ix in by_edu.values()]

        self.concept_graph = build_concept_graph(self.index, self.concept_emb)
        self.prereq = prerequisite_graph(self.index, self.sequences, related=self.concept_graph)
        self.concept_diff = concept_difficulty(self.index, self.prereq, self.sequences)
        self.video_diff = video_difficulty(self.index, self.concept_diff, texts)
        self.video_levels = to_levels(self.video_diff)
        self.influence, self.educator_net = educator_influence(videos, self.index, self.concept_graph)
        self.predictor = LevelPredictor.train(self.video_emb, self.video_levels)
        self.cluster_names = self._name_clusters()
        self.build_seconds = round(time.perf_counter() - t0, 2)

    # ------------------------------------------------------------------ helpers
    def _name_clusters(self) -> dict[int, str]:
        names = {}
        for k in range(self.clusters.k):
            members = [i for i, l in enumerate(self.clusters.labels) if l == k]
            counts = Counter(c for i in members for c in self.index.doc_concepts[i])
            # distinctive concepts: frequent in the cluster relative to the corpus
            ranked = sorted(counts, key=lambda c: -(counts[c] / len(members)) / (self.index.doc_freq[c] / len(self.videos)) * np.log1p(counts[c]))
            names[k] = ", ".join(ranked[:3]) if ranked else f"cluster {k}"
        return names

    def _best_video_for(self, concept: str) -> dict | None:
        cands = [(i, len(d[concept])) for i, d in enumerate(self.index.doc_concepts) if concept in d]
        if not cands:
            return None
        # prefer videos that focus on the concept and are not harder than necessary
        i = max(cands, key=lambda t: (t[1], -self.video_diff[t[0]]))[0]
        v = self.videos[i]
        return {"video_id": v.video_id, "title": v.title, "educator": v.educator, "level": self.video_levels[i], "url": v.url}

    # ------------------------------------------------------------------ API views
    def summary(self) -> dict:
        p = self.predictor
        return {"videos": len(self.videos), "educators": len({v.educator for v in self.videos}),
                "concepts": len(self.index.concepts), "concept_edges": self.concept_graph.number_of_edges(),
                "prerequisite_edges": self.prereq.number_of_edges(), "clusters": self.clusters.k,
                "silhouette": round(self.clusters.silhouette, 4), "embedder": self.embedder.name,
                "level_counts": dict(Counter(self.video_levels)),
                "predictor": {"cv_accuracy": None if p.cv_accuracy is None else round(p.cv_accuracy, 4), "cv_folds": p.cv_folds,
                              "majority_baseline": round(p.majority_baseline, 4)},
                "synthetic_videos": sum(bool(v.synthetic) for v in self.videos), "build_seconds": self.build_seconds}

    def videos_view(self) -> list[dict]:
        out = []
        for i, v in enumerate(self.videos):
            top = sorted(self.index.doc_concepts[i], key=lambda c: -len(self.index.doc_concepts[i][c]))[:6]
            out.append({**v.to_dict(), "transcript": v.transcript[:400], "cluster": int(self.clusters.labels[i]),
                        "cluster_name": self.cluster_names[int(self.clusters.labels[i])], "difficulty": round(float(self.video_diff[i]), 4),
                        "level": self.video_levels[i], "top_concepts": top,
                        "x": float(self.clusters.coords_2d[i, 0]), "y": float(self.clusters.coords_2d[i, 1])})
        return out

    def concepts_view(self) -> list[dict]:
        g = self.concept_graph
        return sorted([{"concept": c, "doc_freq": self.index.doc_freq[c], "pagerank": round(g.nodes[c]["pagerank"], 5),
                        "community": g.nodes[c].get("community", 0), "difficulty": self.concept_diff[c],
                        "prerequisites": sorted(self.prereq.predecessors(c)), "unlocks": sorted(self.prereq.successors(c))}
                       for c in self.index.concepts], key=lambda r: -r["pagerank"])

    def graph_view(self, kind: str = "concept") -> dict:
        if kind == "prerequisite":
            g = self.prereq
            links = [{"source": a, "target": b, "weight": d["support"]} for a, b, d in g.edges(data=True)]
        else:
            g = self.concept_graph
            links = [{"source": a, "target": b, "weight": round(d["weight"], 3), "cooccur": d["cooccur"],
                      "similarity": round(d["similarity"], 3)} for a, b, d in g.edges(data=True)]
        cg = self.concept_graph
        nodes = [{"id": c, "pagerank": cg.nodes[c]["pagerank"], "community": cg.nodes[c].get("community", 0),
                  "difficulty": self.concept_diff[c], "doc_freq": self.index.doc_freq[c]} for c in g.nodes]
        return {"kind": kind, "nodes": nodes, "links": links}

    def clusters_view(self) -> dict:
        sizes = Counter(int(l) for l in self.clusters.labels)
        return {"k": self.clusters.k, "silhouette": self.clusters.silhouette,
                "silhouette_by_k": {str(k): round(v, 4) for k, v in self.clusters.silhouette_by_k.items()},
                "clusters": [{"id": k, "name": self.cluster_names[k], "size": sizes[k]} for k in range(self.clusters.k)]}

    def educators_view(self) -> dict:
        links = [{"source": a, "target": b, "weight": d["weight"]} for a, b, d in self.educator_net.edges(data=True)]
        return {"educators": self.influence, "precedence_links": links}

    def learning_path_view(self, goal: str | None) -> dict:
        if not goal:
            order = full_curriculum(self.prereq, self.concept_diff)
            goal_c = None
        else:
            goal_c = match_concept(goal, self.index.concepts, self.concept_emb, self.embedder)
            order = learning_path(self.prereq, self.concept_diff, goal_c)
        steps = []
        for n, c in enumerate(order, 1):
            d = self.concept_diff[c]
            level = "beginner" if d < 0.34 else "intermediate" if d < 0.67 else "advanced"
            steps.append({"step": n, "concept": c, "difficulty": d, "level": level, "video": self._best_video_for(c)})
        return {"goal": goal_c, "steps": steps}

    def predict(self, text: str) -> dict:
        if len(text.split()) < 20:
            raise ValueError("Please provide at least 20 words of transcript.")
        from learnnet.transcripts import clean_transcript

        clean = clean_transcript(text)
        vec = self.embedder.encode([clean])[0]
        sims = self.clusters.centroids @ vec / (np.linalg.norm(self.clusters.centroids, axis=1) * np.linalg.norm(vec) + 1e-12)
        cluster = int(np.argmax(sims))
        nearest = np.argsort(-(self.video_emb @ vec))[:3]
        low = clean.lower()
        found = [c for c in self.index.concepts if c in low]
        return {**self.predictor.predict(vec), "cluster": cluster, "cluster_name": self.cluster_names[cluster],
                "concepts_found": found[:15],
                "similar_videos": [{"video_id": self.videos[i].video_id, "title": self.videos[i].title,
                                    "educator": self.videos[i].educator, "similarity": round(float(self.video_emb[i] @ vec), 4)} for i in nearest],
                "predictor_cv_accuracy": self.predictor.cv_accuracy}
