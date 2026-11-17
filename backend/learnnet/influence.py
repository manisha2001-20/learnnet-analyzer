"""Pedagogical influence scoring for educators.

Components (each min-max scaled to 0..1 across educators):
  coverage      share of the corpus' concepts the educator teaches
  foundational  PageRank mass of the concepts they teach (teaching central concepts matters more)
  precedence    PageRank of the educator in a "taught-it-first" network: edge B -> A when A published
                a concept before B covered it (influence flows to the earlier teacher)
  breadth       number of distinct concept communities (topic areas) covered
  reach         log(1 + total views)   [only if view counts are available]

Influence = 0.25 coverage + 0.25 foundational + 0.30 precedence + 0.10 breadth + 0.10 reach.
This is a transparent heuristic for comparing educators within one corpus, not an absolute measure.
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np

from learnnet.concepts import ConceptIndex
from learnnet.transcripts import Video

WEIGHTS = {"coverage": 0.25, "foundational": 0.25, "precedence": 0.30, "breadth": 0.10, "reach": 0.10}


def educator_influence(videos: list[Video], index: ConceptIndex, concept_graph: nx.Graph) -> tuple[list[dict], nx.DiGraph]:
    educators = sorted({v.educator for v in videos})
    taught: dict[str, set] = defaultdict(set)
    first_time: dict[tuple[str, str], str] = {}
    views = defaultdict(int)
    for v, concepts in zip(videos, index.doc_concepts):
        views[v.educator] += int(v.views or 0)
        for c in concepts:
            taught[v.educator].add(c)
            key = (v.educator, c)
            if v.published and (key not in first_time or v.published < first_time[key]):
                first_time[key] = v.published

    pr = nx.get_node_attributes(concept_graph, "pagerank")
    comm = nx.get_node_attributes(concept_graph, "community")
    n_concepts = max(1, len(index.concepts))

    net = nx.DiGraph()
    net.add_nodes_from(educators)
    for c in index.concepts:
        teachers = [(first_time[(e, c)], e) for e in educators if (e, c) in first_time]
        teachers.sort()
        for i, (t_b, b) in enumerate(teachers):
            for t_a, a in teachers[:i]:
                if t_a < t_b and a != b:
                    w = net[b][a]["weight"] + 1 if net.has_edge(b, a) else 1
                    net.add_edge(b, a, weight=w)
    precedence = nx.pagerank(net, weight="weight") if net.number_of_edges() else {e: 1 / len(educators) for e in educators}

    raw = {e: {"coverage": len(taught[e]) / n_concepts,
               "foundational": sum(pr.get(c, 0) for c in taught[e]),
               "precedence": precedence.get(e, 0.0),
               "breadth": len({comm.get(c) for c in taught[e]}),
               "reach": float(np.log1p(views[e]))} for e in educators}
    has_views = any(views.values())
    scaled = {}
    for comp in WEIGHTS:
        vals = np.array([raw[e][comp] for e in educators], float)
        s = (vals - vals.min()) / (np.ptp(vals) + 1e-12) if np.ptp(vals) > 0 else np.full(len(vals), 0.5)
        scaled[comp] = dict(zip(educators, s))
    weights = dict(WEIGHTS) if has_views else {k: v / (1 - WEIGHTS["reach"]) for k, v in WEIGHTS.items() if k != "reach"}
    rows = []
    for e in educators:
        score = sum(weights[k] * scaled[k][e] for k in weights)
        rows.append({"educator": e, "influence": round(100 * score, 1), "videos": sum(v.educator == e for v in videos),
                     "concepts_taught": len(taught[e]), "total_views": views[e],
                     "components": {k: round(float(scaled[k][e]), 3) for k in weights},
                     "raw": {k: round(float(raw[e][k]), 4) for k in raw[e]}})
    rows.sort(key=lambda r: -r["influence"])
    return rows, net
