"""Concept relationship graph (NetworkX): co-occurrence + semantic similarity + prerequisite direction."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np

from learnnet.concepts import ConceptIndex


def build_concept_graph(index: ConceptIndex, concept_embeddings: np.ndarray | None = None,
                        min_cooccur: int = 2, sim_threshold: float = 0.6) -> nx.Graph:
    g = nx.Graph()
    for c in index.concepts:
        g.add_node(c, doc_freq=index.doc_freq[c], score=index.scores[c])
    co = defaultdict(int)
    for d in index.doc_concepts:
        for a, b in combinations(sorted(d), 2):
            co[(a, b)] += 1
    for (a, b), w in co.items():
        if w >= min_cooccur:
            g.add_edge(a, b, cooccur=w, similarity=0.0, weight=float(w))
    if concept_embeddings is not None and len(index.concepts) > 1:
        sims = concept_embeddings @ concept_embeddings.T
        for i, j in combinations(range(len(index.concepts)), 2):
            if sims[i, j] >= sim_threshold:
                a, b = index.concepts[i], index.concepts[j]
                if g.has_edge(a, b):
                    g[a][b]["similarity"] = float(sims[i, j])
                    g[a][b]["weight"] += float(sims[i, j])
                else:
                    g.add_edge(a, b, cooccur=0, similarity=float(sims[i, j]), weight=float(sims[i, j]))
    pr = nx.pagerank(g, weight="weight") if g.number_of_edges() else {n: 1 / max(1, len(g)) for n in g}
    nx.set_node_attributes(g, pr, "pagerank")
    try:
        comms = nx.algorithms.community.greedy_modularity_communities(g, weight="weight") if g.number_of_edges() else [set(g)]
    except ZeroDivisionError:
        comms = [set(g)]
    for k, comm in enumerate(comms):
        for n in comm:
            g.nodes[n]["community"] = k
    return g


def home_videos(index: ConceptIndex, top_n: int = 4) -> dict[str, set[int]]:
    """Videos that are *about* a concept: it is among the video's `top_n` most-mentioned concepts."""
    homes: dict[str, set[int]] = defaultdict(set)
    for i, d in enumerate(index.doc_concepts):
        for c in sorted(d, key=lambda c: -len(d[c]))[:top_n]:
            homes[c].add(i)
    return homes


def prerequisite_graph(index: ConceptIndex, sequences: list[list[int]], related: nx.Graph | None = None,
                       threshold: float = 0.25) -> nx.DiGraph:
    """A -> B ("A is a prerequisite of B") using reference distance (RefD, Liang et al., EMNLP 2015).

    RefD(A, B) = P(video mentions A | video is about B) - P(video mentions B | video is about A).
    If lessons about B keep referring to A but lessons about A rarely mention B, A is a prerequisite
    of B. A small bonus comes from each educator's publishing order (A taught in an earlier video).
    Edges require RefD + bonus >= `threshold` and the concepts to be related in `related`.
    Cycles are broken by removing the weakest edge; a transitive reduction keeps the graph readable.
    """
    homes = home_videos(index)
    mentions = [set(d) for d in index.doc_concepts]

    def ref(a: str, b: str) -> float:  # how often lessons about b mention a
        hb = homes.get(b, set())
        return sum(a in mentions[i] for i in hb) / len(hb) if hb else 0.0

    order_votes = defaultdict(int)
    for seq in sequences:
        first_video = {}
        for rank, vid in enumerate(seq):
            for c in index.doc_concepts[vid]:
                first_video.setdefault(c, rank)
        for a, b in combinations(first_video, 2):
            if first_video[a] < first_video[b]:
                order_votes[(a, b)] += 1
            elif first_video[b] < first_video[a]:
                order_votes[(b, a)] += 1
    n_seq = max(1, len(sequences))

    dg = nx.DiGraph()
    dg.add_nodes_from(index.concepts)
    for a, b in combinations(index.concepts, 2):
        if related is not None and not related.has_edge(a, b):
            continue
        refd = ref(a, b) - ref(b, a)
        bonus = 0.1 * (order_votes.get((a, b), 0) - order_votes.get((b, a), 0)) / n_seq
        score = refd + bonus
        if score >= threshold:
            dg.add_edge(a, b, support=round(score, 3))
        elif -score >= threshold:
            dg.add_edge(b, a, support=round(-score, 3))
    while True:
        try:
            cycle = nx.find_cycle(dg)
        except nx.NetworkXNoCycle:
            break
        weakest = min(cycle, key=lambda e: dg.edges[e[0], e[1]]["support"])
        dg.remove_edge(weakest[0], weakest[1])
    reduced = nx.transitive_reduction(dg)
    reduced.add_nodes_from(dg.nodes(data=True))
    for a, b in reduced.edges:
        reduced[a][b]["support"] = dg[a][b]["support"]
    return reduced
