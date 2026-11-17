"""Beginner-to-advanced learning paths over the prerequisite graph."""
from __future__ import annotations

import warnings

import networkx as nx
import numpy as np


def match_concept(goal: str, concepts: list[str], concept_emb: np.ndarray | None = None, embedder=None,
                  min_similarity: float = 0.45) -> str:
    g = goal.strip().lower()
    if not g:
        raise ValueError("Goal is empty.")
    if g in concepts:
        return g
    contains = [c for c in concepts if g in c or c in g]
    if contains:
        return max(contains, key=len)
    if embedder is not None and concept_emb is not None:
        q = embedder.encode([goal])[0]
        sims = concept_emb @ q
        if float(sims.max()) >= min_similarity:
            return concepts[int(np.argmax(sims))]
    words = set(g.split())
    best = max(concepts, key=lambda c: len(words & set(c.split())))
    if not words & set(best.split()):
        raise ValueError(f"No concept in this corpus matches '{goal}'. Try one from the concept list.")
    return best


def learning_path(prereq: nx.DiGraph, difficulty: dict[str, float], goal: str, max_steps: int = 12) -> list[str]:
    """All prerequisites of `goal` in topological order (ties broken by difficulty), ending with the goal."""
    if goal not in prereq:
        raise ValueError(f"Unknown concept: {goal}")
    needed = nx.ancestors(prereq, goal)
    sub = prereq.subgraph(needed | {goal})
    order = list(nx.lexicographical_topological_sort(sub, key=lambda c: (difficulty.get(c, 0), c)))
    if len(order) > max_steps:  # keep the most relevant: closest prerequisites by shortest-path distance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            dist = dict(nx.single_target_shortest_path_length(prereq, goal))
        keep = set(sorted(needed, key=lambda c: (dist.get(c, 99), -difficulty.get(c, 0)))[: max_steps - 1]) | {goal}
        order = [c for c in order if c in keep]
    return order


def full_curriculum(prereq: nx.DiGraph, difficulty: dict[str, float]) -> list[str]:
    return list(nx.lexicographical_topological_sort(prereq, key=lambda c: (difficulty.get(c, 0), c)))
