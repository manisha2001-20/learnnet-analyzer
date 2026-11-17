"""Concept and video difficulty scores (heuristic, transparent, documented in docs/ARCHITECTURE.md)."""
from __future__ import annotations

import re

import networkx as nx
import numpy as np

from learnnet.concepts import ConceptIndex

LEVELS = ["beginner", "intermediate", "advanced"]


def _minmax(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (np.ptp(x) + 1e-9)


def concept_difficulty(index: ConceptIndex, prereq: nx.DiGraph, sequences: list[list[int]]) -> dict[str, float]:
    """difficulty = 0.45·prerequisite depth + 0.35·(how late it is first taught) + 0.20·rarity, each scaled to 0..1."""
    concepts = index.concepts
    depth = {}
    for c in nx.topological_sort(prereq):
        preds = list(prereq.predecessors(c))
        depth[c] = 1 + max((depth[p] for p in preds), default=-1)
    # mean relative position (0 = first video, 1 = last) at which each educator first teaches the concept
    positions: dict[str, list[float]] = {c: [] for c in concepts}
    for seq in sequences:
        seen = set()
        for rank, vid in enumerate(seq):
            for c in index.doc_concepts[vid]:
                if c not in seen:
                    seen.add(c)
                    positions[c].append(rank / max(1, len(seq) - 1))
    d = np.array([depth.get(c, 0) for c in concepts], float)
    late = np.array([np.mean(positions[c]) if positions[c] else 1.0 for c in concepts], float)
    rare = np.array([1.0 / index.doc_freq[c] for c in concepts])
    score = 0.45 * _minmax(d) + 0.35 * _minmax(late) + 0.20 * _minmax(rare)
    return {c: float(round(s, 4)) for c, s in zip(concepts, score)}


def readability(text: str) -> float:
    """Average words per sentence × average characters per word, scaled roughly to 0..1."""
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"[A-Za-z]+", text)
    if not sents or not words:
        return 0.0
    wps = len(words) / len(sents)
    cpw = sum(map(len, words)) / len(words)
    return float(min(1.0, (wps / 25) * 0.5 + ((cpw - 3.5) / 4) * 0.5))


def video_difficulty(index: ConceptIndex, concept_diff: dict[str, float], texts: list[str]) -> np.ndarray:
    out = []
    for d, text in zip(index.doc_concepts, texts):
        cd = [concept_diff[c] for c in d] or [0.0]
        out.append(0.8 * float(np.mean(sorted(cd)[-5:])) + 0.2 * readability(text))  # hardest concepts dominate
    return np.array(out)


def to_levels(scores: np.ndarray) -> list[str]:
    q1, q2 = np.quantile(scores, [1 / 3, 2 / 3])
    return [LEVELS[0] if s <= q1 else LEVELS[1] if s <= q2 else LEVELS[2] for s in scores]
