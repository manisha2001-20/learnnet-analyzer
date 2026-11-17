"""Concept extraction: candidate key-phrases ranked by corpus TF-IDF, optionally re-ranked semantically."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

EXTRA_STOP = {"video", "today", "lesson", "welcome", "remember", "covered", "start", "look", "quick", "summary", "next",
              "time", "let", "know", "questions", "exercise", "description", "subscribe", "series", "part", "code",
              "coffee", "notebook", "linked", "happy", "coding", "simple", "words", "hello", "hey", "everyone", "back",
              "really", "lot", "way", "thing", "things", "good", "new", "make", "makes", "use", "used", "using", "just",
              "called", "called", "tells", "help", "helps", "important", "common", "usually", "often", "like", "dojo",
              "meera", "lab", "speaking", "mathematically", "basically", "okay", "alright", "practise", "yourself",
              "build", "result", "grab", "coding", "today", "number", "numbers", "data science", "case", "set"}
STOP = set(ENGLISH_STOP_WORDS) | EXTRA_STOP
_WORD = re.compile(r"[a-z][a-z0-9\-]+")


def normalise(term: str) -> str:
    words = term.split()
    last = words[-1]
    if len(last) > 4 and last.endswith("ies"):
        last = last[:-3] + "y"
    elif len(last) > 3 and last.endswith("s") and not last.endswith(("ss", "us", "is", "sis")):
        last = last[:-1]
    return " ".join(words[:-1] + [last])


GENERIC = {"model", "value", "values", "random", "compute", "computes", "positive", "event", "store", "plot", "linear",
           "decision", "data", "set", "step", "line", "function", "learn", "learning", "train", "training", "number",
           "variable", "feature", "features", "result", "error", "group", "groups", "table", "type", "item", "items",
           "order", "point", "points", "pattern", "patterns", "class", "label", "weight", "weights", "layer", "layers",
           "input", "output", "target", "test", "tool", "tools", "average", "simple", "large", "small", "fast", "deep",
           "basic", "basics", "single", "many", "every", "first", "best", "sorted", "middle", "long", "matter", "matters",
           "task", "tasks", "true", "pass", "image", "images", "performance", "outcome", "outcomes", "program",
           "statement", "operation", "operations", "analysis", "shape", "squared", "descriptive", "given", "based",
           "express", "uncertainty", "summarise", "describe", "store", "stores", "string", "code", "block", "word",
           "words", "text", "time", "information", "direction", "size", "scale", "power", "approach"}
VERBS = {"let", "lets", "map", "maps", "assume", "assumes", "designed", "occur", "occurs", "far", "away", "train",
         "trains", "shows", "show", "find", "finds", "gives", "give", "take", "takes", "called", "multiply", "apply",
         "applies", "need", "needs", "works", "work", "learns", "predicts", "predict", "measures", "measure", "squashes",
         "fits", "fit", "builds", "split", "splits", "reduces", "reduce", "keeps", "keep", "detect", "detects", "moves",
         "move", "points", "slides", "replaced", "trained", "looks", "decide", "computes", "made", "combine", "lies",
         "extend", "extends", "roughly", "straight", "run", "quickly", "directly", "exactly", "smoothly", "understand",
         "update", "explain", "explains", "push", "pull", "pulls", "sits", "change", "changes", "turn", "turns"}
_SENT_SPLIT = re.compile(r"[.!?;:]+\s*")


def candidate_phrases(text: str, max_n: int = 3) -> list[tuple[str, int]]:
    """Return (phrase, word_position) for 1-3 word phrases with no stopwords, not crossing sentence boundaries."""
    out = []
    offset = 0
    for sentence in _SENT_SPLIT.split(text.lower()):
        words = _WORD.findall(sentence)
        for n in range(1, max_n + 1):
            for i in range(len(words) - n + 1):
                gram = words[i:i + n]
                if any(w in STOP or len(w) < 3 for w in gram):
                    continue
                if n == 1 and (gram[0] in GENERIC or len(gram[0]) < 4):
                    continue
                if gram[0] in GENERIC and gram[-1] in GENERIC:
                    continue
                if any(w in VERBS for w in gram):
                    continue
                out.append((normalise(" ".join(gram)), offset + i))
        offset += len(words)
    return out


@dataclass
class ConceptIndex:
    concepts: list[str]
    doc_concepts: list[dict[str, list[float]]]   # per video: concept -> relative positions (0..1) of mentions
    doc_freq: dict[str, int]
    scores: dict[str, float]


def extract_concepts(texts: list[str], top_k: int = 60, min_df: int = 2, embedder=None) -> ConceptIndex:
    """Pick the corpus' top_k concepts and record where each appears in each video."""
    if len(texts) < 2:
        raise ValueError("Need at least two transcripts to extract shared concepts.")
    per_doc: list[dict[str, list[float]]] = []
    df_counter: Counter = Counter()
    for text in texts:
        cands = candidate_phrases(text)
        n_words = max(1, len(_WORD.findall(text.lower())))  # same tokenisation as candidate_phrases
        pos: dict[str, list[float]] = defaultdict(list)
        for phrase, i in cands:
            pos[phrase].append(i / n_words)
        per_doc.append(pos)
        df_counter.update(pos.keys())

    vocab = [p for p, c in df_counter.items() if c >= min_df and c <= 0.8 * len(texts)]
    if not vocab:
        raise ValueError("No concepts found; transcripts may be too short or too different.")
    docs_as_terms = [" ".join(t.replace(" ", "_") for t, ps in d.items() for _ in ps) for d in per_doc]
    tfidf = TfidfVectorizer(vocabulary=[v.replace(" ", "_") for v in vocab], token_pattern=r"[^ ]+", sublinear_tf=True)
    m = tfidf.fit_transform(docs_as_terms)
    base = np.asarray(m.max(axis=0).todense()).ravel() + 0.15 * np.asarray(m.mean(axis=0)).ravel()
    # prefer multi-word technical phrases slightly
    bonus = np.array([1.0 + 0.15 * (len(v.split()) - 1) for v in vocab])
    score = base * bonus
    if embedder is not None:  # semantic re-rank: closeness to the corpus centroid (KeyBERT-style)
        try:
            emb = embedder.encode(vocab)
            centroid = embedder.encode([" ".join(texts)[:20000]])[0]
            sim = emb @ centroid
            score = score * (0.6 + 0.4 * (sim - sim.min()) / (np.ptp(sim) + 1e-9))
        except Exception:
            pass
    order = np.argsort(-score)
    chosen: list[str] = []
    for idx in order:
        c = vocab[idx]
        # drop a phrase if it is contained in an already chosen longer/shorter phrase with similar frequency
        cw = set(c.split())
        if any(((c in k or k in c) or len(cw & set(k.split())) >= 2) and abs(df_counter[c] - df_counter[k]) <= 1 for k in chosen):
            continue
        chosen.append(c)
        if len(chosen) >= top_k:
            break
    scores = {vocab[i]: float(score[i]) for i in order if vocab[i] in chosen}
    doc_concepts = [{c: ps for c, ps in d.items() if c in scores} for d in per_doc]
    return ConceptIndex(chosen, doc_concepts, {c: df_counter[c] for c in chosen}, scores)
