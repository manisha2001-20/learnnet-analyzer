"""Predicts the level and topic cluster of a NEW transcript from its embedding.

The level labels come from the corpus difficulty analysis (difficulty.py); a classifier learns to map
embeddings to those levels so new videos can be placed without rebuilding the graph. Cross-validated
accuracy is computed at build time and reported by the API (no hard-coded numbers).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


@dataclass
class LevelPredictor:
    model: LogisticRegression | None
    classes: list[str]
    cv_accuracy: float | None
    cv_folds: int
    majority_baseline: float

    @classmethod
    def train(cls, emb: np.ndarray, levels: list[str], seed: int = 0) -> "LevelPredictor":
        y = np.array(levels)
        classes, counts = np.unique(y, return_counts=True)
        baseline = float(counts.max() / counts.sum())
        if len(classes) < 2:
            return cls(None, list(classes), None, 0, baseline)
        clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")
        folds = int(min(5, counts.min()))
        cv = None
        if folds >= 2:
            cv = float(cross_val_score(clf, emb, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed)).mean())
        clf.fit(emb, y)
        return cls(clf, list(clf.classes_), cv, folds, baseline)

    def predict(self, vec: np.ndarray) -> dict:
        if self.model is None:
            return {"level": self.classes[0] if self.classes else None, "probabilities": {}}
        p = self.model.predict_proba(vec.reshape(1, -1))[0]
        return {"level": self.classes[int(np.argmax(p))], "probabilities": {c: round(float(x), 4) for c, x in zip(self.classes, p)}}
