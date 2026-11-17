import os

# tests use the offline TF-IDF embedder (fast, deterministic, no model download)
os.environ.setdefault("EMBEDDING_BACKEND", "tfidf")
