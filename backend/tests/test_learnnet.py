import json

import networkx as nx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from learnnet.analyzer import Analysis
from learnnet.clustering import cluster_embeddings
from learnnet.concepts import candidate_phrases, extract_concepts, normalise
from learnnet.embeddings import TfidfSvdEmbedder
from learnnet.learning_path import learning_path, match_concept
from learnnet.transcripts import (SAMPLE_CORPUS, TranscriptError, clean_transcript, extract_video_id, load_corpus,
                                  make_video)


@pytest.fixture(scope="module")
def videos():
    return load_corpus()


@pytest.fixture(scope="module")
def analysis(videos):
    return Analysis(videos, embedder=TfidfSvdEmbedder())  # offline + deterministic


# ---------------------------------------------------------------- transcripts
def test_clean_transcript_removes_noise():
    raw = "[Music] 00:12 um so today uh we learn [Applause] gradient   descent"
    assert clean_transcript(raw) == "so today we learn gradient descent"


@pytest.mark.parametrize("url", ["https://www.youtube.com/watch?v=dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ",
                                 "dQw4w9WgXcQ", "https://www.youtube.com/shorts/dQw4w9WgXcQ"])
def test_extract_video_id(url):
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_extract_video_id_rejects_garbage():
    with pytest.raises(TranscriptError):
        extract_video_id("not a url")


def test_make_video_validation():
    with pytest.raises(TranscriptError, match="missing"):
        make_video({"video_id": "x"})
    with pytest.raises(TranscriptError, match="too short"):
        make_video({"video_id": "x", "educator": "e", "title": "t", "transcript": "too short"})


def test_sample_corpus_is_marked_synthetic(videos):
    assert len(videos) == 43 and all(v.synthetic for v in videos)


# ---------------------------------------------------------------- concepts
def test_normalise_plurals():
    assert normalise("neural networks") == "neural network"
    assert normalise("probabilities") == "probability"
    assert normalise("analysis") == "analysis"


def test_candidate_phrases_skip_stopwords_and_sentence_breaks():
    phrases = {p for p, _ in candidate_phrases("The gradient descent works. Python lists are useful.")}
    assert "gradient descent" in phrases
    assert not any("descent python" in p for p in phrases)
    assert not any(p.startswith("the ") for p in phrases)


def test_extract_concepts_requires_two_docs():
    with pytest.raises(ValueError):
        extract_concepts(["only one document here"])


def test_key_concepts_found(analysis):
    concepts = set(analysis.index.concepts)
    for expected in ["gradient descent", "neural network", "logistic regression", "random forest", "backpropagation"]:
        assert expected in concepts, expected


# ---------------------------------------------------------------- graph / paths
def test_prerequisite_graph_is_dag(analysis):
    assert nx.is_directed_acyclic_graph(analysis.prereq)
    assert analysis.prereq.number_of_edges() > 10


def test_learning_path_ends_with_goal_and_respects_order(analysis):
    path = learning_path(analysis.prereq, analysis.concept_diff, "backpropagation")
    assert path[-1] == "backpropagation"
    pos = {c: i for i, c in enumerate(path)}
    for a, b in analysis.prereq.subgraph(path).edges:
        assert pos[a] < pos[b]


def test_neural_network_precedes_backprop(analysis):
    assert nx.has_path(analysis.prereq, "neural network", "backpropagation")


def test_match_concept():
    cs = ["gradient descent", "neural network", "k-mean"]
    assert match_concept("Neural Network", cs) == "neural network"
    assert match_concept("gradient", cs) == "gradient descent"
    with pytest.raises(ValueError):
        match_concept("astrophysics", cs)
    with pytest.raises(ValueError):
        match_concept("  ", cs)


def test_clustering_selects_k_by_silhouette():
    rng = np.random.default_rng(0)
    centers = np.eye(3) * 5
    pts = np.vstack([c + rng.normal(0, 0.1, (10, 3)) for c in centers])
    res = cluster_embeddings(pts, k_max=6)
    assert res.k == 3 and res.coords_2d.shape == (30, 2)
    with pytest.raises(ValueError):
        cluster_embeddings(pts[:2])


# ---------------------------------------------------------------- scores
def test_levels_balanced_and_difficulty_ordering(analysis):
    levels = analysis.video_levels
    assert set(levels) == {"beginner", "intermediate", "advanced"}
    d = analysis.concept_diff
    assert d["python basic"] < d["backpropagation"]
    assert d["numpy"] < d["convolutional neural network"]


def test_influence_scores(analysis):
    rows = analysis.educators_view()["educators"]
    assert len(rows) == 5
    assert all(0 <= r["influence"] <= 100 for r in rows)
    assert rows == sorted(rows, key=lambda r: -r["influence"])
    assert set(rows[0]["components"]) == {"coverage", "foundational", "precedence", "breadth", "reach"}


def test_predict_new_transcript(analysis):
    text = ("In this lesson we study convolutional neural networks for images. A convolution slides a filter over the image "
            "and pooling layers shrink the feature maps, and the network is trained with backpropagation.")
    out = analysis.predict(text)
    assert out["level"] in {"beginner", "intermediate", "advanced"}
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-6
    assert "Convolutional" in out["similar_videos"][0]["title"]
    with pytest.raises(ValueError):
        analysis.predict("too short")


def test_predictor_beats_majority_baseline(analysis):
    p = analysis.predictor
    assert p.cv_accuracy is not None and p.cv_accuracy > p.majority_baseline


def test_analysis_needs_enough_videos(videos):
    with pytest.raises(ValueError):
        Analysis(videos[:3], embedder=TfidfSvdEmbedder())


# ---------------------------------------------------------------- API
@pytest.fixture()
def client(tmp_path):
    app = create_app(corpus_path=SAMPLE_CORPUS, user_path=tmp_path / "user.json", embedder_backend="tfidf")
    return TestClient(app)


def test_api_read_endpoints(client):
    assert client.get("/api/health").json()["status"] == "ok"
    s = client.get("/api/summary").json()
    assert s["videos"] == 43 and s["embedder"] == "tfidf-svd"
    assert len(client.get("/api/videos").json()) == 43
    assert len(client.get("/api/concepts").json()) == 60
    g = client.get("/api/graph?kind=prerequisite").json()
    assert g["links"] and g["nodes"]
    assert client.get("/api/graph?kind=bad").status_code == 422
    assert client.get("/api/clusters").json()["k"] >= 2
    assert len(client.get("/api/educators").json()["educators"]) == 5


def test_api_learning_path(client):
    p = client.get("/api/learning-path", params={"goal": "backpropagation"}).json()
    assert p["goal"] == "backpropagation" and p["steps"][-1]["concept"] == "backpropagation"
    assert p["steps"][-1]["video"]["title"]
    assert client.get("/api/learning-path", params={"goal": "astrophysics"}).status_code == 404
    assert len(client.get("/api/learning-path").json()["steps"]) == 60


def test_api_predict_validation(client):
    assert client.post("/api/predict", json={"text": "short"}).status_code == 422
    ok = client.post("/api/predict", json={"text": "gradient descent " * 15})
    assert ok.status_code == 200 and "level" in ok.json()


def test_api_add_video_and_duplicates(client, tmp_path):
    body = {"educator": "New Teacher", "title": "Gradient descent intuition",
            "transcript": "Gradient descent is an optimization algorithm. The learning rate controls the step size. " * 4}
    r = client.post("/api/videos", json=body)
    assert r.status_code == 201 and r.json()["summary"]["videos"] == 44
    saved = json.loads((tmp_path / "user.json").read_text())
    assert saved[0]["educator"] == "New Teacher" and saved[0]["synthetic"] is False
    assert client.post("/api/videos", json={"educator": "x", "title": "y"}).status_code == 422
    assert client.post("/api/videos", json={**body, "transcript": "too short"}).status_code == 422
