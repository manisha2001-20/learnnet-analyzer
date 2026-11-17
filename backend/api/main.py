"""FastAPI backend for LearnNet Analyzer.

Run (from backend/):  uvicorn api.main:app --reload --port 8000
If frontend/dist exists (npm run build), the React dashboard is served at "/".
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from learnnet.analyzer import Analysis
from learnnet.embeddings import make_embedder
from learnnet.transcripts import DATA_DIR, TranscriptError, Video, fetch_youtube_transcript, load_corpus, make_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("learnnet.api")
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, description="Transcript text (at least 20 words)")


class VideoIn(BaseModel):
    educator: str = Field(..., min_length=1, max_length=120)
    title: str = Field(..., min_length=1, max_length=300)
    transcript: str | None = Field(None, description="Transcript text. Leave empty to fetch from YouTube.")
    youtube_url: str | None = None
    published: str | None = None
    views: int = 0


class State:
    def __init__(self, corpus_path: Path | None, user_path: Path, embedder_backend: str | None):
        self.corpus_path = corpus_path
        self.user_path = user_path
        self.embedder = make_embedder(embedder_backend)
        self.lock = threading.Lock()
        self.analysis: Analysis | None = None
        self.error: str | None = None

    def videos(self) -> list[Video]:
        vids = load_corpus(self.corpus_path)
        if self.user_path.exists():
            vids += [make_video(r) for r in json.loads(self.user_path.read_text(encoding="utf-8"))]
        return vids

    def rebuild(self) -> None:
        with self.lock:
            emb = self.embedder
            if hasattr(emb, "svd"):  # TF-IDF fallback must be refitted on the new corpus
                emb = make_embedder("tfidf")
            self.analysis = Analysis(self.videos(), embedder=emb)
            self.error = None
            log.info("analysis rebuilt: %s", self.analysis.summary())


def create_app(corpus_path: Path | None = None, user_path: Path | None = None, embedder_backend: str | None = None) -> FastAPI:
    state = State(corpus_path, user_path or DATA_DIR / "user_videos.json", embedder_backend)
    try:
        state.rebuild()
    except (TranscriptError, ValueError) as exc:
        state.error = str(exc)
        log.error("initial analysis failed: %s", exc)

    app = FastAPI(title="LearnNet Analyzer API", version="1.0.0",
                  description="YouTube educator network study: concepts, clusters, learning paths, influence and prediction.")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])
    app.state.learnnet = state

    def A() -> Analysis:
        if state.analysis is None:
            raise HTTPException(503, f"Analysis not available: {state.error}")
        return state.analysis

    @app.get("/api/health")
    def health():
        return {"status": "ok" if state.analysis else "degraded", "error": state.error,
                "embedder": state.analysis.embedder.name if state.analysis else None}

    @app.get("/api/summary")
    def summary():
        return A().summary()

    @app.get("/api/videos")
    def videos():
        return A().videos_view()

    @app.get("/api/concepts")
    def concepts():
        return A().concepts_view()

    @app.get("/api/graph")
    def graph(kind: str = "concept"):
        if kind not in ("concept", "prerequisite"):
            raise HTTPException(422, "kind must be 'concept' or 'prerequisite'")
        return A().graph_view(kind)

    @app.get("/api/clusters")
    def clusters():
        return A().clusters_view()

    @app.get("/api/educators")
    def educators():
        return A().educators_view()

    @app.get("/api/learning-path")
    def path(goal: str | None = None):
        try:
            return A().learning_path_view(goal)
        except ValueError as exc:
            raise HTTPException(404, str(exc))

    @app.post("/api/predict")
    def predict(body: PredictIn):
        try:
            return A().predict(body.text)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.post("/api/videos", status_code=201)
    def add_video(body: VideoIn):
        text = (body.transcript or "").strip()
        vid_id = None
        if not text:
            if not body.youtube_url:
                raise HTTPException(422, "Provide a transcript or a YouTube URL.")
            try:
                text = fetch_youtube_transcript(body.youtube_url)
            except TranscriptError as exc:
                raise HTTPException(422, str(exc))
        if body.youtube_url:
            from learnnet.transcripts import extract_video_id
            try:
                vid_id = extract_video_id(body.youtube_url)
            except TranscriptError:
                vid_id = None
        existing = json.loads(state.user_path.read_text(encoding="utf-8")) if state.user_path.exists() else []
        record = {"video_id": vid_id or f"user-{len(existing) + 1:03d}", "educator": body.educator.strip(),
                  "title": body.title.strip(), "transcript": text, "published": body.published or date.today().isoformat(),
                  "views": body.views, "url": body.youtube_url or "", "synthetic": False}
        try:
            make_video(record)  # validates
        except TranscriptError as exc:
            raise HTTPException(422, str(exc))
        if any(r["video_id"] == record["video_id"] for r in existing):
            raise HTTPException(409, "This video was already added.")
        state.user_path.parent.mkdir(parents=True, exist_ok=True)
        state.user_path.write_text(json.dumps(existing + [record], indent=1), encoding="utf-8")
        try:
            state.rebuild()
        except ValueError as exc:
            raise HTTPException(422, f"Video saved but analysis failed: {exc}")
        return {"added": record["video_id"], "summary": state.analysis.summary()}

    @app.post("/api/rebuild")
    def rebuild():
        try:
            state.rebuild()
        except (TranscriptError, ValueError) as exc:
            raise HTTPException(422, str(exc))
        return state.analysis.summary()

    if FRONTEND_DIST.exists():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str):
            target = FRONTEND_DIST / full_path
            if full_path and target.is_file():
                return FileResponse(target)
            return FileResponse(FRONTEND_DIST / "index.html")

    return app


app = create_app()
