"""Transcript loading, YouTube fetching and cleaning."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_CORPUS = DATA_DIR / "sample_corpus.json"

_NOISE = re.compile(r"\[(?:music|applause|laughter|inaudible|__)\]|\(\s*(?:music|applause)\s*\)", re.I)
_TIMESTAMP = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_FILLERS = re.compile(r"\b(?:um+|uh+|you know|like,|i mean|basically|okay so|alright,? so)\b[,]?\s*", re.I)
_WS = re.compile(r"\s+")
_YT_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})")


class TranscriptError(ValueError):
    pass


@dataclass
class Video:
    video_id: str
    educator: str
    title: str
    transcript: str
    published: str = ""
    views: int = 0
    url: str = ""
    synthetic: bool = False
    clean_text: str = field(default="", repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("clean_text")
        return d


def clean_transcript(text: str) -> str:
    text = _NOISE.sub(" ", text or "")
    text = _TIMESTAMP.sub(" ", text)
    text = _FILLERS.sub("", text)
    text = _WS.sub(" ", text).strip()
    return text


def load_corpus(path: str | Path | None = None) -> list[Video]:
    p = Path(path) if path else SAMPLE_CORPUS
    if not p.exists():
        raise TranscriptError(f"Corpus not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TranscriptError(f"Corpus is not valid JSON: {exc}") from exc
    return [make_video(r) for r in raw]


def make_video(r: dict) -> Video:
    missing = {"video_id", "educator", "title", "transcript"} - set(r)
    if missing:
        raise TranscriptError(f"Video record missing fields: {', '.join(sorted(missing))}")
    if len(str(r["transcript"]).split()) < 20:
        raise TranscriptError(f"Transcript for {r['video_id']} is too short (need at least 20 words).")
    v = Video(**{k: r[k] for k in Video.__dataclass_fields__ if k in r and k != "clean_text"})
    v.clean_text = clean_transcript(v.transcript)
    return v


def extract_video_id(url_or_id: str) -> str:
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    m = _YT_ID.search(s)
    if not m:
        raise TranscriptError("Could not find a YouTube video id in that URL.")
    return m.group(1)


def fetch_youtube_transcript(url_or_id: str, languages: tuple[str, ...] = ("en", "en-US", "en-GB")) -> str:
    """Download captions with youtube-transcript-api. Requires internet; not all videos have transcripts."""
    vid = extract_video_id(url_or_id)
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:  # pragma: no cover
        raise TranscriptError("youtube-transcript-api is not installed") from exc
    try:
        if hasattr(YouTubeTranscriptApi, "get_transcript"):  # youtube-transcript-api < 1.0
            parts = [p["text"] for p in YouTubeTranscriptApi.get_transcript(vid, languages=list(languages))]
        else:                                                # >= 1.0: instance API returning snippets
            parts = [snippet.text for snippet in YouTubeTranscriptApi().fetch(vid, languages=list(languages))]
    except Exception as exc:  # the library raises many specific exception types
        raise TranscriptError(f"No transcript available for {vid}: {type(exc).__name__}") from exc
    if not parts:
        raise TranscriptError(f"Transcript for {vid} is empty.")
    return " ".join(parts)
