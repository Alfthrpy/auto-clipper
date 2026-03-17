"""
Scorer — rank chunks by highlight potential.

Supports three modes:
  - "tfidf"  → text-only TF-IDF uniqueness scoring (MVP)
  - "audio"  → audio-only energy/pitch scoring
  - "fused"  → weighted combination of text + audio (recommended)
"""
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import Config
from pipeline.segmenter import Chunk


@dataclass
class ScoredChunk:
    """A chunk with its computed highlight score."""
    chunk: Chunk
    score: float


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _score_tfidf(chunks: list[Chunk]) -> np.ndarray:
    """
    TF-IDF text scoring.
    Uniqueness (inverse centrality) + speech density.
    """
    texts = [c.text for c in chunks]

    vectorizer = TfidfVectorizer(max_features=5000, stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Uniqueness: distance from the mean content
    mean_vector = np.asarray(tfidf_matrix.mean(axis=0))
    similarities = cosine_similarity(tfidf_matrix, mean_vector).flatten()
    uniqueness = 1.0 - similarities

    # Speech density: words per second
    densities = np.array([
        len(c.text.split()) / max(c.duration, 1.0) for c in chunks
    ])

    uniqueness_norm = _normalize(uniqueness)
    density_norm = _normalize(densities)

    return 0.7 * uniqueness_norm + 0.3 * density_norm


def score_chunks(
    chunks: list[Chunk],
    config: Config,
    audio_path: str | None = None,
) -> list[ScoredChunk]:
    """
    Score chunks using the configured scoring mode.

    Args:
        chunks:     List of text chunks to score.
        config:     Pipeline configuration (uses scoring_mode, audio_weight).
        audio_path: Path to WAV file (required for 'audio' and 'fused' modes).

    Returns:
        List of ScoredChunk sorted by score descending.
    """
    if not chunks:
        return []

    mode = config.scoring_mode
    scores: np.ndarray

    if mode == "tfidf":
        print("[score] Mode: tfidf (text-only)")
        scores = _score_tfidf(chunks)

    elif mode == "audio":
        print("[score] Mode: audio (energy/pitch only)")
        if not audio_path:
            raise ValueError("audio_path is required for 'audio' scoring mode")
        from pipeline.audio_scorer import score_audio
        scores = score_audio(audio_path, chunks)

    elif mode == "fused":
        print(f"[score] Mode: fused (text={1 - config.audio_weight:.0%} + "
              f"audio={config.audio_weight:.0%})")
        if not audio_path:
            raise ValueError("audio_path is required for 'fused' scoring mode")

        text_scores = _score_tfidf(chunks)
        from pipeline.audio_scorer import score_audio
        audio_scores = score_audio(audio_path, chunks)

        w = config.audio_weight
        scores = (1 - w) * text_scores + w * audio_scores

    else:
        raise ValueError(f"Unknown scoring mode: '{mode}'. Use: tfidf, audio, fused")

    # Build sorted results
    scored = [
        ScoredChunk(chunk=c, score=round(float(s), 4))
        for c, s in zip(chunks, scores)
    ]
    scored.sort(key=lambda x: x.score, reverse=True)

    print(f"[score] Scored {len(scored)} chunks  "
          f"(top={scored[0].score:.3f}, bottom={scored[-1].score:.3f})")

    return scored
