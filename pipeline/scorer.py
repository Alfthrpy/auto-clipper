"""
Scorer — rank chunks by highlight potential.

Supports three modes:
  - "tfidf"  → text-only TF-IDF uniqueness scoring (MVP)
  - "audio"  → audio-only energy/pitch scoring
  - "fused"  → weighted combination of text + audio (recommended)
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import Config
from pipeline.segmenter import Chunk
from pipeline.audio_scorer import score_audio
from pipeline.semantic_scorer import score_semantic_llm


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
    video_path: Path,
    config: Config
) -> list[ScoredChunk]:
    """Score chunks based on the configured mode."""
    if not chunks:
        return []

    # Prepare individual dimension arrays
    tfidf_scores = np.zeros(len(chunks))
    audio_scores = np.zeros(len(chunks))
    llm_scores = np.zeros(len(chunks))

    # Calculate TF-IDF
    if config.scoring_mode in ["tfidf", "fused"]:
        tfidf_scores = _score_tfidf(chunks)

    # Calculate Audio
    if config.scoring_mode in ["audio", "fused"]:
        audio_scores = score_audio(chunks, video_path)
        
    # Calculate Semantic / LLM
    if config.scoring_mode in ["llm", "fused"]:
        # Pre-filter logic to save token limits if total chunks > top_n limit
        if len(chunks) > config.semantic_pre_filter_top_n:
            # Create a simple rough draft score using TF-IDF and Audio to prune
            rough_scores = tfidf_scores * (1 - config.audio_weight) + audio_scores * config.audio_weight
            if rough_scores.sum() == 0:
                # Fallback if both were not calculated
                rough_scores = _score_tfidf(chunks)
                
            top_indices = np.argsort(rough_scores)[-config.semantic_pre_filter_top_n:]
            # Send only the top N to LLM
            filtered_chunks = [chunks[i] for i in sorted(top_indices)]
            f_llm_scores = score_semantic_llm(filtered_chunks, config)
            
            # Map back to full length array
            for f_idx, orig_idx in enumerate(sorted(top_indices)):
                llm_scores[orig_idx] = f_llm_scores[f_idx]
        else:
            # Score all chunks
            llm_scores = score_semantic_llm(chunks, config)

    # --- Fusion & Mapping ---
    result = []
    
    # Pre-calculate factors for fused
    w_aud = config.audio_weight
    w_sem = config.semantic_weight
    # tfidf weight is whatever is left over
    w_tfidf = max(0.0, 1.0 - w_aud - w_sem)

    for i, chunk in enumerate(chunks):
        if config.scoring_mode == "fused":
            # Example blend: 30% audio, 50% semantic, 20% tfidf
            final_score = (
                tfidf_scores[i] * w_tfidf +
                audio_scores[i] * w_aud +
                llm_scores[i] * w_sem
            )
        elif config.scoring_mode == "audio":
            final_score = audio_scores[i]
        elif config.scoring_mode == "llm":
            final_score = llm_scores[i]
        else:
            final_score = tfidf_scores[i]

        result.append(ScoredChunk(chunk=chunk, score=final_score))

    result.sort(key=lambda x: x.score, reverse=True)
    return result
