"""
Scorer — rank chunks by highlight potential using TF-IDF.

Strategy: chunks that are most *dissimilar* to the average content
are more likely to be highlights (unexpected = interesting).
We also boost chunks with higher word density (more "content").
"""
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pipeline.segmenter import Chunk


@dataclass
class ScoredChunk:
    """A chunk with its computed highlight score."""
    chunk: Chunk
    score: float




def score_chunks(chunks: list[Chunk]) -> list[ScoredChunk]:
    """
    Score each chunk for highlight potential.

    Scoring formula:
      - uniqueness: 1 - cosine_similarity(chunk, mean_vector)
        → chunks that stand out from the overall content score higher
      - density: normalized word count per second
        → chunks with more speech activity score higher
      - final_score = 0.7 * uniqueness + 0.3 * density

    Args:
        chunks: List of text chunks to score.

    Returns:
        List of ScoredChunk sorted by score descending.
    """
    if not chunks:
        return []

    texts = [c.text for c in chunks]

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=None,     # keep all words — important for non-English
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    # -- Uniqueness score --
    mean_vector = np.asarray(tfidf_matrix.mean(axis=0))
    similarities = cosine_similarity(tfidf_matrix, mean_vector).flatten()
    uniqueness = 1.0 - similarities  # more different = more interesting

    # -- Density score (words per second) --
    densities = np.array([
        len(c.text.split()) / max(c.duration, 1.0) for c in chunks
    ])

    # Normalize both to 0-1
    def normalize(arr: np.ndarray) -> np.ndarray:
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val < 1e-9:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    uniqueness_norm = normalize(uniqueness)
    density_norm = normalize(densities)

    # Combined score
    scores = 0.7 * uniqueness_norm + 0.3 * density_norm

    # Build results
    scored = [
        ScoredChunk(chunk=c, score=round(float(s), 4))
        for c, s in zip(chunks, scores)
    ]
    scored.sort(key=lambda x: x.score, reverse=True)

    print(f"[score] Scored {len(scored)} chunks  "
          f"(top={scored[0].score:.3f}, bottom={scored[-1].score:.3f})")

    return scored
