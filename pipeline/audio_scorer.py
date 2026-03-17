"""
Audio Scorer — analyse audio features per chunk using librosa.

Extracts per-chunk scores from:
  - RMS energy        → loud moments (cheers, shouts, emphasis)
  - Spectral centroid → brightness (excitement indicator)
  - Pitch variation   → expressive speech / surprise
  - Zero-crossing rate → noise/energy bursts
"""
import numpy as np
import librosa

from pipeline.segmenter import Chunk


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def score_audio(audio_path: str, chunks: list[Chunk]) -> np.ndarray:
    """
    Compute an audio-based highlight score for each chunk.

    Loads the audio once, then extracts features for each chunk's
    time window. Returns a numpy array of scores (0-1) aligned
    with the chunks list.

    Args:
        audio_path: Path to the WAV file (16kHz mono expected).
        chunks:     List of Chunk objects with start/end times.

    Returns:
        np.ndarray of shape (len(chunks),) with scores in [0, 1].
    """
    print("[audio] Loading audio for feature extraction ...")
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    total_samples = len(y)

    rms_scores = []
    spectral_scores = []
    pitch_var_scores = []
    zcr_scores = []

    for chunk in chunks:
        # Slice audio for this chunk
        start_sample = int(chunk.start * sr)
        end_sample = min(int(chunk.end * sr), total_samples)

        if start_sample >= end_sample:
            rms_scores.append(0.0)
            spectral_scores.append(0.0)
            pitch_var_scores.append(0.0)
            zcr_scores.append(0.0)
            continue

        y_chunk = y[start_sample:end_sample]

        # RMS energy — mean loudness of this chunk
        rms = librosa.feature.rms(y=y_chunk)[0]
        rms_scores.append(float(rms.mean()))

        # Spectral centroid — brightness (higher = more energetic)
        cent = librosa.feature.spectral_centroid(y=y_chunk, sr=sr)[0]
        spectral_scores.append(float(cent.mean()))

        # Pitch variation — std of F0 (more variation = more expressive)
        pitches, magnitudes = librosa.piptrack(y=y_chunk, sr=sr)
        # Get the most prominent pitch per frame
        pitch_per_frame = []
        for t in range(pitches.shape[1]):
            idx = magnitudes[:, t].argmax()
            p = pitches[idx, t]
            if p > 0:
                pitch_per_frame.append(p)
        pitch_std = float(np.std(pitch_per_frame)) if pitch_per_frame else 0.0
        pitch_var_scores.append(pitch_std)

        # Zero-crossing rate — noisiness / rapid changes
        zcr = librosa.feature.zero_crossing_rate(y_chunk)[0]
        zcr_scores.append(float(zcr.mean()))

    # Convert to arrays and normalize
    rms_arr = _normalize(np.array(rms_scores))
    spectral_arr = _normalize(np.array(spectral_scores))
    pitch_arr = _normalize(np.array(pitch_var_scores))
    zcr_arr = _normalize(np.array(zcr_scores))

    # Weighted combination
    # RMS (loudness) is the strongest signal, then pitch variability
    audio_scores = (
        0.40 * rms_arr
        + 0.25 * pitch_arr
        + 0.20 * spectral_arr
        + 0.15 * zcr_arr
    )

    print(f"[audio] Done — scored {len(chunks)} chunks  "
          f"(top={audio_scores.max():.3f}, bottom={audio_scores.min():.3f})")

    return audio_scores
