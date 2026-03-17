"""
Transcriber — speech-to-text with pluggable backends.

Supported backends:
  - "local"  → faster-whisper (runs on your machine, CPU or GPU)
  - "groq"   → Groq cloud API (free, fast, needs API key)

All backends return the same list[Segment] with timestamps.
"""
from dataclasses import dataclass

from config import Config


@dataclass
class Segment:
    """A single transcript segment with timestamps."""
    start: float   # seconds
    end: float     # seconds
    text: str


def transcribe(audio_path: str, config: Config) -> list[Segment]:
    """
    Transcribe audio using the configured backend.

    Args:
        audio_path: Path to a WAV file.
        config:     Pipeline configuration.

    Returns:
        List of Segment objects with start, end, and text.
    """
    backend = config.transcribe_backend

    if backend == "local":
        from pipeline.backends.local_whisper import transcribe_local
        return transcribe_local(audio_path, config)

    elif backend == "groq":
        from pipeline.backends.groq_whisper import transcribe_groq
        return transcribe_groq(audio_path, config)

    else:
        raise ValueError(
            f"Unknown transcription backend: '{backend}'. "
            f"Supported: 'local', 'groq'"
        )
