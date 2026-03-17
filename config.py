"""
Auto-Clipper Pipeline Configuration
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # -- Transcription backend --
    transcribe_backend: str = "local"   # "local" | "groq"
    groq_api_key: str | None = None     # only needed for backend="groq"

    # -- Whisper STT (local backend) --
    whisper_model: str = "small"
    whisper_device: str = "auto"        # "auto" | "cpu" | "cuda"
    whisper_compute_type: str = "auto"  # "auto" | "int8" | "float16" | "float32"
    language: str | None = None  # None = auto detect

    # -- Segmentation --
    min_chunk_duration: float = 15.0   # seconds – minimum chunk length
    max_chunk_duration: float = 60.0   # seconds – maximum chunk length
    silence_gap_threshold: float = 2.0 # seconds – gap to trigger split
    chunk_duration_seconds: float = 300.0 # seconds – chunk duration

    # -- Scoring --
    scoring_mode: str = "fused"    # "tfidf" | "audio" | "fused"
    audio_weight: float = 0.5     # for fused mode: 0.0 = all text, 1.0 = all audio
    top_k: int = 5  # number of clips to extract

    # -- Clipping --
    clip_padding: float = 2.0  # seconds of padding before/after clip

    # -- Paths --
    output_dir: Path = field(default_factory=lambda: Path("output"))
    temp_dir: Path = field(default_factory=lambda: Path("temp"))
    cache_dir: Path = field(default_factory=lambda: Path("models"))  # model cache

    def ensure_dirs(self):
        """Create output, temp, and cache directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
