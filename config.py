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
    min_chunk_duration: float = 20.0   # seconds – minimum chunk length
    max_chunk_duration: float = 35.0   # seconds – maximum chunk length
    silence_gap_threshold: float = 2.0 # seconds – gap to trigger split
    chunk_duration_seconds: float = 300.0 # seconds – chunk duration

    # -- Scoring --
    scoring_mode: str = "fused"    # "tfidf" | "audio" | "llm" | "fused"
    audio_weight: float = 0.3      # for fused mode
    semantic_weight: float = 0.5   # for fused mode (value meaning it's highly prioritized)
    semantic_pre_filter_top_n: int = 40  # Max chunks to send to LLM (for long videos to save rate limit)
    top_k: int = 5  # number of clips to extract

    # -- Clipping --
    clip_padding: float = 3.0  # seconds of padding before/after clip

    # -- Subtitle --
    subtitle_font: str = "Arial Black"
    subtitle_fontsize: int = 52
    subtitle_words_per_line: int = 4
    subtitle_highlight_color: tuple[int, int, int] = (255, 255, 0)   # Yellow (R,G,B)
    subtitle_text_color: tuple[int, int, int] = (255, 255, 255)      # White
    subtitle_margin_v: int = 250  # pixels from bottom

    # -- Rendering --
    render_enabled: bool = True
    render_width: int = 1080
    render_height: int = 1920
    render_crf: int = 23          # quality (lower = better, 18-28 typical)
    render_preset: str = "medium" # x264 preset (ultrafast→veryslow)

    # -- Hook & TTS Voice (Phase 2) --
    hook_enabled: bool = True
    hook_voice_id: str = "en-US-ChristopherNeural" # Bahasa Indonesia male. Use "en-US-ChristopherNeural" for EN.
    hook_ducking_volume: float = 0.2        # Volume of original video while TTS is playing
    
    # -- Paths --
    output_dir: Path = field(default_factory=lambda: Path("output"))
    temp_dir: Path = field(default_factory=lambda: Path("temp"))
    cache_dir: Path = field(default_factory=lambda: Path("models"))  # model cache

    def ensure_dirs(self):
        """Create output, temp, and cache directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

