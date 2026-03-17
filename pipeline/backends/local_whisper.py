"""
Local Whisper backend — faster-whisper running on your machine.
Supports CPU (int8) and GPU (float16) with auto-detection.
"""
from faster_whisper import WhisperModel

from config import Config
from pipeline.transcriber import Segment


def _resolve_device(config: Config) -> tuple[str, str]:
    """Auto-detect the best device and compute type."""
    device = config.whisper_device
    compute = config.whisper_compute_type

    if device == "auto":
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                device = "cuda"
                print("[local] 🚀 CUDA GPU detected — using GPU acceleration")
            else:
                device = "cpu"
                print("[local] No CUDA GPU found — using CPU")
        except Exception:
            device = "cpu"
            print("[local] CUDA not available — using CPU")

    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"

    return device, compute


def transcribe_local(audio_path: str, config: Config) -> list[Segment]:
    """Transcribe using faster-whisper locally."""
    device, compute_type = _resolve_device(config)

    print(f"[local] Loading model '{config.whisper_model}' "
          f"(device={device}, compute={compute_type})")

    model = WhisperModel(
        config.whisper_model,
        device=device,
        compute_type=compute_type,
        download_root=str(config.cache_dir),
    )

    print("[local] Transcribing ...")
    raw_segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language=config.language,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    segments: list[Segment] = []
    for seg in raw_segments:
        segments.append(Segment(
            start=round(seg.start, 2),
            end=round(seg.end, 2),
            text=seg.text.strip(),
        ))

    duration_min = info.duration / 60
    print(f"[local] Done — {len(segments)} segments, "
          f"language: {info.language} ({info.language_probability:.0%}), "
          f"duration: {duration_min:.1f} min")

    return segments
