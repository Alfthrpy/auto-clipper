"""
Local Whisper backend — faster-whisper running on your machine.
Supports CPU (int8) and GPU (float16) with auto-detection.
"""
from __future__ import annotations

import gc
import logging
import re
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
from faster_whisper import WhisperModel

from config import Config
from pipeline.transcriber import Segment, Word

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Resource check helpers
# ─────────────────────────────────────────────

def _check_resources(device: str) -> None:
    """
    Raise RuntimeError dengan pesan jelas jika resource tidak cukup.
    Dipanggil SEBELUM model load dan SEBELUM transcribe dimulai.
    """
    if device == "cuda":
        try:
            import subprocess
            # Gunakan nvidia-smi untuk mengecek VRAM (lebih reliable di Windows/Linux)
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,nounits,noheader"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                # Ambil GPU pertama (index 0)
                gpu0_mem = result.stdout.strip().split('\n')[0]
                free_mb_str, total_mb_str = gpu0_mem.split(', ')
                free_mb = float(free_mb_str)
                total_mb = float(total_mb_str)
                
                print(f"[local] VRAM: {free_mb:.0f} MB free / {total_mb:.0f} MB total")
                if free_mb < 1500:
                    raise RuntimeError(
                        f"VRAM tidak cukup: hanya {free_mb:.0f} MB tersisa "
                        f"(minimum ~1500 MB). Tutup aplikasi GPU lain atau "
                        f"ganti --device cpu."
                    )
        except (subprocess.SubprocessError, FileNotFoundError, ValueError):
            pass  # Abaikan jika nvidia-smi tidak ditemukan/gagal

    # RAM check (berlaku untuk CPU & GPU karena model tetap load ke RAM)
    try:
        import psutil
        ram = psutil.virtual_memory()
        free_mb = ram.available / (1024 ** 2)
        print(f"[local] RAM: {free_mb:.0f} MB available")
        if free_mb < 1000:
            raise RuntimeError(
                f"RAM tidak cukup: hanya {free_mb:.0f} MB tersisa "
                f"(minimum ~1000 MB). Tutup aplikasi lain terlebih dahulu."
            )
    except ImportError:
        logger.warning("[local] psutil tidak tersedia — skip RAM check. "
                       "Install: pip install psutil")


# ─────────────────────────────────────────────
# Device resolution
# ─────────────────────────────────────────────

def _resolve_device(config: Config) -> tuple[str, str]:
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


# ─────────────────────────────────────────────
# Model cache
# ─────────────────────────────────────────────

_MODEL_CACHE: dict[str, WhisperModel] = {}

def _get_model(config: Config, device: str, compute_type: str) -> WhisperModel:
    model_key = f"{config.whisper_model}_{device}_{compute_type}"
    if model_key not in _MODEL_CACHE:
        _check_resources(device)  # ← cek resource SEBELUM load model
        print(f"[local] Loading model '{config.whisper_model}' "
              f"(device={device}, compute={compute_type})")
        _MODEL_CACHE[model_key] = WhisperModel(
            config.whisper_model,
            device=device,
            compute_type=compute_type,
            download_root=str(config.cache_dir),
        )
    return _MODEL_CACHE[model_key]


# ─────────────────────────────────────────────
# FFmpeg helpers
# ─────────────────────────────────────────────

def _get_duration(ffmpeg: str, audio_path: Path) -> float:
    """Return total duration in seconds. Raises RuntimeError if probe fails."""
    result = subprocess.run(
        [ffmpeg, "-i", str(audio_path), "-f", "null", "-"],
        capture_output=True, text=True,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
    if not match:
        raise RuntimeError(
            f"Tidak bisa membaca durasi audio dari '{audio_path}'. "
            f"Pastikan file tidak corrupt.\nFFmpeg stderr:\n{result.stderr[-500:]}"
        )
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def _extract_chunk(
    ffmpeg: str,
    audio_path: Path,
    chunk_path: Path,
    offset: float,
    duration: float,
) -> None:
    """Extract satu chunk WAV. Raises RuntimeError jika ffmpeg gagal."""
    cmd = [
        ffmpeg,
        "-ss", str(offset),
        "-i", str(audio_path),
        "-t", str(duration),
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-y", str(chunk_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg gagal extract chunk offset={offset:.0f}s.\n"
            f"stderr: {result.stderr[-300:]}"
        )
    if not chunk_path.exists() or chunk_path.stat().st_size < 1000:
        raise RuntimeError(
            f"Chunk offset={offset:.0f}s menghasilkan file kosong atau terlalu kecil. "
            f"Kemungkinan offset melebihi durasi aktual audio."
        )


# ─────────────────────────────────────────────
# Main transcribe
# ─────────────────────────────────────────────

def transcribe_local(audio_path: str, config: Config) -> list[Segment]:
    """
    Transcribe menggunakan faster-whisper secara lokal.
    - Chunking otomatis untuk audio panjang
    - Feedback jelas jika resource tidak cukup
    - Tidak silent-fail jika satu chunk gagal
    """
    device, compute_type = _resolve_device(config)
    model = _get_model(config, device, compute_type)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    audio_file = Path(audio_path)

    total_duration = _get_duration(ffmpeg, audio_file)
    print(f"[local] Durasi total: {total_duration / 60:.1f} menit")

    chunk_duration = config.chunk_duration_seconds  # default 600
    total_chunks = int(total_duration / chunk_duration) + 1
    all_segments: list[Segment] = []

    _check_resources(device)  # ← cek resource sekali lagi sebelum transcribe mulai

    with tempfile.TemporaryDirectory(prefix="whisper_chunk_") as tmpdir:
        for chunk_index in range(total_chunks):
            offset = chunk_index * chunk_duration
            if offset >= total_duration:
                break

            chunk_path = Path(tmpdir) / f"chunk_{chunk_index:03d}.wav"
            end_time = min(offset + chunk_duration, total_duration)

            print(f"[local] Chunk {chunk_index + 1}/{total_chunks} "
                  f"({offset / 60:.1f} - {end_time / 60:.1f} menit) ...")

            # ── Extract chunk ──────────────────────────────
            try:
                _extract_chunk(ffmpeg, audio_file, chunk_path, offset, chunk_duration)
            except RuntimeError as e:
                # Chunk terakhir yang melebihi durasi → normal, stop
                if chunk_index > 0 and "terlalu kecil" in str(e):
                    print(f"[local] Chunk {chunk_index + 1} kosong — selesai.")
                    break
                raise  # Re-raise untuk error yang tidak terduga

            # ── Transcribe chunk ───────────────────────────
            try:
                raw_segments, info = model.transcribe(
                    str(chunk_path),
                    beam_size=5,
                    language=config.language,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    word_timestamps=True,
                )

                chunk_count = 0
                for seg in raw_segments:  # exhaust generator segera
                    # Parse word-level timestamps
                    words = None
                    if seg.words:
                        words = [
                            Word(
                                start=round(w.start + offset, 2),
                                end=round(w.end + offset, 2),
                                text=w.word.strip(),
                            )
                            for w in seg.words
                            if w.word.strip()
                        ]

                    all_segments.append(Segment(
                        start=round(seg.start + offset, 2),
                        end=round(seg.end + offset, 2),
                        text=seg.text.strip(),
                        words=words,
                    ))
                    chunk_count += 1

                print(f"[local] ✓ Chunk {chunk_index + 1} — {chunk_count} segmen")

            except Exception as e:
                raise RuntimeError(
                    f"Transcribe gagal di chunk {chunk_index + 1} "
                    f"(offset {offset:.0f}s).\n"
                    f"Error: {type(e).__name__}: {e}\n"
                    f"Tip: Coba kurangi chunk_duration atau ganti ke device='cpu'."
                ) from e

            finally:
                # Hapus chunk file segera setelah diproses — hemat disk
                if chunk_path.exists():
                    chunk_path.unlink(missing_ok=True)

            # Explicit GC setelah setiap chunk — bantu stabilitas CUDA di Windows
            gc.collect()

    print(f"[local] Done — {len(all_segments)} segmen total "
          f"({chunk_index + 1} chunk diproses)")
    return all_segments