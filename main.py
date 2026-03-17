"""
Auto-Clipper MVP — extract highlight clips from a video.

Usage:
    python main.py video.mp4
    python main.py video.mp4 --top 3 --model medium
    python main.py video.mp4 --language id
"""
import argparse
import sys
import time
from pathlib import Path

from config import Config
from pipeline.audio_extractor import extract_audio
from pipeline.transcriber import transcribe
from pipeline.segmenter import merge_segments
from pipeline.scorer import score_chunks
from pipeline.clipper import select_top_clips, extract_clips


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Auto-Clipper: extract highlight clips from a video.",
    )
    p.add_argument(
        "video",
        type=Path,
        help="Path to the input video file.",
    )
    p.add_argument(
        "--top", "-k",
        type=int,
        default=5,
        help="Number of highlight clips to extract (default: 5).",
    )
    p.add_argument(
        "--model", "-m",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: small).",
    )
    p.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="Language code, e.g. 'id', 'en'. Auto-detect if omitted.",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output"),
        help="Output directory for clips (default: output/).",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("models"),
        help="Directory to cache heavy model files (default: models/).",
    )
    p.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for local transcription (default: auto-detect).",
    )
    p.add_argument(
        "--backend", "-b",
        type=str,
        default="local",
        choices=["local", "groq"],
        help="Transcription backend: local (faster-whisper) or groq (cloud API). Default: local.",
    )
    p.add_argument(
        "--groq-api-key",
        type=str,
        default=None,
        help="Groq API key (or set GROQ_API_KEY env var). Only needed for --backend groq.",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Validate input
    video_path: Path = args.video
    if not video_path.exists():
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    # Configure
    config = Config(
        transcribe_backend=args.backend,
        groq_api_key=args.groq_api_key,
        whisper_model=args.model,
        whisper_device=args.device,
        language=args.language,
        top_k=args.top,
        output_dir=args.output,
        cache_dir=args.cache_dir,
    )
    config.ensure_dirs()

    print("=" * 60)
    print(f"  Auto-Clipper")
    print(f"  Video  : {video_path.name}")
    print(f"  Backend: {config.transcribe_backend}")
    if config.transcribe_backend == "local":
        print(f"  Model  : {config.whisper_model}")
        print(f"  Device : {config.whisper_device}")
    print(f"  Top-K  : {config.top_k}")
    print(f"  Cache  : {config.cache_dir}/")
    print(f"  Output : {config.output_dir}/")
    print("=" * 60)

    t0 = time.time()

    # ── Step 1: Extract audio ──────────────────────────────────
    audio_path = config.temp_dir / f"{video_path.stem}.wav"
    extract_audio(video_path, audio_path)

    # ── Step 2: Transcribe ─────────────────────────────────────
    segments = transcribe(str(audio_path), config)
    if not segments:
        print("No speech detected in the video. Exiting.")
        sys.exit(0)

    # ── Step 3: Segment (merge into chunks) ────────────────────
    chunks = merge_segments(segments, config)
    if not chunks:
        print("No chunks after segmentation. Exiting.")
        sys.exit(0)

    # Print chunks for visibility
    print("\n── Transcript Chunks ─────────────────────────────────")
    for i, c in enumerate(chunks, 1):
        preview = c.text[:80] + "..." if len(c.text) > 80 else c.text
        print(f"  {i:2d}. [{c.start:6.1f}s → {c.end:6.1f}s] {preview}")

    # ── Step 4: Score ──────────────────────────────────────────
    scored = score_chunks(chunks)

    print("\n── Highlight Scores ──────────────────────────────────")
    for i, sc in enumerate(scored, 1):
        marker = " ★" if i <= config.top_k else ""
        preview = sc.chunk.text[:60] + "..." if len(sc.chunk.text) > 60 else sc.chunk.text
        print(f"  {i:2d}. score={sc.score:.3f}  "
              f"[{sc.chunk.start:6.1f}s → {sc.chunk.end:6.1f}s]{marker}  {preview}")

    # ── Step 5: Select & extract clips ─────────────────────────
    selected = select_top_clips(scored, config.top_k)

    print(f"\n── Extracting {len(selected)} clips ───────────────────────────")
    output_files = extract_clips(video_path, selected, config)

    # ── Summary ────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Generated {len(output_files)} clips:")
    for f in output_files:
        size_mb = f.stat().st_size / 1_048_576
        print(f"    → {f.name}  ({size_mb:.1f} MB)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
