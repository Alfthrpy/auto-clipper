"""
Auto-Clipper MVP — extract highlight clips from a video.

Usage:
    python main.py video.mp4
    python main.py video.mp4 --top 3 --model medium
    python main.py video.mp4 --language id
    python main.py video.mp4 --no-render          # raw clips only (fast)
"""
import argparse
import json
import sys
import time
from pathlib import Path

from config import Config
from pipeline.audio_extractor import extract_audio
from pipeline.transcriber import transcribe, Segment, Word
from pipeline.segmenter import merge_segments
from pipeline.scorer import score_chunks
from pipeline.clipper import select_top_clips, extract_clips
from pipeline.subtitle_generator import generate_subtitle_for_clip, generate_hook_ass
from pipeline.renderer import render_vertical_clip, _get_audio_duration
from pipeline.hook_generator import generate_hook_text, generate_hook_audio


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
        help="Transcription backend: local (faster-whisper) or groq. Default: local.",
    )
    p.add_argument(
        "--groq-api-key",
        type=str,
        default=None,
        help="Groq API key (or set GROQ_API_KEY env var).",
    )
    p.add_argument(
        "--scoring", "-s",
        choices=["tfidf", "audio", "llm", "fused"],
        default="fused",
        help="Scoring mode. Default: fused.",
    )
    p.add_argument(
        "--audio-weight",
        type=float,
        default=0.5,
        help="Audio weight in fused mode (0.0=all text, 1.0=all audio). Default: 0.5.",
    )
    # ── Render flags ──
    p.add_argument(
        "--no-render",
        action="store_true",
        help="Skip rendering — just extract raw clips (fast, no re-encode).",
    )
    p.add_argument(
        "--render-preset",
        type=str,
        default="medium",
        choices=["ultrafast", "veryfast", "fast", "medium", "slow"],
        help="x264 encoding preset (default: medium). Use 'ultrafast' for speed.",
    )
    p.add_argument(
        "--no-hook",
        action="store_true",
        help="Disable Hook Text & AI Voice Over generation. Default is enabled.",
    )
    return p


# ── Cache helpers ──────────────────────────────────────────────

def _cache_key(video_path: Path, config: Config) -> str:
    return (f"{video_path.stem}_{config.transcribe_backend}"
            f"_{config.whisper_model}_{config.language or 'auto'}.json")


def _load_cached_segments(cache_file: Path) -> list[Segment] | None:
    if not cache_file.exists():
        return None
    print(f"[transcribe] ♻️  Loading cached transcript from {cache_file.name}")
    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        Segment(
            start=d["start"],
            end=d["end"],
            text=d["text"],
            words=[Word(**w) for w in d["words"]] if d.get("words") else None,
        )
        for d in data
    ]


def _save_segments_cache(segments: list[Segment], cache_file: Path):
    seg_dicts = []
    for s in segments:
        d = {"start": s.start, "end": s.end, "text": s.text}
        if s.words:
            d["words"] = [
                {"start": w.start, "end": w.end, "text": w.text}
                for w in s.words
            ]
        else:
            d["words"] = None
        seg_dicts.append(d)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(seg_dicts, f, indent=2, ensure_ascii=False)
    print(f"[transcribe] 💾 Saved transcript cache to {cache_file.name}")


def _collect_all_words(segments: list[Segment]) -> list[Word]:
    """Flatten word-level timestamps from all segments."""
    words: list[Word] = []
    for seg in segments:
        if seg.words:
            words.extend(seg.words)
    return words


# ── Main pipeline ─────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Validate input
    video_path: Path = args.video
    if not video_path.exists():
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    config = Config(
        transcribe_backend=args.backend,
        groq_api_key=args.groq_api_key,
        whisper_model=args.model,
        whisper_device=args.device,
        scoring_mode=args.scoring,
        audio_weight=args.audio_weight,
        top_k=args.top,
        language=args.language,
        output_dir=Path(args.output),
        cache_dir=args.cache_dir,
        render_enabled=not args.no_render,
        render_preset=args.render_preset,
        hook_enabled=not args.no_hook,
    )
    config.ensure_dirs()

    render_label = "ON (vertical + subtitle)" if config.render_enabled else "OFF (raw clips)"
    hook_label = f"ON ({config.hook_voice_id})" if config.hook_enabled else "OFF"

    print("=" * 60)
    print(f"  Auto-Clipper")
    print(f"  Video  : {video_path.name}")
    print(f"  Backend: {config.transcribe_backend}")
    if config.transcribe_backend == "local":
        print(f"  Model  : {config.whisper_model}")
        print(f"  Device : {config.whisper_device}")
    print(f"  Scoring: {config.scoring_mode}")
    if config.scoring_mode in ["fused", "audio", "tfidf"]:
        print(f"  Weights: text={1 - config.audio_weight:.0%} / audio={config.audio_weight:.0%}")
    print(f"  Top-K  : {config.top_k}")
    print(f"  Render : {render_label}")
    print(f"  Hook AI: {hook_label}")
    print(f"  Output : {config.output_dir}/")
    print("=" * 60)

    t0 = time.time()

    # ── Step 1: Extract audio ──────────────────────────────────
    audio_path = config.temp_dir / f"{video_path.stem}.wav"
    extract_audio(video_path, audio_path)

    # ── Step 2: Transcribe (with caching) ──────────────────────
    cache_file = config.temp_dir / _cache_key(video_path, config)
    segments = _load_cached_segments(cache_file)

    if segments is None:
        segments = transcribe(str(audio_path), config)
        if segments:
            _save_segments_cache(segments, cache_file)

    if not segments:
        print("No speech detected in the video. Exiting.")
        sys.exit(0)

    # Collect all word-level timestamps
    all_words = _collect_all_words(segments)
    print(f"[words] Collected {len(all_words)} word-level timestamps")

    # ── Step 3: Segment (merge into chunks) ────────────────────
    chunks = merge_segments(segments, config)
    if not chunks:
        print("No chunks after segmentation. Exiting.")
        sys.exit(0)

    print("\n── Transcript Chunks ─────────────────────────────────")
    for i, c in enumerate(chunks, 1):
        clean_text = c.text.replace('\n', ' ').replace('\r', '')
        preview = clean_text[:80] + "..." if len(clean_text) > 80 else clean_text
        print(f"  {i:2d}. [{c.start:6.1f}s → {c.end:6.1f}s] {preview}")

    # 4. Score chunks
    print("\n[score] Scoring & ranking chunks...")
    # Note: audio_path string represents the path to .wav
    scored = score_chunks(chunks, audio_path, config)
    
    # Optional debug output for semantic mode
    if config.scoring_mode in ["llm", "fused"]:
        print("\n[score] Semantic Top 3 evaluation:")
        for sc in scored[:3]:
            print(f"  ↳ [{sc.score:.3f}] {sc.chunk.text[:60]}...")

    print("\n── Highlight Scores ──────────────────────────────────")
    for i, sc in enumerate(scored, 1):
        marker = " ★" if i <= config.top_k else ""
        clean_text = sc.chunk.text.replace('\n', ' ').replace('\r', '')
        preview = clean_text[:60] + "..." if len(clean_text) > 60 else clean_text
        print(f"  {i:2d}. score={sc.score:.3f}  "
              f"[{sc.chunk.start:6.1f}s → {sc.chunk.end:6.1f}s]{marker}  {preview}")

    # ── Step 5: Select top clips ───────────────────────────────
    selected = select_top_clips(scored, config.top_k)

    # ── Step 6: Generate output ────────────────────────────────
    stem = video_path.stem
    output_files: list[Path] = []

    if not config.render_enabled:
        # Fast mode: raw clips (no re-encode)
        print(f"\n── Extracting {len(selected)} raw clips ──────────────────")
        output_files = extract_clips(video_path, selected, config)
    else:
        # Full mode: subtitle + vertical render
        print(f"\n── Rendering {len(selected)} vertical shorts ─────────────")
        for i, sc in enumerate(selected, 1):
            clip_start = max(0, sc.chunk.start - config.clip_padding)
            clip_end = sc.chunk.end + config.clip_padding

            # 6a. Generate subtitle
            sub_path = config.temp_dir / f"{stem}_clip{i:02d}.ass"
            subtitle_file = None
            if all_words:
                subtitle_file = generate_subtitle_for_clip(
                    all_words, clip_start, clip_end, sub_path, config
                )

            # 6b. Generate Hook Text & Voice (if enabled)
            hook_text = None
            hook_audio_file = None
            hook_subtitle_file = None
            if config.hook_enabled:
                # Ambil sebagian teks saja untuk konteks ke LLM agar cepat
                clip_text_preview = sc.chunk.text[:500] if len(sc.chunk.text) > 500 else sc.chunk.text
                hook_text = generate_hook_text(clip_text_preview, config.groq_api_key, config.language)
                
                if hook_text:
                    hook_audio_file = config.temp_dir / f"{stem}_clip{i:02d}_hook.mp3"
                    generate_hook_audio(hook_text, hook_audio_file, config.hook_voice_id)
                    
                    # Cek durasi voice over menggunakan probe fungsi _get_audio_duration
                    import imageio_ffmpeg
                    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                    h_dur = _get_audio_duration(ffmpeg_path, hook_audio_file)
                    # Beri buffer sedikit durasi baca
                    h_dur = h_dur + 0.3 if h_dur > 0 else 30.0 
                    
                    hook_subtitle_file = config.temp_dir / f"{stem}_clip{i:02d}_hook.ass"
                    generate_hook_ass(hook_text, hook_subtitle_file, config, duration=h_dur)

            # 6c. Render vertical video
            out_path = config.output_dir / f"{stem}_short{i:02d}.mp4"
            print(f"[clip {i}/{len(selected)}] "
                  f"{clip_start:.1f}s → {clip_end:.1f}s  "
                  f"(score: {sc.score:.3f})")

            render_vertical_clip(
                video_path, clip_start, clip_end,
                subtitle_file, out_path, config,
                hook_audio_file, hook_subtitle_file
            )
            output_files.append(out_path)
            
            # 6d. Save context string for the Upload script
            clip_text = " ".join([w.text for w in all_words if clip_start <= w.start <= clip_end])
            context_data = {
                "video_source": str(video_path.name),
                "score": sc.score,
                "hook": hook_text or "",
                "transcript": clip_text,
                "language": config.language
            }
            json_path = config.output_dir / f"{stem}_short{i:02d}.json"
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(context_data, jf, indent=2, ensure_ascii=False)

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
