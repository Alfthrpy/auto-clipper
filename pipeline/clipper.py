"""
Clipper — extract top-K highlight clips from the original video.
Uses imageio-ffmpeg for the bundled ffmpeg binary.
"""
import subprocess
from pathlib import Path

import imageio_ffmpeg

from config import Config
from pipeline.scorer import ScoredChunk


def get_ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _clips_overlap(a: ScoredChunk, b: ScoredChunk) -> bool:
    """Check if two clips overlap in time."""
    return a.chunk.start < b.chunk.end and b.chunk.start < a.chunk.end


def select_top_clips(
    scored: list[ScoredChunk],
    top_k: int,
) -> list[ScoredChunk]:
    """
    Select top-K non-overlapping clips using greedy non-maximum suppression.

    Args:
        scored: Scored chunks sorted descending by score.
        top_k:  Max number of clips to return.

    Returns:
        Selected clips, sorted by start time.
    """
    selected: list[ScoredChunk] = []

    for candidate in scored:
        if len(selected) >= top_k:
            break
        # Check overlap with already selected
        if any(_clips_overlap(candidate, s) for s in selected):
            continue
        selected.append(candidate)

    # Sort by time for nice sequential output
    selected.sort(key=lambda x: x.chunk.start)
    return selected


def extract_clips(
    video_path: Path,
    clips: list[ScoredChunk],
    config: Config,
) -> list[Path]:
    """
    Extract video clips from the source file.

    Adds padding around each clip and uses ffmpeg copy mode for speed
    (no re-encoding needed for rough cuts).

    Args:
        video_path: Path to the original video.
        clips:      List of selected ScoredChunks.
        config:     Pipeline configuration.

    Returns:
        List of paths to the generated clip files.
    """
    ffmpeg = get_ffmpeg()
    output_paths: list[Path] = []
    stem = video_path.stem

    for i, sc in enumerate(clips, 1):
        start = max(0, sc.chunk.start - config.clip_padding)
        end = sc.chunk.end + config.clip_padding

        out_file = config.output_dir / f"{stem}_clip{i:02d}.mp4"

        cmd = [
            ffmpeg,
            "-ss", f"{start:.2f}",
            "-i", str(video_path),
            "-to", f"{end - start:.2f}",  # duration relative to -ss
            "-c", "copy",                  # no re-encoding = fast
            "-avoid_negative_ts", "make_zero",
            "-y",
            str(out_file),
        ]

        print(f"[clip {i}/{len(clips)}] "
              f"{start:.1f}s → {end:.1f}s  (score: {sc.score:.3f})  "
              f"→ {out_file.name}")

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"  ⚠ ffmpeg warning: {result.stderr.decode(errors='replace')[-200:]}")

        output_paths.append(out_file)

    print(f"[clip] Extracted {len(output_paths)} clips to {config.output_dir}/")
    return output_paths
