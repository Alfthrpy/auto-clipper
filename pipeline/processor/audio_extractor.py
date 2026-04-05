"""
Audio Extractor — extract audio track from video as WAV using ffmpeg.
Uses imageio-ffmpeg so no system ffmpeg install is needed.
"""
import subprocess
from pathlib import Path

import imageio_ffmpeg


def get_ffmpeg() -> str:
    """Get the path to the bundled ffmpeg binary."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """
    Extract audio from a video file as 16 kHz mono WAV.

    Args:
        video_path:  Path to the input video file.
        output_path: Path where the .wav file will be written.

    Returns:
        The output_path on success.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero code.
    """
    ffmpeg = get_ffmpeg()
    cmd = [
        ffmpeg,
        "-i", str(video_path),
        "-vn",                     # no video
        "-acodec", "pcm_s16le",    # 16-bit PCM
        "-ar", "16000",            # 16 kHz (what Whisper expects)
        "-ac", "1",                # mono
        "-y",                      # overwrite
        str(output_path),
    ]

    print(f"[extract] Extracting audio → {output_path.name}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (code {result.returncode}):\n"
            f"{result.stderr.decode(errors='replace')}"
        )

    print(f"[extract] Done  ({output_path.stat().st_size / 1_048_576:.1f} MB)")
    return output_path
