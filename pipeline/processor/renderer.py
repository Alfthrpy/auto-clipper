"""
Renderer — produce vertical (9:16) video with burned-in subtitles.

Pipeline per clip:
  1. Seek to clip start in source video
  2. Center-crop to 9:16 aspect ratio
  3. Scale to 1080×1920
  4. Burn ASS subtitle overlay
  5. Encode with H.264 + AAC
"""
import subprocess
from pathlib import Path

import imageio_ffmpeg

from config import Config


def get_ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _get_audio_duration(ffmpeg: str, path: Path) -> float:
    cmd = [
        ffmpeg, "-i", str(path), "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    import re
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
    if match:
        h, m, s = match.groups()
        return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0

def render_vertical_clip(
    video_path: Path,
    clip_start: float,
    clip_end: float,
    subtitle_path: Path | None,
    output_path: Path,
    config: Config,
    hook_audio_path: Path | None = None,
    hook_subtitle_path: Path | None = None,
) -> Path:
    """
    Render a single vertical clip with subtitle, and optional hook text/TTS.

    Args:
        video_path:     Source video file.
        clip_start:     Start time in seconds (absolute).
        clip_end:       End time in seconds (absolute).
        subtitle_path:  Path to .ass subtitle file (or None to skip).
        hook_audio_path: Optional path to TTS hook audio.
        hook_subtitle_path: Optional path to top hook subtitle.
        output_path:    Where to write the rendered clip.
        config:         Pipeline config.

    Returns:
        Path to the rendered output file.

    Raises:
        RuntimeError: If ffmpeg encoding fails.
    """
    ffmpeg = get_ffmpeg()
    duration = clip_end - clip_start
    w, h = config.render_width, config.render_height

    # ── Build video filter chain ────────────────────────────────
    filters = []

    # 1. Center-crop to 9:16 aspect ratio
    filters.append("crop=trunc(ih*9/16/2)*2:ih")
    # 2. Scale to target resolution (1080×1920)
    filters.append(f"scale={w}:{h}:flags=lanczos")
    
    def _safe_ass(p: Path) -> str:
        import os
        try:
            safe = os.path.relpath(p, Path.cwd())
        except ValueError:
            safe = str(p.resolve())
        safe = safe.replace("\\", "/").replace(":", "\\\\:")
        return safe

    # 3. Burn Hook Subtitle (if any)
    if hook_subtitle_path and hook_subtitle_path.exists():
        filters.append(f"ass='{_safe_ass(hook_subtitle_path)}'")

    # 4. Burn Karaoke Subtitle
    if subtitle_path and subtitle_path.exists():
        filters.append(f"ass='{_safe_ass(subtitle_path)}'")

    vf = ",".join(filters)

    # ── Build ffmpeg command ────────────────────────────────────
    cmd = [
        ffmpeg,
        "-ss", f"{clip_start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(video_path),
    ]
    
    if hook_audio_path and hook_audio_path.exists():
        hook_dur = _get_audio_duration(ffmpeg, hook_audio_path)
        cmd.extend(["-i", str(hook_audio_path)])
        
        duck_vol = config.hook_ducking_volume
        # Audio filter: Video audio is [0:a], TTS audio is [1:a]
        # volume: duck original audio for hook_dur seconds
        af = (
            f"[0:a]volume='if(between(t,0,{hook_dur:.2f}),{duck_vol},1.0)':eval=frame[abg];"
            f"[1:a]volume=1.5[atts];" # slight boost to TTS
            f"[abg][atts]amix=inputs=2:duration=longest:normalize=0[aout]"
        )
        cmd.extend(["-filter_complex", f"[0:v]{vf}[vout];{af}"])
        cmd.extend(["-map", "[vout]", "-map", "[aout]"])
    else:
        cmd.extend(["-vf", vf])
        
    cmd.extend([
        "-c:v", "libx264",
        "-preset", config.render_preset,
        "-crf", str(config.render_crf),
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-y",
        str(output_path),
    ])

    print(f"[render] Encoding {clip_start:.1f}s → {clip_end:.1f}s  "
          f"({duration:.1f}s, {w}×{h}) ...")

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        raise RuntimeError(
            f"Render failed (code {result.returncode}):\n{stderr[-600:]}"
        )

    size_mb = output_path.stat().st_size / 1_048_576
    print(f"[render] ✓ {output_path.name}  ({size_mb:.1f} MB)")
    return output_path
