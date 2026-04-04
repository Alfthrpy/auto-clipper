"""
Groq Whisper backend — transcription via Groq cloud API.

Free tier: ~7,200 audio-seconds/day (~2 hours).
Uses Whisper large-v3 on Groq's LPU hardware → very fast.

Requirements:
  - pip install groq
  - Set GROQ_API_KEY env variable or pass via config
"""
import os
from pathlib import Path

from config import Config
from pipeline.transcriber import Segment, Word

from dotenv import load_dotenv
load_dotenv()

def transcribe_groq(audio_path: str, config: Config) -> list[Segment]:
    """Transcribe using Groq's Whisper API."""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError(
            "Groq backend requires the 'groq' package.\n"
            "Install it with: pip install groq"
        )

    api_key = config.groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "Groq API key not found. Set it via:\n"
            "  - Environment variable: GROQ_API_KEY=your_key\n"
            "  - CLI flag: --groq-api-key your_key\n"
            "  - .env file: GROQ_API_KEY=your_key\n\n"
            "Get a free key at: https://console.groq.com/keys"
        )

    client = Groq(api_key=api_key)
    audio_file = Path(audio_path)

    print(f"[groq] Uploading & transcribing ({audio_file.stat().st_size / 1_048_576:.1f} MB) ...")

    # Groq accepts files up to 25 MB. For larger files we chunk.
    file_size_mb = audio_file.stat().st_size / 1_048_576

    if file_size_mb > 25:
        return _transcribe_chunked(client, audio_path, config)

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(audio_file.name, f),
            model="whisper-large-v3",
            response_format="verbose_json",
            language=config.language,
            timestamp_granularities=["segment", "word"],
        )

    segments = _parse_response(response)
    print(f"[groq] Done — {len(segments)} segments")
    return segments


def _transcribe_chunked(client, audio_path: str, config: Config) -> list[Segment]:
    """Split large audio into <25 MB chunks and transcribe each."""
    import subprocess
    import tempfile

    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    audio_file = Path(audio_path)

    # Get total duration
    probe_cmd = [
        ffmpeg, "-i", str(audio_file),
        "-f", "null", "-"
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)

    # Estimate chunk duration: 16kHz mono 16-bit ≈ 32 KB/s → 25 MB ≈ 800s
    chunk_duration = 600  # 10 minutes per chunk (safe margin)

    # Get duration from ffmpeg stderr
    import re
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
    if duration_match:
        h, m, s = duration_match.groups()
        total_duration = int(h) * 3600 + int(m) * 60 + float(s)
    else:
        total_duration = 7200  # fallback 2h

    all_segments: list[Segment] = []
    offset = 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_index = 0
        while offset < total_duration:
            chunk_path = Path(tmpdir) / f"chunk_{chunk_index:03d}.wav"
            cmd = [
                ffmpeg,
                "-ss", str(offset),
                "-i", str(audio_file),
                "-t", str(chunk_duration),
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                "-y", str(chunk_path),
            ]
            subprocess.run(cmd, capture_output=True)

            if not chunk_path.exists() or chunk_path.stat().st_size < 1000:
                break

            print(f"[groq] Chunk {chunk_index + 1} "
                  f"({offset:.0f}s - {offset + chunk_duration:.0f}s) ...")

            with open(chunk_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    file=(chunk_path.name, f),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    language=config.language,
                    timestamp_granularities=["segment", "word"],
                )

            # Adjust timestamps by offset
            for seg in _parse_response(response):
                all_segments.append(Segment(
                    start=round(seg.start + offset, 2),
                    end=round(seg.end + offset, 2),
                    text=seg.text,
                ))

            offset += chunk_duration
            chunk_index += 1

    print(f"[groq] Done — {len(all_segments)} segments ({chunk_index} chunks)")
    return all_segments


def _parse_response(response) -> list[Segment]:
    """Parse Groq API response into Segment list with word timestamps."""
    segments: list[Segment] = []

    # Parse word-level data (flat list from API)
    all_words: list[Word] = []
    if hasattr(response, "words") and response.words:
        for w in response.words:
            word_text = w.get("word", "").strip() if isinstance(w, dict) else getattr(w, "word", "").strip()
            w_start = w.get("start", 0) if isinstance(w, dict) else getattr(w, "start", 0)
            w_end = w.get("end", 0) if isinstance(w, dict) else getattr(w, "end", 0)
            if word_text:
                all_words.append(Word(
                    start=round(float(w_start), 2),
                    end=round(float(w_end), 2),
                    text=word_text,
                ))

    if hasattr(response, "segments") and response.segments:
        for seg in response.segments:
            text = seg.get("text", "").strip() if isinstance(seg, dict) else getattr(seg, "text", "").strip()
            start = seg.get("start", 0) if isinstance(seg, dict) else getattr(seg, "start", 0)
            end = seg.get("end", 0) if isinstance(seg, dict) else getattr(seg, "end", 0)
            if text:
                seg_start = round(float(start), 2)
                seg_end = round(float(end), 2)
                # Match words to this segment by time range
                seg_words = [
                    w for w in all_words
                    if w.start >= seg_start - 0.05 and w.end <= seg_end + 0.05
                ] or None
                segments.append(Segment(
                    start=seg_start,
                    end=seg_end,
                    text=text,
                    words=seg_words,
                ))

    return segments
