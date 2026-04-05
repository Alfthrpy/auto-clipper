"""
Auto-Clipper Uploader CLI
A dedicated script to generate metadata and upload a rendered short to YouTube.

Usage:
    python upload.py output/video3_short01.mp4
"""
import argparse
import json
import sys
from pathlib import Path

from config import Config
from pipeline.generator.metadata_generator import generate_youtube_metadata
from pipeline.uploader import upload_to_youtube


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate metadata and upload a rendered clip to YouTube Shorts."
    )
    p.add_argument(
        "video",
        type=Path,
        help="Path to the rendered video file (e.g. output/video3_short01.mp4).",
    )
    p.add_argument(
        "--groq-api-key",
        type=str,
        default=None,
        help="Groq API key (or set GROQ_API_KEY env var).",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    video_path: Path = args.video
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # 1. Cari Metadata Mentah JSON dari Clipper
    json_path = video_path.with_suffix(".json")
    if not json_path.exists():
        print(f"Error: Context file not found: {json_path}")
        print("Pastikan video dirender dengan main.py terbaru yang menyimpan file context .json.")
        sys.exit(1)

    print(f"Loading context from {json_path.name}...")
    with open(json_path, "r", encoding="utf-8") as f:
        context = json.load(f)

    # Susun ulang teks untuk LLM
    hook_text = context.get("hook", "")
    transcript = context.get("transcript", "")
    clip_text = ""
    if hook_text:
        clip_text += f"HOOK: {hook_text}\n\n"
    clip_text += f"TRANSCRIPT:\n{transcript}"

    if not clip_text.strip():
        print("Error: No text found in context file to generate metadata.")
        sys.exit(1)

    # 2. Configure API & Generate SEO Metadata
    print("\n[metadata] Menghubungi Groq untuk algoritma Judul & Tags...")
    lang = context.get("language")
    config = Config(
        groq_api_key=args.groq_api_key,
        language=lang,
        upload_enabled=True, # Force set true for manual
    )
    
    metadata = generate_youtube_metadata(clip_text, config)
    if not metadata:
        print("Error: Gagal generate metadata dari LLM.")
        sys.exit(1)

    print("\n── YouTube Metadata ─────────────────────────────────")
    print(f"Title : {metadata.get('title')}")
    desc = metadata.get('description', '')
    # Tampilkan preview deskripsi (max 80 chars)
    print(f"Desc  : {desc[:80]}...")
    print(f"Tags  : {', '.join(metadata.get('tags', []))}")
    print("─────────────────────────────────────────────────────\n")

    # 3. Konfirmasi Upload
    ans = input(f"Yakin ingin upload '{video_path.name}' ke YouTube? (y/n): ")
    if ans.lower() != 'y':
        print("Upload dibatalkan.")
        sys.exit(0)

    # 4. Upload ke Youtube
    success = upload_to_youtube(video_path, metadata, config)
    if success:
        # Pindahkan atau hapus file json & mp4? (Biarkan saja sementara)
        print("\nSelesai! ✨")
    else:
        print("\nUpload gagal atau dibatalkan.")
        sys.exit(1)


if __name__ == "__main__":
    main()
