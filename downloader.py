import argparse
import sys
from pathlib import Path
from pytubefix import YouTube
from pytubefix.cli import on_progress

from config import Config

def download_video(url: str, config: Config) -> Path:
    """
    Download YouTube video using pytubefix, returning the path to the downloaded file.
    """
    config.ensure_dirs()
    
    print(f"Fetching video info for: {url}")
    try:
        yt = YouTube(url, on_progress_callback=on_progress)
    except Exception as e:
        print(f"Error fetching YouTube video: {e}")
        sys.exit(1)
        
    print(f"\nTitle: {yt.title}")
    print(f"Channel: {yt.author}")
    print(f"Length: {yt.length} seconds")
    
    import subprocess
    import imageio_ffmpeg
    import os

    # Select progressive stream based on config
    # Pytubefix highest progressive stream is usually 720p
    video_stream = None
    is_adaptive = False

    if config.download_resolution == "highest":
        video_stream = yt.streams.get_highest_resolution()
    else:
        # Tries to match progressive first
        video_stream = yt.streams.filter(progressive=True, subtype="mp4", resolution=config.download_resolution).first()
        
        # If no progressive found (e.g., 1080p), search adaptive streams
        if not video_stream:
            video_stream = yt.streams.filter(adaptive=True, subtype="mp4", resolution=config.download_resolution).first()
            if video_stream:
                is_adaptive = True
            else:
                print(f"Warning: Resolution {config.download_resolution} not found. Falling back to highest resolution progressive stream.")
                video_stream = yt.streams.get_highest_resolution()
            
    if not video_stream:
        print("Error: Could not find a suitable mp4 stream.")
        sys.exit(1)

    print(f"\nDownloading video stream: {video_stream.resolution} ({video_stream.filesize_mb:.1f} MB)")
    
    if not is_adaptive:
        # Just download progressive
        out_path = video_stream.download(output_path=str(config.vid_dir))
    else:
        # Download adaptive video and audio, then mux
        audio_stream = yt.streams.get_audio_only(subtype="mp4")
        if not audio_stream:
            print("Error: Could not find an audio stream.")
            sys.exit(1)
            
        print(f"Downloading audio stream: ({audio_stream.filesize_mb:.1f} MB)")
        vid_file = video_stream.download(output_path=str(config.vid_dir), filename_prefix="vid_")
        aud_file = audio_stream.download(output_path=str(config.vid_dir), filename_prefix="aud_")
        
        # Determine final output path
        from urllib.parse import unquote
        import re
        safe_title = re.sub(r'[\\/*?:"<>|]', "", yt.title)
        out_path = str(config.vid_dir / f"{safe_title}.mp4")
        
        # Mux with FFmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"\nMuxing video and audio with FFmpeg...")
        cmd = [
            ffmpeg_exe, "-y",
            "-i", vid_file,
            "-i", aud_file,
            "-c:v", "copy",
            "-c:a", "aac",
            out_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            print("Error: FFmpeg muxing failed.")
            sys.exit(1)
            
        # Clean up temporary adaptive files
        try:
            os.remove(vid_file)
            os.remove(aud_file)
        except OSError as e:
            print(f"Warning: Failed to delete temp files: {e}")
            
    print(f"\nDownload completed: {out_path}")
    return Path(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a YouTube video for auto-clipper.")
    parser.add_argument("url", type=str, help="YouTube video URL")
    
    args = parser.parse_args()
    cfg = Config()
    
    download_video(args.url, cfg)
