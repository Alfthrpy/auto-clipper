"""
YouTube Uploader — Authenticate and auto-upload short videos via YouTube Data API v3.
Requires 'client_secrets.json' from Google Cloud Console.
"""
import os
import logging
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from config import Config

logger = logging.getLogger(__name__)

# Oauth scopes for uploading videos
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def _get_youtube_service():
    """Lakukan otentikasi ke Google API dan kembalikan service object."""
    creds = None
    
    # Path ke file secrets bawaan Google Cloud Console
    client_secrets_file = "client_secrets.json"
    token_file = "token.json"
    
    # Cek apakah sudah pernah login
    if os.path.exists(token_file):
        try:
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)
        except Exception as e:
            logger.warning(f"[uploader] Invalid token file: {e}")
            
    # Jika tidak valid / belum pernah login, minta user login via Browser
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.error(f"[uploader] Coudln't refresh token: {e}")
                creds = None
                
        if not creds:
            if not os.path.exists(client_secrets_file):
                logger.error(f"[uploader] ERROR: File '{client_secrets_file}' tidak ditemukan!")
                logger.error("Silakan ikuti instruksi di Google Cloud Console untuk mendownload OAuth Client ID.")
                return None
                
            logger.info("[uploader] Meminta Otorisasi YouTube melalui Browser...")
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
            creds = flow.run_local_server(port=0)
            
        # Simpan creds untuk eksekusi selanjutnya
        with open(token_file, "w") as token:
            token.write(creds.to_json())

    return build("youtube", "v3", credentials=creds)


def upload_to_youtube(
    video_path: Path,
    metadata: dict,
    config: Config
) -> bool:
    """
    Eksekusi HTTP Resumable Upload ke YouTube khusus format Shorts.
    """
    if not config.upload_enabled:
        return False
        
    youtube = _get_youtube_service()
    if not youtube:
        return False

    title = metadata.get("title", f"Auto-Clipper Short {video_path.stem}")
    description = metadata.get("description", "Dibuat otomatis oleh AutoClipper AI.")
    
    # Supaya otomatis ke Shorts, masukkan #shorts dkk ke ujung deskripsi
    tags = metadata.get("tags", [])
    if "shorts" not in [t.lower() for t in tags]:
        tags.append("shorts")
    
    hashtags_str = " ".join([f"#{t.replace(' ', '')}" for t in tags])
    final_desc = f"{description}\n\n{hashtags_str}"

    print(f"\n[uploader] 🚀 Persiapan Upload: {video_path.name}")
    print(f"  ↳ Judul: {title}\n  ↳ Desc: {final_desc.splitlines()[0]}...")
    
    body = {
        "snippet": {
            "title": title,
            "description": final_desc,
            "tags": tags,
            "categoryId": config.youtube_category_id
        },
        "status": {
            "privacyStatus": config.youtube_privacy_status,
            "selfDeclaredMadeForKids": False
        }
    }

    # Upload Media file
    try:
        media = MediaFileUpload(
            str(video_path), 
            mimetype='video/mp4',
            chunksize=-1,
            resumable=True
        )
        
        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media
        )

        response = None
        # Eksekusi Upload dengan progres tracking
        print("[uploader] Mengunggah video...")
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"[uploader] Progres: {int(status.progress() * 100)}%", end="\r")

        print(f"\n[uploader] ✅ Upload Berhasil! Video ID: {response.get('id')}")
        print(f"Cek di: https://youtu.be/{response.get('id')}")
        return True
        
    except Exception as e:
        logger.error(f"[uploader] Upload Gagal: {e}")
        return False
