"""
Metadata Generator — YouTube Shorts Title, Description, and Tags.
Niche-tuned for Money / Business / Smart Financial Insight content.
Uses JSON Mode on Groq API.
"""
import json
import logging
import os
import time
from groq import Groq, RateLimitError

from config import Config
from dotenv import load_dotenv
load_dotenv()


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tone and Anti-AI Instructions
# ---------------------------------------------------------------------------
ANTI_AI_INSTRUCTIONS = """
TONE & STYLE (CRITICAL):
- Write like a real, experienced human YouTuber. Be raw, conversational, and direct.
- DO NOT sound like a corporate robot or an AI.
- BANNED WORDS/CONCEPTS: Avoid typical AI buzzwords (e.g., "Unlock", "Discover", "Delve into", "Revolutionize", "Rahasia Tersembunyi", "Wajib Tahu", "Mengungkap").
- EMOJI RULE: Use STRICTLY 0 or 1 emoji across the entire metadata. Real creators often use NO emojis for a more serious, authentic vibe. Never use typical AI emojis like 🚀, 💡, 🔥, or 🧠.
"""

# ---------------------------------------------------------------------------
# Title patterns - Humanized and raw
# ---------------------------------------------------------------------------
TITLE_PATTERNS = """
Choose ONE of these proven title patterns. Adapt the meaning into the target language using a casual, human tone:

  A) CONTRARIAN STATEMENT
     Pattern: "[Common belief] is actually [making you poor/a scam]"
     (e.g., "Nabung di bank itu aslinya bikin miskin")
  
  B) HIDDEN MECHANISM
     Pattern: "How [Banks/System] actually make money off you"
     (e.g., "Cara bank muterin uang gaji lo tiap bulan")
  
  C) CURIOSITY GAP (Punchy & Short)
     Pattern: "Why you should never [do normal financial thing]"
     (e.g., "Kenapa gue stop nabung buat dana darurat")
  
  D) BLUNT TRUTH
     Pattern: "The brutal truth about [Topic]"
     (e.g., "Realita pahit soal passive income")
"""

# ---------------------------------------------------------------------------
# Tag taxonomy
# ---------------------------------------------------------------------------
TAG_STRATEGY = """
Generate exactly 15 tags using this layered SEO strategy, translated to the target language:

  BROAD (3 tags) — generic terms (e.g., uang, finansial).
  MID-TIER (5 tags) — topic-specific terms matching the clip content.
  LONG-TAIL (4 tags) — 3-5 word phrases, conversational (e.g., "cara ngatur duit gaji").
  NICHE VIRAL (3 tags) — trending terms.

Return tags as a flat JSON array of strings. No hashtag symbols (#).
"""

# ---------------------------------------------------------------------------
# Description structure - Casual and brief
# ---------------------------------------------------------------------------
DESCRIPTION_STRUCTURE = """
Write the description in this exact structure, translated to the target language:

  LINE 1 — THE HOOK (1 sentence):
    A casual, blunt statement about the video's core message. Make it sound like you're texting a friend a harsh truth. 

  LINE 2 — CONTEXT (1-2 sentences):
    Explain briefly what the video breaks down. No fluff, no formal conclusions.

  LINE 3 — CTA (1 short sentence):
    Very casual call to action (e.g., "Subscribe buat insight realistis lainnya.")

  LINE 4 — HASHTAGS (exactly 3):
    Keep it minimal. 2 content hashtags + #Shorts.
"""

def _build_prompts(clip_text: str, lang_instruction: str) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) pair."""
    system_prompt = (
        "You are an elite YouTube Shorts SEO strategist specializing in "
        "Money and Business content. You know how to write viral metadata that feels 100% human.\n\n"
        "You MUST respond ONLY with a valid JSON object. No markdown, no extra keys.\n\n"
        f"CRITICAL LANGUAGE RULE:\n"
        f"You MUST write ALL output fields entirely in {lang_instruction}. "
        f"Use natural, conversational, and slightly informal phrasing native to that language.\n\n"
        + ANTI_AI_INSTRUCTIONS
        + "\n--- TITLE INSTRUCTIONS ---\n"
        + TITLE_PATTERNS
        + "\nAdditional title rules:\n"
        "  - Maximum 55 characters (shorter is more human).\n"
        "  - NO title case (Don't Capitalize Every Word). Use standard sentence case or all lowercase for a raw aesthetic.\n\n"
        "--- DESCRIPTION INSTRUCTIONS ---\n"
        + DESCRIPTION_STRUCTURE
        + "\n--- TAG INSTRUCTIONS ---\n"
        + TAG_STRATEGY
        + "\n\nOUTPUT SCHEMA (follow exactly):\n"
        '{"title": "<string>", "description": "<string>", "tags": ["<string>", ...]}'
    )

    user_prompt = (
        f"TARGET LANGUAGE: {lang_instruction}\n\n"
        f"Generate human-like, non-spammy YouTube Shorts metadata for this video transcript:\n\n{clip_text}"
    )

    return system_prompt, user_prompt


def _validate_metadata(metadata: dict) -> dict:
    """
    Validate and sanitize LLM output.
    Returns cleaned metadata or raises ValueError if critically malformed.
    """
    required_keys = {"title", "description", "tags"}
    if not required_keys.issubset(metadata.keys()):
        missing = required_keys - metadata.keys()
        raise ValueError(f"Missing required keys: {missing}")

    # Title: enforce character limit and strip leading hashtags
    title = str(metadata["title"]).strip().lstrip("#").strip()
    if len(title) > 65:  
        logger.warning(f"[metadata] Title too long ({len(title)} chars), truncating.")
        title = title[:60].rstrip()
    metadata["title"] = title

    # Description: ensure it's a string
    metadata["description"] = str(metadata["description"]).strip()

    # Tags: ensure it's a list of strings, strip # symbols
    tags = metadata.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).lstrip("#").strip() for t in tags if t]
    if len(tags) < 5:
        logger.warning(f"[metadata] Only {len(tags)} tags returned.")
    metadata["tags"] = tags

    return metadata


def generate_youtube_metadata(
    clip_text: str,
    config: Config,
    max_retries: int = 3,
) -> dict:
    """
    Generate YouTube Shorts metadata (title, description, tags) from a clip transcript.
    """
    api_key = config.groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("[metadata] No Groq API key. Skipping metadata generation.")
        return {}

    if not clip_text or not clip_text.strip():
        logger.warning("[metadata] Empty clip text. Skipping.")
        return {}

    # Define language clearly
    lang_instruction = "English" if getattr(config, "language", "id") == "en" else "Indonesian (Bahasa Indonesia - gunakan gaya bahasa kasual, sedikit santai, bukan bahasa baku formal)"
    
    model_name = getattr(config, "metadata_model", None) or getattr(config, "semantic_model", "llama-3.1-8b-instant")

    client = Groq(api_key=api_key)
    system_prompt, user_prompt = _build_prompts(clip_text, lang_instruction)

    delay = 2.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                model=model_name,
                temperature=0.7, # Sedikit diturunkan agar tidak terlalu "halu" dan tetap sesuai struktur
                max_tokens=800,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            metadata = json.loads(raw)
            metadata = _validate_metadata(metadata)

            logger.info(
                f"[metadata] ✓ Title: '{metadata['title']}' | "
                f"{len(metadata['tags'])} tags generated"
            )
            return metadata

        except RateLimitError:
            logger.warning(f"[metadata] Rate limited on attempt {attempt + 1}. Waiting {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[metadata] Parse/validation error on attempt {attempt + 1}: {e}")
            time.sleep(1.0)

        except Exception as e:
            logger.warning(f"[metadata] Unexpected error on attempt {attempt + 1}: {e}")
            time.sleep(1.0)

    logger.error("[metadata] Failed to generate metadata after all retries.")
    return {}