"""
Hook Generator — niche-adaptive hook sentence + TTS voiceover.

Uses the active ContentProfile for framework selection and prompt context.
Uses Groq LLM + edge-tts for voiceover generation.
"""
import asyncio
import logging
from pathlib import Path
import edge_tts
from groq import Groq

from pipeline.content_profiles import get_profile, ContentProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Framework selection — now profile-driven
# ---------------------------------------------------------------------------

def _pick_framework(clip_text: str, profile: ContentProfile) -> dict:
    """Auto-select the best hook framework based on signal keywords."""
    lowered = clip_text.lower()
    for framework_name, signals in profile.hook_signals.items():
        if any(s in lowered for s in signals):
            match = next(
                (f for f in profile.hook_frameworks if f["name"] == framework_name),
                None,
            )
            if match:
                return match
    # Default: return the first framework in the profile
    return profile.hook_frameworks[0]


# ---------------------------------------------------------------------------
# Prompt builder — profile-driven niche context
# ---------------------------------------------------------------------------

def _build_prompt(
    clip_text: str,
    lang_instruction: str,
    framework: dict,
    profile: ContentProfile,
) -> str:
    is_indonesian = "Indonesia" in lang_instruction
    examples = framework.get(
        "examples_id" if is_indonesian else "examples_en",
        framework.get("examples_en", []),
    )
    examples_block = "\n".join(f"  - {ex}" for ex in examples)

    return f"""You are a top-performing short-form video editor specializing in {profile.label} content.
Your clips consistently hit 1M+ views because your hooks make people feel like they MUST watch the rest.

---
CLIP TRANSCRIPT:
\"\"\"{clip_text}\"\"\"

---
HOOK FRAMEWORK: {framework["name"].replace("_", " ").upper()}
Rule: {framework["instruction"]}

REFERENCE EXAMPLES (same framework, {lang_instruction}):
{examples_block}

---
NICHE CONTEXT:
- Audience: {profile.audience}
- What stops them from scrolling: feeling they're about to learn or experience something they didn't expect
- Tone: {profile.tone}

CONSTRAINTS (NON-NEGOTIABLE):
- Language: {lang_instruction} only
- Length: 5–9 words maximum
- NO filler words (like, just, very, really, so)
- NO questions — statements hit harder
- Output: the hook sentence ONLY — no quotes, no hashtags, no explanation

Write the hook now:"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_hook_text(
    clip_text: str,
    api_key: str | None = None,
    language: str | None = None,
    framework_name: str | None = None,
    content_niche: str = "money",
) -> str:
    """
    Generate a hook sentence for a clip, using the niche-appropriate frameworks.

    Args:
        clip_text:      Transcript text from the video clip.
        api_key:        Groq API key. Falls back to GROQ_API_KEY env var.
        language:       Language code — "id" (default) or "en".
        framework_name: Override auto-selection with a specific framework name.
        content_niche:  Content niche key (e.g., "money", "tech", "gaming").

    Returns:
        Hook sentence string, or "" on failure.
    """
    if not api_key:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        logger.warning("[hook] No Groq API key. Skipping.")
        return ""

    if not clip_text or not clip_text.strip():
        logger.warning("[hook] Empty clip text. Skipping.")
        return ""

    # Load niche profile
    profile = get_profile(content_niche)

    lang_instruction = "bahasa Indonesia"
    if language and "en" in language.lower():
        lang_instruction = "English"

    if framework_name:
        framework = next(
            (f for f in profile.hook_frameworks if f["name"] == framework_name),
            _pick_framework(clip_text, profile),
        )
    else:
        framework = _pick_framework(clip_text, profile)

    logger.info(f"[hook] Niche: {profile.niche} | Framework: {framework['name']} | Lang: {lang_instruction}")
    prompt = _build_prompt(clip_text, lang_instruction, framework, profile)

    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.85,
            max_tokens=48,
            stop=["\n", ".", "!"],
        )
        hook = (
            response.choices[0].message.content
            .strip().strip('"').strip("'").strip()
        )
        logger.info(f"[hook] Result: '{hook}'")
        return hook

    except Exception as e:
        logger.error(f"[hook] Groq error: {e}")
        return ""


def generate_hook_audio(hook_text: str, output_path: Path, voice_id: str) -> Path:
    if not hook_text:
        logger.warning("[hook] Empty hook text — skipping audio.")
        return output_path

    async def _generate():
        communicate = edge_tts.Communicate(hook_text, voice_id)
        await communicate.save(str(output_path))

    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pool.submit(asyncio.run, _generate()).result()
    except RuntimeError:
        asyncio.run(_generate())

    logger.info(f"[hook] Audio saved: {output_path}")
    return output_path