"""
Hook Generator — Use Groq to generate a catchy hook, and edge-tts to generate voice over.
"""
import asyncio
import logging
import random
from pathlib import Path
import edge_tts
from groq import Groq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hook framework definitions
# Each entry has a name, a one-line instruction, and bilingual few-shot examples.
# ---------------------------------------------------------------------------
HOOK_FRAMEWORKS = [
    {
        "name": "curiosity_gap",
        "instruction": (
            "Open a curiosity gap — reveal just enough to make the viewer desperate "
            "to know what happens next. Tease the *outcome* without spoiling it."
        ),
        "examples_id": [
            "Cara ini bikin dia diam dalam 3 detik.",
            "Ternyata yang dia sembunyikan lebih parah dari itu.",
            "Semua orang salah tentang satu hal ini.",
        ],
        "examples_en": [
            "Nobody warned me this would happen.",
            "This changes everything you thought you knew.",
            "The part they cut out is the most important.",
        ],
    },
    {
        "name": "shock_or_controversy",
        "instruction": (
            "State something counterintuitive, taboo, or boldly controversial that "
            "challenges a widely-held belief. Make it feel like a confession or hot take."
        ),
        "examples_id": [
            "Makin rajin nabung, makin miskin — ini faktanya.",
            "Dokter nggak mau kamu tahu hal ini.",
            "Sekolah tinggi bukan jaminan sukses — buktinya ada di sini.",
        ],
        "examples_en": [
            "Working harder is actually making you broke.",
            "The advice everyone gives is completely wrong.",
            "I got fired — best thing that ever happened.",
        ],
    },
    {
        "name": "relatability_pain",
        "instruction": (
            "Call out a specific frustration or embarrassing situation your audience "
            "has silently experienced. Make them feel seen and understood instantly."
        ),
        "examples_id": [
            "Capek kerja keras tapi tetap nggak cukup?",
            "Udah ngejelasin berkali-kali tapi tetap nggak dimengerti.",
            "Pernah ngerasa usahamu nggak pernah dihargai?",
        ],
        "examples_en": [
            "Tired of doing everything right and getting nothing?",
            "You've felt this but were too scared to say it.",
            "Everyone around you is moving forward except you.",
        ],
    },
    {
        "name": "urgency_stakes",
        "instruction": (
            "Raise the stakes immediately — make the viewer feel they will lose something "
            "valuable or make a costly mistake if they stop watching."
        ),
        "examples_id": [
            "Jangan buka aplikasi itu sebelum nonton ini.",
            "Kalau kamu di posisi ini, stop dulu sekarang.",
            "Kesalahan ini bisa biaya kamu jutaan rupiah.",
        ],
        "examples_en": [
            "Stop what you're doing and watch this first.",
            "This mistake is costing you every single day.",
            "You have 24 hours before this stops working.",
        ],
    },
]


def _pick_framework(clip_text: str) -> dict:
    """
    Heuristic sederhana untuk memilih framework berdasarkan konten clip.
    Bisa diganti dengan classifier jika dataset tersedia.
    """
    lowered = clip_text.lower()

    controversy_signals = ["wrong", "myth", "lie", "secret", "salah", "bohong", "rahasia", "mitos"]
    pain_signals = ["struggle", "hard", "tired", "stress", "susah", "capek", "lelah", "gagal"]
    urgency_signals = ["limited", "deadline", "now", "today", "sekarang", "segera", "batas"]

    if any(w in lowered for w in controversy_signals):
        return next(f for f in HOOK_FRAMEWORKS if f["name"] == "shock_or_controversy")
    if any(w in lowered for w in pain_signals):
        return next(f for f in HOOK_FRAMEWORKS if f["name"] == "relatability_pain")
    if any(w in lowered for w in urgency_signals):
        return next(f for f in HOOK_FRAMEWORKS if f["name"] == "urgency_stakes")

    # Default: curiosity gap — works well as a universal fallback
    return next(f for f in HOOK_FRAMEWORKS if f["name"] == "curiosity_gap")


def _build_prompt(clip_text: str, lang_instruction: str, framework: dict) -> str:
    """
    Bangun prompt dengan framework, examples, dan constraints yang jelas.
    Menggunakan struktur: Role → Context → Framework → Examples → Constraints → Task.
    """
    is_indonesian = "Indonesia" in lang_instruction
    examples = framework["examples_id"] if is_indonesian else framework["examples_en"]
    examples_block = "\n".join(f"  - {ex}" for ex in examples)

    return f"""You are an elite short-form video editor who has cracked the algorithm on TikTok, Reels, and YouTube Shorts.
Your single job right now is to write the opening hook line for a clip.

---
CLIP TRANSCRIPT:
\"\"\"{clip_text}\"\"\"

---
HOOK FRAMEWORK TO USE: {framework["name"].replace("_", " ").upper()}
Rule: {framework["instruction"]}

REFERENCE EXAMPLES (same framework, {lang_instruction}):
{examples_block}

---
CONSTRAINTS (NON-NEGOTIABLE):
- Language: {lang_instruction} only
- Length: 4–9 words maximum — every word must earn its place
- Tone: direct, punchy, zero filler words
- Output: the hook sentence ONLY — no quotes, no hashtags, no explanation, no period at the end

Write the hook now:"""


def generate_hook_text(
    clip_text: str,
    api_key: str | None = None,
    language: str | None = None,
    framework_name: str | None = None,
) -> str:
    """
    Kirim teks dari clip ke LLM untuk mendapatkan kalimat Hook bombastis.

    Args:
        clip_text:      Transkrip teks dari klip video.
        api_key:        Groq API key. Jika None, dicari dari env GROQ_API_KEY.
        language:       Kode bahasa, mis. "id" atau "en". Default "id".
        framework_name: Nama framework hook (curiosity_gap, shock_or_controversy,
                        relatability_pain, urgency_stakes). Jika None, dipilih otomatis.

    Returns:
        String kalimat hook, atau "" jika gagal.
    """
    if not api_key:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        logger.warning("[hook] No Groq API Key found. Hook text generation skipped.")
        return ""

    if not clip_text or not clip_text.strip():
        logger.warning("[hook] Empty clip text. Skipping hook generation.")
        return ""

    # --- Language ---
    lang_instruction = "bahasa Indonesia"
    if language and "en" in language.lower():
        lang_instruction = "English"

    # --- Framework selection ---
    if framework_name:
        framework = next(
            (f for f in HOOK_FRAMEWORKS if f["name"] == framework_name),
            _pick_framework(clip_text),
        )
    else:
        framework = _pick_framework(clip_text)

    logger.info(f"[hook] Using framework: {framework['name']}")

    # --- Build prompt ---
    prompt = _build_prompt(clip_text, lang_instruction, framework)

    # --- Call LLM ---
    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",   # 70B > 8B untuk hook quality
            temperature=0.85,                   # Sedikit lebih tinggi untuk variasi kreatif
            max_tokens=48,                      # Hook pendek — tidak perlu budget besar
            stop=["\n", ".", "!"],              # Stop di akhir kalimat pertama
        )
        hook = (
            response.choices[0].message.content
            .strip()
            .strip('"')
            .strip("'")
            .strip()
        )
        logger.info(f"[hook] Generated: '{hook}' (framework: {framework['name']})")
        return hook

    except Exception as e:
        logger.error(f"[hook] Error calling Groq: {e}")
        return ""


def generate_hook_audio(hook_text: str, output_path: Path, voice_id: str) -> Path:
    """
    Gunakan edge-tts untuk menggenerate file audio dari teks hook.

    Returns:
        Path ke file audio yang dihasilkan.
    """
    if not hook_text:
        logger.warning("[hook] Empty hook text — skipping audio generation.")
        return output_path

    async def _generate():
        communicate = edge_tts.Communicate(hook_text, voice_id)
        await communicate.save(str(output_path))

    try:
        loop = asyncio.get_running_loop()
        # Sudah ada event loop (mis. di Jupyter / async pipeline)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pool.submit(asyncio.run, _generate()).result()
    except RuntimeError:
        # Tidak ada running loop — jalur normal untuk pipeline sinkron
        asyncio.run(_generate())

    logger.info(f"[hook] Audio saved to: {output_path}")
    return output_path