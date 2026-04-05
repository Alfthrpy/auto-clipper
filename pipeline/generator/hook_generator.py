"""
Hook Generator — Money / Business / Smart Insight niche.
Uses Groq LLM + edge-tts for voiceover generation.
"""
import asyncio
import logging
from pathlib import Path
import edge_tts
from groq import Groq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hook frameworks — tuned specifically for Money / Business / Smart Insight.
# ---------------------------------------------------------------------------
HOOK_FRAMEWORKS = [
    {
        "name": "wealth_revelation",
        "instruction": (
            "Reveal a financial truth, money mechanic, or economic insight that most "
            "people are completely unaware of. Frame it as something hidden, suppressed, "
            "or overlooked — not taught in school, not discussed in mainstream media."
        ),
        "examples_id": [
            "Bank tidak pernah ngajarin ini ke kamu",
            "Cara orang kaya nyimpen uang beda banget",
            "Inflasi diam-diam nyuri kekayaanmu tiap hari",
            "Ini yang bikin gaji besar tapi tetap bokek",
        ],
        "examples_en": [
            "Banks will never tell you this exists",
            "The wealthy use a money rule no one teaches",
            "Your savings account is quietly draining you",
            "This is why high earners still go broke",
        ],
    },
    {
        "name": "contrarian_finance",
        "instruction": (
            "Challenge a widely-held belief about money, success, or business that "
            "most people treat as gospel. The take must be bold and defensible — "
            "not just provocative for its own sake."
        ),
        "examples_id": [
            "Rajin nabung justru bikin kamu miskin",
            "Kerja keras bukan cara jadi kaya",
            "Utang itu bukan musuh — salah paham terbesar soal duit",
            "Beli rumah sekarang bisa jadi keputusan terburukmu",
        ],
        "examples_en": [
            "Saving money is keeping you poor",
            "Hard work alone has never made anyone wealthy",
            "Debt is not your enemy — you've been lied to",
            "Buying a house right now might ruin you financially",
        ],
    },
    {
        "name": "system_expose",
        "instruction": (
            "Expose how an economic system, financial institution, or business structure "
            "is designed in a way that disadvantages ordinary people — or advantages "
            "those who understand it. Make it feel like pulling back a curtain."
        ),
        "examples_id": [
            "Sistem ini dirancang biar kamu tetap butuh gaji",
            "Kenapa pajak orang kaya lebih kecil dari kamu",
            "Cara perusahaan besar bayar nol pajak secara legal",
            "Ini kenapa harga rumah terus naik dan sengaja dibiarkan",
        ],
        "examples_en": [
            "The system is designed to keep you employed forever",
            "Why the wealthy pay less tax than you legally",
            "How billion-dollar companies pay zero tax — legally",
            "Housing prices don't rise by accident — here's why",
        ],
    },
    {
        "name": "insight_urgency",
        "instruction": (
            "Create a sense that the viewer is currently missing out, making a costly "
            "mistake, or losing money right now due to lack of this specific knowledge. "
            "The urgency must be grounded in real economic or financial stakes."
        ),
        "examples_id": [
            "Tiap bulan kamu tunda ini, kerugianmu makin besar",
            "Keputusan finansial ini bisa beda hasilnya 20 tahun lagi",
            "Orang-orang yang skip ini nyesel 5 tahun kemudian",
            "Satu kesalahan ini bisa biaya kamu ratusan juta",
        ],
        "examples_en": [
            "Every month you delay this costs you more",
            "This one decision separates you 20 years from now",
            "People who ignored this regretted it within five years",
            "One financial mistake quietly costing you millions",
        ],
    },
]

_SIGNALS = {
    "contrarian_finance": [
        "saving", "save", "work hard", "hustle", "buy house", "property",
        "nabung", "kerja keras", "beli rumah", "investasi saham", "kripto",
    ],
    "system_expose": [
        "tax", "bank", "government", "corporation", "institution", "policy",
        "pajak", "pemerintah", "korporasi", "sistem", "regulasi",
    ],
    "insight_urgency": [
        "delay", "late", "miss", "regret", "opportunity", "compound",
        "terlambat", "rugi", "nyesel", "kesempatan", "bunga",
    ],
}


def _pick_framework(clip_text: str) -> dict:
    lowered = clip_text.lower()
    for framework_name, signals in _SIGNALS.items():
        if any(s in lowered for s in signals):
            return next(f for f in HOOK_FRAMEWORKS if f["name"] == framework_name)
    # Default: wealth_revelation is the highest-performing hook type for this niche
    return next(f for f in HOOK_FRAMEWORKS if f["name"] == "wealth_revelation")


def _build_prompt(clip_text: str, lang_instruction: str, framework: dict) -> str:
    is_indonesian = "Indonesia" in lang_instruction
    examples = framework["examples_id"] if is_indonesian else framework["examples_en"]
    examples_block = "\n".join(f"  - {ex}" for ex in examples)

    return f"""You are a top-performing short-form video editor specializing in Money, Business, and Financial Insight content.
Your clips consistently hit 1M+ views because your hooks make people feel like they're about to learn something that will change their financial life.

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
- Audience: people who want to understand money, build wealth, or navigate the economy smarter
- What stops them from scrolling: feeling they're about to learn something valuable they didn't know
- Tone: sharp, confident, slightly provocative — NOT motivational fluff or generic advice

CONSTRAINTS (NON-NEGOTIABLE):
- Language: {lang_instruction} only
- Length: 5–9 words maximum
- NO filler words (like, just, very, really, so)
- NO questions — statements hit harder in this niche
- Output: the hook sentence ONLY — no quotes, no hashtags, no explanation

Write the hook now:"""


def generate_hook_text(
    clip_text: str,
    api_key: str | None = None,
    language: str | None = None,
    framework_name: str | None = None,
) -> str:
    """
    Generate a hook sentence for a Money/Business/Insight clip.

    Args:
        clip_text:      Transcript text from the video clip.
        api_key:        Groq API key. Falls back to GROQ_API_KEY env var.
        language:       Language code — "id" (default) or "en".
        framework_name: Override auto-selection with a specific framework name.

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

    lang_instruction = "bahasa Indonesia"
    if language and "en" in language.lower():
        lang_instruction = "English"

    if framework_name:
        framework = next(
            (f for f in HOOK_FRAMEWORKS if f["name"] == framework_name),
            _pick_framework(clip_text),
        )
    else:
        framework = _pick_framework(clip_text)

    logger.info(f"[hook] Framework: {framework['name']} | Lang: {lang_instruction}")
    prompt = _build_prompt(clip_text, lang_instruction, framework)

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