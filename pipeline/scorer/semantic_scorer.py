"""
Semantic Scorer — Money / Business / Smart Insight niche.
Evaluates chunk quality on dimensions relevant to financial/business content.
Uses JSON Mode batched inferencing on Groq to preserve free-tier limits.
"""
import json
import logging
import time
import numpy as np
from groq import Groq, RateLimitError, InternalServerError, APIConnectionError

from config import Config
from pipeline.processor.segmenter import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default blend weights for Money / Business / Smart Insight niche.
#
# Rationale:
#   insight_depth   0.45 — audience subscribes FOR the knowledge, this is the core value
#   viral_potential 0.30 — still matters for reach, but secondary to substance
#   contrarian_edge 0.25 — bold takes drive comments/shares; important but can't dominate
#
# Compare to generic entertainment niche (viral 0.45 / insight 0.30 / controversy 0.25)
# — the weights are intentionally flipped to reflect this niche's audience expectation.
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "insight_depth":   0.45,
    "viral_potential": 0.30,
    "contrarian_edge": 0.25,
}

# ---------------------------------------------------------------------------
# Scoring rubric with money/business-specific anchor examples.
# Anchor points prevent score drift across batches.
# ---------------------------------------------------------------------------
SCORING_RUBRIC = """
NICHE CONTEXT: Money, Business, and Smart Financial/Economic Insight content.
Score each segment from 0–10 using these anchor points:

[insight_depth] — How valuable and substantive is the financial/business knowledge here?
  0  : Zero substance. Empty motivation, vague platitudes, or filler talk.
  3  : Surface-level. Common knowledge most people already have (e.g., "spend less than you earn").
  5  : Decent point. Useful but not surprising — something you'd find in any finance article.
  7  : Genuinely valuable. Specific mechanism, surprising statistic, or non-obvious insight.
  10 : Mind-opening. Reveals how money/systems actually work in a way most people never knew.
       Example: explaining how inflation is a hidden tax, or how compound interest works against debtors.

[viral_potential] — How likely is this to make a money-conscious viewer stop scrolling and share?
  0  : Completely forgettable. No emotional pull, no tension, nothing to react to.
  3  : Mildly interesting but no reason to share or save.
  5  : Solid content. Worth watching but won't trigger a strong share impulse.
  7  : Strong hook potential. Has a clear "wait, what?" or "I need to show this to someone" moment.
  10 : Explosive stop-scroll energy. Triggers immediate emotional reaction (anger, shock, or revelation).
       Example: a claim that directly contradicts what most people do with their money.

[contrarian_edge] — Does it challenge mainstream financial advice or conventional wisdom?
  0  : Completely safe and agreeable. Nobody would push back.
  3  : Slightly opinionated but not really debate-worthy.
  5  : Takes a clear stance — some people would agree, some would disagree.
  7  : Bold financial take. Challenges popular advice (buying a house, saving in a bank, etc.)
  10 : Polarizing. Will generate passionate comments from both "finally someone said it" and "this is wrong".
       Example: "Mutual funds are designed to make fund managers rich, not you."
"""

FEW_SHOT_EXAMPLE = """
INPUT EXAMPLE:
[{"id": 42, "text": "Most people think a savings account protects their money, but with inflation running at 5% and your bank paying 0.5% interest, you're actually losing purchasing power every single year. The bank profits from lending your money at 10% while paying you almost nothing."}]

OUTPUT EXAMPLE:
{"results": [{"id": 42, "insight_depth": 8.5, "viral_potential": 7.5, "contrarian_edge": 7.0}]}
"""


def _normalize(arr: np.ndarray) -> np.ndarray:
    if len(arr) == 0:
        return arr
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _truncate_at_sentence(text: str, max_chars: int = 800) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for sep in (".", "!", "?", "\n"):
        last_idx = truncated.rfind(sep)
        if last_idx > max_chars * 0.5:
            return truncated[: last_idx + 1].strip()
    return truncated.strip()


def _build_batch_prompt(chunk_data: list[dict]) -> tuple[str, str]:
    system_prompt = (
        "You are an expert content analyst for Money, Business, and Financial Insight short-form video.\n"
        "Your job is to score video transcript segments on three dimensions relevant to this niche.\n"
        "You MUST respond with valid JSON only — no explanation, no markdown, no extra keys.\n\n"
        + SCORING_RUBRIC
        + "\n"
        + FEW_SHOT_EXAMPLE
    )

    user_prompt = (
        "Score each segment below using the rubric. "
        "Return ONLY a JSON object matching this schema exactly:\n"
        '{"results": [{"id": <int>, "insight_depth": <float 0-10>, "viral_potential": <float 0-10>, "contrarian_edge": <float 0-10>}]}\n\n'
        f"SEGMENTS:\n{json.dumps(chunk_data, ensure_ascii=False, indent=2)}"
    )

    return system_prompt, user_prompt


def _call_groq_with_retry(
    client: Groq,
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_retries: int = 5,
) -> dict:
    delay = 2.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
                temperature=0.1,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)

        except RateLimitError as e:
            if attempt == max_retries - 1:
                logger.error(f"[semantic_scorer] Rate limit persists after {max_retries} attempts.")
                raise
            retry_after = getattr(e, "response", None)
            retry_after = (
                float(retry_after.headers.get("retry-after", delay))
                if retry_after else delay
            )
            logger.warning(f"[semantic_scorer] Rate limited. Waiting {retry_after:.1f}s...")
            time.sleep(retry_after)
            delay = min(delay * 2, 60.0)

        except (InternalServerError, APIConnectionError) as e:
            if attempt == max_retries - 1:
                logger.error(f"[semantic_scorer] API error after {max_retries} attempts: {e}")
                raise
            logger.warning(f"[semantic_scorer] Server error, retry {attempt + 1} in {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)

        except json.JSONDecodeError as e:
            logger.error(f"[semantic_scorer] LLM returned invalid JSON: {e}")
            raise

        except Exception as e:
            logger.error(f"[semantic_scorer] Non-retryable error: {e}")
            raise


def _parse_and_blend(item: dict, n_chunks: int, w_ins: float, w_vir: float, w_con: float) -> tuple:
    """Validate, clamp, and blend a single LLM score item."""
    idx = item.get("id")
    if idx is None or not isinstance(idx, int) or not (0 <= idx < n_chunks):
        logger.warning(f"[semantic_scorer] Unexpected id: {idx!r}")
        return None, 0.0

    def _clamp(v):
        return max(0.0, min(10.0, float(v)))

    ins = _clamp(item.get("insight_depth", 0))
    vir = _clamp(item.get("viral_potential", 0))
    con = _clamp(item.get("contrarian_edge", 0))

    blended = (ins * w_ins) + (vir * w_vir) + (con * w_con)
    return idx, blended


def score_semantic_llm(
    chunks: list[Chunk],
    config: Config,
    max_batch_size: int = 10,
) -> np.ndarray:
    """
    Score chunks for Money/Business/Insight content quality using an LLM.

    Scoring dimensions (niche-adjusted):
      - insight_depth:   Depth and value of financial/business knowledge (weight: 0.45)
      - viral_potential: Stop-scroll and shareability for this audience (weight: 0.30)
      - contrarian_edge: Challenges mainstream financial advice (weight: 0.25)

    Weights can be overridden via config.semantic_weights dict.

    Args:
        chunks:         List of Chunk objects.
        config:         Pipeline config with optional groq_api_key and semantic_weights.
        max_batch_size: Chunks per API call. Keep at 10 for reliable JSON output.

    Returns:
        np.ndarray of normalized scores [0.0, 1.0], same length as chunks.
    """
    if not chunks:
        return np.array([])

    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = config.groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("[semantic_scorer] No Groq API key. Returning zeros.")
        return np.zeros(len(chunks))

    # --- Blend weights ---
    weights = {**DEFAULT_WEIGHTS, **getattr(config, "semantic_weights", {})}
    w_ins = float(weights.get("insight_depth",   DEFAULT_WEIGHTS["insight_depth"]))
    w_vir = float(weights.get("viral_potential", DEFAULT_WEIGHTS["viral_potential"]))
    w_con = float(weights.get("contrarian_edge", DEFAULT_WEIGHTS["contrarian_edge"]))
    # Normalize to sum = 1
    w_total = w_ins + w_vir + w_con
    w_ins, w_vir, w_con = w_ins / w_total, w_vir / w_total, w_con / w_total

    model_name = getattr(config, "semantic_model", "llama-3.1-8b-instant")
    client = Groq(api_key=api_key)

    n = len(chunks)
    final_scores = np.zeros(n)
    scored_ids: set[int] = set()
    total_batches = (n + max_batch_size - 1) // max_batch_size

    logger.info(
        f"[semantic_scorer] Scoring {n} chunks | model={model_name} | batch={max_batch_size}\n"
        f"  Weights → insight={w_ins:.2f}  viral={w_vir:.2f}  contrarian={w_con:.2f}"
    )

    for batch_num, i in enumerate(range(0, n, max_batch_size), start=1):
        batch = chunks[i: i + max_batch_size]
        chunk_data = [
            {"id": i + j, "text": _truncate_at_sentence(chunk.text)}
            for j, chunk in enumerate(batch)
        ]

        system_prompt, user_prompt = _build_batch_prompt(chunk_data)

        try:
            logger.info(f"[semantic_scorer] Batch {batch_num}/{total_batches} → {len(batch)} chunks")
            result_json = _call_groq_with_retry(client, system_prompt, user_prompt, model_name)

            results = result_json.get("results", [])
            if not results:
                logger.warning(f"[semantic_scorer] Batch {batch_num} returned empty results.")

            for item in results:
                idx, blended = _parse_and_blend(item, n, w_ins, w_vir, w_con)
                if idx is None:
                    continue
                final_scores[idx] = blended
                scored_ids.add(idx)
                logger.debug(
                    f"  chunk[{idx:03d}] "
                    f"insight={item.get('insight_depth', 0):.1f} "
                    f"viral={item.get('viral_potential', 0):.1f} "
                    f"contrarian={item.get('contrarian_edge', 0):.1f} "
                    f"→ blended={blended:.2f}"
                )

        except Exception as e:
            logger.error(f"[semantic_scorer] Batch {batch_num} failed: {e}")

        if batch_num < total_batches:
            time.sleep(2.0)

    unscored = set(range(n)) - scored_ids
    if unscored:
        logger.warning(f"[semantic_scorer] {len(unscored)} chunks not scored: {sorted(unscored)}")

    arr_norm = _normalize(final_scores)
    logger.info(
        f"[semantic_scorer] Done — {len(scored_ids)}/{n} scored | "
        f"top={arr_norm.max():.3f}  bottom={arr_norm.min():.3f}"
    )
    return arr_norm