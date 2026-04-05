"""
Semantic Scorer — niche-adaptive, multi-backend LLM scorer.

Evaluates chunk quality on dimensions defined by the active ContentProfile.
Supports both Groq and Google Gemini (free tier) as LLM backends.

Uses JSON Mode batched inferencing to preserve free-tier rate limits.
"""
import json
import logging
import time
import numpy as np

from config import Config
from pipeline.processor.segmenter import Chunk
from pipeline.content_profiles import get_profile, ContentProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


def _extract_json(text: str) -> dict:
    """Safely extract JSON from an LLM response that might be wrapped in markdown."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


# ---------------------------------------------------------------------------
# Prompt builders — now profile-driven
# ---------------------------------------------------------------------------

def _build_batch_prompt(
    chunk_data: list[dict],
    profile: ContentProfile,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) using the content profile's rubric."""

    # Build JSON output schema from dimensions
    dim_fields = ", ".join(
        f'"{dim}": <float 0-10>' for dim in profile.scoring_dimensions
    )
    output_schema = (
        '{"results": [{"id": <int>, ' + dim_fields + '}]}'
    )

    system_prompt = (
        f"You are an expert content analyst for {profile.label} short-form video.\n"
        f"Your job is to score video transcript segments on dimensions relevant to this niche.\n"
        f"You MUST respond with valid JSON only — no explanation, no markdown, no extra keys.\n\n"
        + profile.scoring_rubric
        + "\n"
        + profile.few_shot_example
    )

    user_prompt = (
        "Score each segment below using the rubric. "
        "Return ONLY a JSON object matching this schema exactly:\n"
        f"{output_schema}\n\n"
        f"SEGMENTS:\n{json.dumps(chunk_data, ensure_ascii=False, indent=2)}"
    )

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Backend: Groq
# ---------------------------------------------------------------------------

def _call_groq_with_retry(
    client,
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_retries: int = 5,
) -> dict:
    from groq import RateLimitError, InternalServerError, APIConnectionError

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
            return _extract_json(response.choices[0].message.content)

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


# ---------------------------------------------------------------------------
# Backend: Google Gemini
# ---------------------------------------------------------------------------

def _call_gemini_with_retry(
    client,
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_retries: int = 5,
) -> dict:
    """Call Gemini API with retry logic, using JSON mode."""
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

    delay = 5.0  # Gemini free tier has 15 RPM — start with a longer delay
    for attempt in range(max_retries):
        try:
            from google.genai import types

            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=0.1,
                    max_output_tokens=512,
                ),
            )
            return _extract_json(response.text)

        except ResourceExhausted as e:
            if attempt == max_retries - 1:
                logger.error(f"[semantic_scorer] Gemini rate limit persists after {max_retries} attempts.")
                raise
            logger.warning(f"[semantic_scorer] Gemini rate limited. Waiting {delay:.1f}s...")
            time.sleep(delay)
            delay = min(delay * 2, 120.0)

        except ServiceUnavailable as e:
            if attempt == max_retries - 1:
                logger.error(f"[semantic_scorer] Gemini service error after {max_retries} attempts: {e}")
                raise
            logger.warning(f"[semantic_scorer] Gemini server error, retry {attempt + 1} in {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, 120.0)

        except json.JSONDecodeError as e:
            logger.error(f"[semantic_scorer] Gemini returned invalid JSON: {e}")
            raise

        except Exception as e:
            logger.error(f"[semantic_scorer] Gemini non-retryable error: {e}")
            raise


# ---------------------------------------------------------------------------
# Score blending
# ---------------------------------------------------------------------------

def _parse_and_blend(
    item: dict,
    n_chunks: int,
    dimensions: list[str],
    weights: list[float],
) -> tuple:
    """Validate, clamp, and blend a single LLM score item using profile dimensions."""
    idx = item.get("id")
    if idx is None or not isinstance(idx, int) or not (0 <= idx < n_chunks):
        logger.warning(f"[semantic_scorer] Unexpected id: {idx!r}")
        return None, 0.0

    def _clamp(v):
        return max(0.0, min(10.0, float(v)))

    values = [_clamp(item.get(dim, 0)) for dim in dimensions]
    blended = sum(v * w for v, w in zip(values, weights))
    return idx, blended


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def score_semantic_llm(
    chunks: list[Chunk],
    config: Config,
    max_batch_size: int = 10,
) -> np.ndarray:
    """
    Score chunks using the active content profile and chosen LLM backend.

    Reads config.content_niche to select rubric/weights,
    and config.semantic_backend to select Groq or Gemini.

    Returns:
        np.ndarray of normalized scores [0.0, 1.0], same length as chunks.
    """
    if not chunks:
        return np.array([])

    import os
    from dotenv import load_dotenv
    load_dotenv()

    # --- Load content profile ---
    profile = get_profile(config.content_niche)

    # --- Resolve weights ---
    weights_dict = {**profile.scoring_weights, **getattr(config, "semantic_weights", {})}
    dimensions = profile.scoring_dimensions
    # Normalize weights to sum = 1
    raw_weights = [float(weights_dict.get(d, 0.0)) for d in dimensions]
    w_total = sum(raw_weights) or 1.0
    norm_weights = [w / w_total for w in raw_weights]

    # --- Select backend ---
    backend = getattr(config, "semantic_backend", "groq")
    model_name = getattr(config, "semantic_model", "") or ""

    if backend == "gemini":
        api_key = config.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("[semantic_scorer] No Gemini API key. Returning zeros.")
            return np.zeros(len(chunks))
        from google import genai
        client = genai.Client(api_key=api_key)
        model_name = model_name or "gemini-2.5-flash"
        call_fn = _call_gemini_with_retry
        batch_delay = 5.0  # Gemini free tier: 15 RPM
    else:
        api_key = config.groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("[semantic_scorer] No Groq API key. Returning zeros.")
            return np.zeros(len(chunks))
        from groq import Groq
        client = Groq(api_key=api_key)
        model_name = model_name or "llama-3.1-8b-instant"
        call_fn = _call_groq_with_retry
        batch_delay = 2.0

    n = len(chunks)
    final_scores = np.zeros(n)
    scored_ids: set[int] = set()
    total_batches = (n + max_batch_size - 1) // max_batch_size

    weight_log = " | ".join(f"{d}={w:.2f}" for d, w in zip(dimensions, norm_weights))
    logger.info(
        f"[semantic_scorer] Scoring {n} chunks | backend={backend} | model={model_name} | "
        f"niche={profile.niche} | batch={max_batch_size}\n"
        f"  Weights → {weight_log}"
    )
    print(
        f"[semantic_scorer] 🎯 Niche: {profile.label} | Backend: {backend.upper()} | "
        f"Model: {model_name}"
    )

    for batch_num, i in enumerate(range(0, n, max_batch_size), start=1):
        batch = chunks[i: i + max_batch_size]
        chunk_data = [
            {"id": i + j, "text": _truncate_at_sentence(chunk.text)}
            for j, chunk in enumerate(batch)
        ]

        system_prompt, user_prompt = _build_batch_prompt(chunk_data, profile)

        try:
            logger.info(f"[semantic_scorer] Batch {batch_num}/{total_batches} → {len(batch)} chunks")
            result_json = call_fn(client, system_prompt, user_prompt, model_name)

            results = result_json.get("results", [])
            if not results:
                logger.warning(f"[semantic_scorer] Batch {batch_num} returned empty results.")

            for item in results:
                idx, blended = _parse_and_blend(item, n, dimensions, norm_weights)
                if idx is None:
                    continue
                final_scores[idx] = blended
                scored_ids.add(idx)
                dim_log = " ".join(
                    f"{d}={item.get(d, 0):.1f}" for d in dimensions
                )
                logger.debug(
                    f"  chunk[{idx:03d}] {dim_log} → blended={blended:.2f}"
                )

        except Exception as e:
            logger.error(f"[semantic_scorer] Batch {batch_num} failed: {e}")

        if batch_num < total_batches:
            time.sleep(batch_delay)

    unscored = set(range(n)) - scored_ids
    if unscored:
        logger.warning(f"[semantic_scorer] {len(unscored)} chunks not scored: {sorted(unscored)}")

    arr_norm = _normalize(final_scores)
    logger.info(
        f"[semantic_scorer] Done — {len(scored_ids)}/{n} scored | "
        f"top={arr_norm.max():.3f}  bottom={arr_norm.min():.3f}"
    )
    return arr_norm