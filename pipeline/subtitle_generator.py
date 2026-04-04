"""
Subtitle Generator — ASS subtitle with karaoke word-by-word highlighting.

Creates vertical-optimized subtitles with:
- Word-by-word color highlight (karaoke effect)
- Keyword emphasis (uppercase for viral words)
- Large, bold font with outline for readability
"""
from pathlib import Path

import pysubs2

from config import Config
from pipeline.transcriber import Word


# Words that get UPPERCASED for emphasis
_KEYWORDS = {
    # English
    "never", "always", "nobody", "everybody", "everyone",
    "truth", "secret", "mistake", "rich", "poor", "fail",
    "success", "money", "billion", "million", "impossible",
    "shocking", "crazy", "actually", "literally", "seriously",
    "wrong", "right", "stop", "remember", "listen", "important",
    "dangerous", "powerful", "biggest", "worst", "best",
    # Indonesian
    "gila", "rahasia", "salah", "gagal", "sukses", "uang",
    "kaya", "miskin", "penting", "bahaya", "mustahil",
    "jangan", "harus", "ingat", "dengar", "besar", "parah",
}


def generate_subtitle_for_clip(
    all_words: list[Word],
    clip_start: float,
    clip_end: float,
    output_path: Path,
    config: Config,
) -> Path | None:
    """
    Generate an ASS subtitle file for a clip segment.

    Flow:
    1. Filter words within clip time range
    2. Adjust timestamps to be clip-relative (start from 0)
    3. Group words into display lines
    4. Generate ASS with karaoke highlighting

    Returns:
        Path to generated .ass file, or None if no words found.
    """
    # 1. Filter and adjust timestamps to clip-relative
    clip_words = [
        Word(
            start=round(w.start - clip_start, 3),
            end=round(w.end - clip_start, 3),
            text=w.text,
        )
        for w in all_words
        if w.start >= clip_start - 0.05 and w.end <= clip_end + 0.05
    ]

    if not clip_words:
        print(f"[subtitle] ⚠ No words found for clip "
              f"{clip_start:.1f}–{clip_end:.1f}s")
        return None

    # 2. Create ASS file with vertical resolution
    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = str(config.render_width)
    subs.info["PlayResY"] = str(config.render_height)

    # 3. Define style
    hr, hg, hb = config.subtitle_highlight_color
    tr, tg, tb = config.subtitle_text_color

    style = pysubs2.SSAStyle(
        fontname=config.subtitle_font,
        fontsize=config.subtitle_fontsize,
        primarycolor=pysubs2.Color(hr, hg, hb, 0),     # After karaoke
        secondarycolor=pysubs2.Color(tr, tg, tb, 0),    # Before karaoke
        outlinecolor=pysubs2.Color(0, 0, 0, 0),         # Black outline
        backcolor=pysubs2.Color(0, 0, 0, 120),           # Semi-transparent bg
        bold=True,
        outline=4,
        shadow=2,
        alignment=2,           # Bottom center
        marginv=config.subtitle_margin_v,
    )
    subs.styles["Default"] = style

    # 4. Group words into display lines
    groups = _group_words(clip_words, config.subtitle_words_per_line)

    # 5. Create karaoke events
    for idx, group in enumerate(groups):
        start_ms = max(0, int(group[0].start * 1000))
        end_ms = int(group[-1].end * 1000) + 200  # +200ms linger

        # Prevent overlap with the next subtitle block to avoid jumping (collision effect)
        if idx + 1 < len(groups):
            next_start_ms = int(groups[idx+1][0].start * 1000)
            if end_ms > next_start_ms:
                end_ms = next_start_ms

        event = pysubs2.SSAEvent(
            start=start_ms,
            end=end_ms,
            style="Default",
        )

        # Build karaoke text with \k tags
        parts = []
        for i, word in enumerate(group):
            if i == 0:
                k_cs = 0  # Highlight first word immediately
            else:
                # Duration from previous word start to this word start
                gap = word.start - group[i - 1].start
                k_cs = max(1, int(gap * 100))

            display = word.text.upper() if _is_keyword(word.text) else word.text
            parts.append(f"{{\\k{k_cs}}}{display}")

        event.text = " ".join(parts)
        subs.events.append(event)

    # 6. Save
    subs.save(str(output_path))
    print(f"[subtitle] ✓ {len(groups)} lines → {output_path.name}")
    return output_path


def generate_hook_ass(
    hook_text: str,
    output_path: Path,
    config: Config,
    duration: float = 30.0,  # default long, we'll clip it in ffmpeg anyway or use exact VO duration
) -> Path:
    """
    Generate an ASS subtitle file specifically for the Hook text.
    It text will be displayed at the top center of the screen statically.
    """
    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = str(config.render_width)
    subs.info["PlayResY"] = str(config.render_height)

    # Style specifically for Hook (Top Center, Yellow, Huge)
    hr, hg, hb = config.subtitle_highlight_color

    style = pysubs2.SSAStyle(
        fontname=config.subtitle_font,
        fontsize=int(config.subtitle_fontsize * 1.5),  # Much larger for the hook
        primarycolor=pysubs2.Color(hr, hg, hb, 0),     # Highlight color
        outlinecolor=pysubs2.Color(0, 0, 0, 0),        # Black outline
        backcolor=pysubs2.Color(0, 0, 0, 150),         # Darker bg
        bold=True,
        outline=5,
        shadow=3,
        alignment=8,           # Top center
        marginv=config.subtitle_margin_v, # Down from top
    )
    subs.styles["HookStyle"] = style

    # Insert line breaks aggressively or format
    words = hook_text.upper().split()
    lines = []
    # group max 3 words per line for huge hook
    for i in range(0, len(words), 3):
        lines.append(" ".join(words[i:i+3]))
    
    formatted_text = "\\N".join(lines)

    event = pysubs2.SSAEvent(
        start=0,
        end=int(duration * 1000), 
        text=formatted_text,
        style="HookStyle",
    )
    subs.events.append(event)

    subs.save(str(output_path))
    print(f"[subtitle] ✓ Hook overlay → {output_path.name}")
    return output_path


def _group_words(
    words: list[Word],
    words_per_line: int,
) -> list[list[Word]]:
    """
    Group words into display lines.

    Each group has at most `words_per_line` words.
    Groups are also split if time gap between consecutive words > 1.5s.
    """
    if not words:
        return []

    groups: list[list[Word]] = []
    current_group: list[Word] = [words[0]]

    for word in words[1:]:
        gap = word.start - current_group[-1].end
        at_limit = len(current_group) >= words_per_line

        if at_limit or gap > 1.5:
            groups.append(current_group)
            current_group = [word]
        else:
            current_group.append(word)

    if current_group:
        groups.append(current_group)

    return groups


def _is_keyword(word: str) -> bool:
    """Check if a word should be emphasized (uppercased)."""
    cleaned = word.strip().lower().rstrip(".,!?;:'\"")
    return cleaned in _KEYWORDS
