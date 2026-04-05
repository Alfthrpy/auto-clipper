"""
Segmenter — merge short Whisper segments into larger chunks
suitable for scoring.
"""
from dataclasses import dataclass

from config import Config
from pipeline.processor.transcriber import Segment


@dataclass
class Chunk:
    """A merged chunk of transcript segments."""
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def merge_segments(segments: list[Segment], config: Config) -> list[Chunk]:
    """
    Merge consecutive Whisper segments into larger chunks.

    Splitting logic:
      - Accumulate segments until we hit max_chunk_duration, OR
      - A silence gap > silence_gap_threshold is detected, OR
      - We run out of segments.
    Chunks shorter than min_chunk_duration are merged into the previous one.

    Args:
        segments: List of raw Whisper segments.
        config:   Pipeline configuration.

    Returns:
        List of Chunk objects.
    """
    if not segments:
        return []

    chunks: list[Chunk] = []
    current_start = segments[0].start
    current_texts: list[str] = []
    current_end = segments[0].end

    def flush():
        if current_texts:
            chunks.append(Chunk(
                start=round(current_start, 2),
                end=round(current_end, 2),
                text=" ".join(current_texts),
            ))

    for i, seg in enumerate(segments):
        gap = seg.start - current_end if i > 0 else 0
        chunk_duration = seg.end - current_start

        # Current chunk duration before adding this segment
        current_chunk_duration = current_end - current_start

        # Split conditions
        hit_max = (seg.end - current_start) > config.max_chunk_duration
        hit_gap_and_long_enough = (gap > config.silence_gap_threshold) and (current_chunk_duration >= config.min_chunk_duration)

        should_split = hit_max or hit_gap_and_long_enough

        if should_split and current_texts:
            flush()
            current_start = seg.start
            current_texts = []

        current_texts.append(seg.text)
        current_end = seg.end

    flush()  # last chunk

    # Merge short trailing chunk into previous
    if len(chunks) > 1 and chunks[-1].duration < config.min_chunk_duration:
        last = chunks.pop()
        chunks[-1] = Chunk(
            start=chunks[-1].start,
            end=last.end,
            text=chunks[-1].text + " " + last.text,
        )

    print(f"[segment] Merged {len(segments)} segments → {len(chunks)} chunks "
          f"(avg {sum(c.duration for c in chunks)/max(len(chunks),1):.0f}s each)")

    return chunks
