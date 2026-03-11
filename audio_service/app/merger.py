"""
Merger Module — Combines transcription segments with diarization segments.
"""

import logging
from typing import List, Dict
from collections import defaultdict

logger = logging.getLogger(__name__)


def _calculate_overlap(seg_start: float, seg_end: float,
                       diar_start: float, diar_end: float) -> float:
    overlap_start = max(seg_start, diar_start)
    overlap_end = min(seg_end, diar_end)
    return max(0.0, overlap_end - overlap_start)


def merge_transcript_and_diarization(
    transcript_segments: List[Dict],
    diarization_segments: List[Dict],
) -> List[Dict]:
    """Assign each transcription segment to a speaker based on time overlap."""
    merged = []

    for t_seg in transcript_segments:
        t_start = t_seg["start"]
        t_end = t_seg["end"]
        text = t_seg["text"]
        confidence = t_seg.get("confidence", 0.0)

        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for d_seg in diarization_segments:
            overlap = _calculate_overlap(t_start, t_end, d_seg["start"], d_seg["end"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d_seg["speaker"]

        merged.append({
            "start": t_start,
            "end": t_end,
            "text": text,
            "speaker": best_speaker,
            "confidence": confidence,
        })

    logger.info(f"Merged {len(merged)} segments across "
                f"{len(set(m['speaker'] for m in merged))} speakers")
    return merged


def group_by_speaker(merged_segments: List[Dict]) -> Dict[str, Dict]:
    """Group all merged segments by speaker and combine their text."""
    speakers = defaultdict(lambda: {
        "segments": [],
        "total_speaking_time": 0.0,
    })

    for seg in merged_segments:
        speaker = seg["speaker"]
        speakers[speaker]["segments"].append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
        })
        speakers[speaker]["total_speaking_time"] += (seg["end"] - seg["start"])

    result = {}
    for speaker, data in speakers.items():
        texts = [s["text"] for s in data["segments"]]
        result[speaker] = {
            "full_transcript": " ".join(texts),
            "segments": data["segments"],
            "total_speaking_time": round(data["total_speaking_time"], 2),
            "num_segments": len(data["segments"]),
        }

    return result


def format_readable_transcript(merged_segments: List[Dict]) -> str:
    """Create a human-readable transcript."""
    def _format_time(seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    lines = []
    for seg in merged_segments:
        start = _format_time(seg["start"])
        end = _format_time(seg["end"])
        lines.append(f"[{start} - {end}] {seg['speaker']}: {seg['text']}")

    return "\n".join(lines)
