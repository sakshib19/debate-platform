"""
Merger Module — Combines transcription segments with diarization segments.

The Problem:
- Whisper gives us:  "Hello everyone" at 0.0s-2.5s (but doesn't know WHO said it)
- Pyannote gives us: SPEAKER_00 spoke from 0.0s-3.0s (but doesn't know WHAT they said)

The Solution:
- Cross-reference timestamps: "Hello everyone" (0.0-2.5) overlaps with SPEAKER_00 (0.0-3.0)
- Therefore: SPEAKER_00 said "Hello everyone"

Algorithm:
For each transcription segment:
    1. Find all diarization segments that overlap with it
    2. Calculate how much each speaker overlaps (in seconds)
    3. Assign the transcription to the speaker with the most overlap
    4. If no overlap found, assign to "UNKNOWN"

Example:
    Transcript:  "Hello" [0.0 - 2.0]     "I disagree" [2.0 - 4.0]
    Diarization: SPEAKER_00 [0.0 - 2.5]   SPEAKER_01 [2.5 - 5.0]

    Result:
    SPEAKER_00: "Hello"        (overlap: 2.0s out of 2.0s = 100%)
    SPEAKER_01: "I disagree"   (overlap: 1.5s out of 2.0s = 75%)
"""

import logging
from typing import List, Dict
from collections import defaultdict

logger = logging.getLogger(__name__)


def _calculate_overlap(seg_start: float, seg_end: float,
                       diar_start: float, diar_end: float) -> float:
    """
    Calculate the overlap duration between two time ranges.

    Example:
        seg:  [1.0 -------- 4.0]
        diar:      [2.0 -------- 5.0]
        overlap:   [2.0 --- 4.0] = 2.0 seconds

    Returns 0.0 if no overlap.
    """
    overlap_start = max(seg_start, diar_start)
    overlap_end = min(seg_end, diar_end)
    overlap = max(0.0, overlap_end - overlap_start)
    return overlap


def merge_transcript_and_diarization(
    transcript_segments: List[Dict],
    diarization_segments: List[Dict],
) -> List[Dict]:
    """
    Assign each transcription segment to a speaker based on time overlap.

    Args:
        transcript_segments: From transcription.py
            [{"start": 0.0, "end": 2.5, "text": "Hello", "confidence": 0.95}, ...]
        diarization_segments: From diarization.py
            [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}, ...]

    Returns:
        Merged segments with speaker labels:
        [
            {
                "start": 0.0,
                "end": 2.5,
                "text": "Hello",
                "speaker": "SPEAKER_00",
                "confidence": 0.95
            },
            ...
        ]
    """
    merged = []

    for t_seg in transcript_segments:
        t_start = t_seg["start"]
        t_end = t_seg["end"]
        text = t_seg["text"]
        confidence = t_seg.get("confidence", 0.0)

        # Find which speaker has the most overlap with this text segment
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
    """
    Group all merged segments by speaker and combine their text.

    Input (merged segments):
    [
        {"speaker": "SPEAKER_00", "text": "Hello everyone", "start": 0.0, "end": 2.5},
        {"speaker": "SPEAKER_01", "text": "I disagree", "start": 2.5, "end": 5.0},
        {"speaker": "SPEAKER_00", "text": "Let me explain", "start": 5.0, "end": 7.5},
    ]

    Output (grouped by speaker):
    {
        "SPEAKER_00": {
            "full_transcript": "Hello everyone Let me explain",
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "Hello everyone"},
                {"start": 5.0, "end": 7.5, "text": "Let me explain"},
            ],
            "total_speaking_time": 5.0,
            "num_segments": 2
        },
        "SPEAKER_01": {
            "full_transcript": "I disagree",
            "segments": [
                {"start": 2.5, "end": 5.0, "text": "I disagree"},
            ],
            "total_speaking_time": 2.5,
            "num_segments": 1
        }
    }
    """
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

    # Build final output
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
    """
    Create a human-readable transcript (for display in UI).

    Output:
        [00:00 - 00:02] SPEAKER_00: Hello everyone
        [00:02 - 00:05] SPEAKER_01: I disagree with that point
        [00:05 - 00:07] SPEAKER_00: Let me explain why
    """

    def _format_time(seconds: float) -> str:
        """Convert seconds to MM:SS format"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    lines = []
    for seg in merged_segments:
        start = _format_time(seg["start"])
        end = _format_time(seg["end"])
        speaker = seg["speaker"]
        text = seg["text"]
        lines.append(f"[{start} - {end}] {speaker}: {text}")

    return "\n".join(lines)