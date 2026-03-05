"""
Evaluator module: AI-powered debate speaker evaluation.

Takes a speaker's transcript + retrieved judging criteria and uses
GPT to generate scores and structured feedback.

Will be implemented in Week 3.
"""


def evaluate_speaker(
    speaker_label: str,
    transcript: str,
    debate_format: str,
    motion: str,
    criteria_context: list,
) -> dict:
    """
    Use GPT to evaluate a speaker based on their transcript and judging criteria.

    Args:
        speaker_label: e.g., "Speaker 1"
        transcript: The speaker's full transcript text
        debate_format: asian_parl, british_parl, or wsdc
        motion: The debate motion/topic
        criteria_context: Retrieved chunks from RAG (judging criteria)

    Returns:
        {
            "scores": {
                "content": 7.5,
                "style": 7.0,
                "structure": 6.5,
                "rebuttal": 7.0,
                "strategy": 6.5,
                "total": 34.5,
            },
            "feedback": "Detailed paragraph feedback...",
            "strengths": "Speaker showed strong...",
            "weaknesses": "Areas needing improvement...",
            "suggestions": "To improve, try...",
        }
    """
    # TODO: Implement in Week 3, Day 15-18
    raise NotImplementedError("Evaluator not yet implemented")
