"""
Diarization Module — Uses pyannote.audio to identify speakers.
"""

import logging
from typing import List, Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)

_pipeline = None


def _patch_hf_hub():
    """
    Patch huggingface_hub to accept the deprecated 'use_auth_token' kwarg.

    pyannote.audio 3.3.2 passes use_auth_token= to hf_hub_download(),
    but huggingface_hub >=0.24 removed that parameter. This shim converts
    it to the current 'token' parameter so the call succeeds.
    Applied to both the module-level function and all pyannote submodules
    that import it directly.
    """
    import importlib
    import huggingface_hub as hh

    _original_download = hh.hf_hub_download

    def _patched_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs.setdefault("token", kwargs.pop("use_auth_token"))
        return _original_download(*args, **kwargs)

    if not getattr(hh, "_patched_use_auth_token", False):
        hh.hf_hub_download = _patched_download
        hh._patched_use_auth_token = True

        # Also patch direct imports in pyannote submodules
        pyannote_modules = [
            "pyannote.audio.core.pipeline",
            "pyannote.audio.core.model",
            "pyannote.audio.core.inference",
            "pyannote.audio.pipelines.speaker_diarization",
        ]
        for mod_name in pyannote_modules:
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, "hf_hub_download"):
                    mod.hf_hub_download = _patched_download
            except ImportError:
                pass


def _get_pipeline():
    """Load the pyannote diarization pipeline (cached after first call)."""
    global _pipeline
    if _pipeline is None:
        if not settings.HF_TOKEN or settings.HF_TOKEN.startswith("hf_your"):
            raise ValueError(
                "HuggingFace token not set! "
                "Set HF_TOKEN in audio_service/.env"
            )

        from huggingface_hub import login

        _patch_hf_hub()
        login(token=settings.HF_TOKEN)

        from pyannote.audio import Pipeline

        logger.info("Loading pyannote diarization pipeline...")
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
        )
        logger.info("Pyannote pipeline loaded successfully")
    return _pipeline


def diarize(audio_path: str, num_speakers: Optional[int] = None) -> List[Dict]:
    """
    Identify speakers in an audio file.

    Returns:
        List of speaker segments: [{"start": 0.0, "end": 3.21, "speaker": "SPEAKER_00"}, ...]
    """
    pipeline = _get_pipeline()

    logger.info(f"Diarizing: {audio_path} (expected speakers: {num_speakers or 'auto'})")

    if num_speakers:
        diarization = pipeline(audio_path, num_speakers=num_speakers)
    else:
        diarization = pipeline(audio_path)

    result = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        result.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker,
        })

    logger.info(f"Diarization complete: {len(result)} segments, "
                f"{len(set(r['speaker'] for r in result))} unique speakers")

    return result
