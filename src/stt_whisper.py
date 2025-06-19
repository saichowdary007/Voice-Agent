from __future__ import annotations

"""Whisper-based Speech-to-Text module used by the WebSocket pipeline.
Loads the Whisper model **once** at import time and exposes an async
`transcribe_bytes` helper that accepts raw PCM/ WAV bytes and returns a
string transcript (empty string if nothing decoded).

The model size can be configured via the `WHISPER_MODEL` env var (tiny, base …)
through `src.config`.
"""

import asyncio
from typing import Optional

from faster_whisper import WhisperModel

from src.config import WHISPER_MODEL

# ---------------------------------------------------------------------------
# Lazy-load model so that container start-up is quick; first call will incur
# a few seconds of CPU initialisation – acceptable for now.
# ---------------------------------------------------------------------------
_model: Optional[WhisperModel] = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


async def transcribe_bytes(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """Transcribe raw audio bytes (PCM/WAV) into text using Whisper.

    Args:
        audio_bytes: Byte string containing mono 16-bit little-endian PCM or
                     WAV data. If WAV, header is ignored automatically.
        sample_rate: Sample rate of given audio. Only used when PCM provided.

    Returns:
        Decoded text. Empty string when nothing recognised.
    """
    if not audio_bytes:
        return ""

    model = _get_model()

    # Run in executor to avoid blocking event loop
    def _run():
        segments, _ = model.transcribe(audio_bytes, beam_size=1, language="en")
        return "".join(seg.text for seg in segments).strip()

    return await asyncio.to_thread(_run) 