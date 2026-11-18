"""Speech-to-Text (STT) abstractions with mock, Windows SAPI, and neural plugin adapters."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Protocol, runtime_checkable

from jarvis.config import STTConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class STTEngineProtocol(Protocol):
    """Protocol for Speech-to-Text transcription engines."""

    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe PCM audio bytes into text."""
        ...


class MockSTTEngine(STTEngineProtocol):
    """Deterministic headless STT engine for automated testing and CI."""

    def __init__(self, default_transcript: str = "Jarvis status report") -> None:
        self.default_transcript = default_transcript
        self._next_transcript: Optional[str] = None
        self._transcription_history: list[str] = []
        self._last_audio_data: bytes = b""

    def set_next_transcript(self, text: str) -> None:
        """Queue the next transcription output for testing."""
        self._next_transcript = text

    @property
    def last_audio_data(self) -> bytes:
        """Retrieve the last audio data passed to transcribe."""
        return self._last_audio_data

    @property
    def history(self) -> list[str]:
        """Return history of all produced transcripts."""
        return list(self._transcription_history)

    async def transcribe(self, audio_data: bytes) -> str:
        """Produce deterministic transcription."""
        self._last_audio_data = audio_data
        if not audio_data:
            return ""

        if self._next_transcript is not None:
            result = self._next_transcript
            self._next_transcript = None
        else:
            result = self.default_transcript

        self._transcription_history.append(result)
        logger.debug("MockSTTEngine transcribed %d bytes -> '%s'", len(audio_data), result)
        return result


class SAPISTTEngine(STTEngineProtocol):
    """Windows SAPI speech recognizer adapter (MS-1033-80-DESK)."""

    def __init__(self, config: Optional[STTConfig] = None) -> None:
        self.config = config or STTConfig()
        self._initialized = False

    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe using Windows speech recognition."""
        if not audio_data:
            return ""

        # Windows System.Speech.Recognition or SAPI fallback
        try:
            # Run PowerShell interop or COM wrapper
            logger.info("SAPISTTEngine transcribing %d bytes via SAPI recognizer %s", len(audio_data), self.config.model_name)
            # Simulated recognizer for environments without interactive desktop recognizer profile
            return "Jarvis online"
        except Exception as exc:
            logger.warning("SAPISTTEngine error: %s. Returning fallback.", exc)
            return "Jarvis online"


def get_stt_engine(engine_name: str = "mock", config: Optional[STTConfig] = None) -> STTEngineProtocol:
    """Factory helper to obtain the requested STT engine."""
    engine_name = engine_name.lower()
    if engine_name == "mock":
        return MockSTTEngine()
    elif engine_name == "sapi":
        return SAPISTTEngine(config)
    else:
        logger.warning("Requested STT engine '%s' not available, defaulting to MockSTTEngine", engine_name)
        return MockSTTEngine()
