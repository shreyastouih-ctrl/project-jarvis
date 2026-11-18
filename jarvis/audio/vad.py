"""Adaptive Voice Activity Detection (VAD) with dynamic noise-floor and circular ring buffer."""

from __future__ import annotations

import collections
import logging
import math
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from jarvis.config import VADConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class VADProtocol(Protocol):
    """Protocol for Voice Activity Detection engines."""

    def process_chunk(self, chunk: bytes) -> Tuple[bool, float]:
        """Process an audio chunk and return (is_speech, energy_rms)."""
        ...

    def reset(self) -> None:
        """Reset internal state, ring buffer, and counters."""
        ...


@dataclass
class VADFrameResult:
    """Rich result returned by VAD detector for each audio chunk."""

    is_speech: bool
    energy_rms: float
    threshold: float
    noise_floor: float
    speech_started: bool = False
    speech_ended: bool = False
    speech_audio: bytes = b""


class VADDetector(VADProtocol):
    """Adaptive streaming RMS Voice Activity Detector.

    Features:
    - Dynamic noise floor tracking via rolling 25th percentile (p25).
    - 1000ms pre-speech circular ring buffer to retain leading phonemes.
    - Minimum speech onset duration verification (rejection of <64ms noise spikes).
    - Silence hangover detection (~300ms) to bridge natural pauses between words.
    """

    def __init__(self, config: Optional[VADConfig] = None) -> None:
        self.config = config or VADConfig()

        # Rolling history of energy levels to calculate p25 noise floor baseline
        self._energy_history: collections.deque[float] = collections.deque(
            maxlen=self.config.history_window_size
        )

        # Circular ring buffer holding recent pre-speech audio chunks
        self._ring_buffer: collections.deque[bytes] = collections.deque(
            maxlen=self.config.pre_speech_chunks
        )

        # Active speech segment accumulator
        self._speech_buffer: list[bytes] = []

        # Internal state counters
        self._in_speech: bool = False
        self._candidate_speech_chunks: list[bytes] = []
        self._consecutive_speech_frames: int = 0
        self._consecutive_silence_frames: int = 0
        self._last_noise_floor: float = self.config.min_noise_floor
        self._last_threshold: float = self.config.min_noise_floor * self.config.energy_multiplier

    @property
    def is_in_speech(self) -> bool:
        """Check if currently inside a confirmed speech segment."""
        return self._in_speech

    @property
    def current_noise_floor(self) -> float:
        """Return the current estimated dynamic noise floor."""
        return self._last_noise_floor

    @property
    def current_threshold(self) -> float:
        """Return the current adaptive speech detection threshold."""
        return self._last_threshold

    @property
    def pre_speech_buffer(self) -> list[bytes]:
        """Return chunks currently held in the pre-speech ring buffer."""
        return list(self._ring_buffer)

    def calculate_rms(self, chunk: bytes) -> float:
        """Compute the Root Mean Square (RMS) energy of a 16-bit PCM chunk."""
        if not chunk or len(chunk) < 2:
            return 0.0

        # Truncate odd-byte length buffers to preserve 16-bit PCM 2-byte alignment
        if len(chunk) % 2 != 0:
            chunk = chunk[:len(chunk) - (len(chunk) % 2)]
        if not chunk:
            return 0.0

        # Fast vectorized numpy conversion
        samples = np.frombuffer(chunk, dtype=np.int16)
        if len(samples) == 0:
            return 0.0

        # Cast to float64 to prevent integer overflow during squaring
        samples_f = samples.astype(np.float64)
        mean_square = np.mean(samples_f ** 2)
        return float(np.sqrt(mean_square))

    def _update_baseline(self, rms: float) -> Tuple[float, float]:
        """Update noise floor baseline and compute adaptive speech threshold."""
        self._energy_history.append(rms)

        if len(self._energy_history) >= 5:
            # 25th percentile of recent energy window
            p25 = float(np.percentile(list(self._energy_history), self.config.noise_floor_percentile))
            noise_floor = max(self.config.min_noise_floor, p25)
        else:
            noise_floor = self.config.min_noise_floor

        threshold = noise_floor * self.config.energy_multiplier
        self._last_noise_floor = noise_floor
        self._last_threshold = threshold
        return noise_floor, threshold

    def process_frame(self, chunk: bytes) -> VADFrameResult:
        """Process an audio chunk through the complete adaptive VAD pipeline."""
        rms = self.calculate_rms(chunk)
        noise_floor, threshold = self._update_baseline(rms)

        is_frame_above_threshold = (rms >= threshold)
        speech_started = False
        speech_ended = False
        completed_audio = b""

        if not self._in_speech:
            if is_frame_above_threshold:
                self._consecutive_speech_frames += 1
                self._candidate_speech_chunks.append(chunk)

                if self._consecutive_speech_frames >= self.config.min_speech_frames:
                    # Confirmed speech onset!
                    self._in_speech = True
                    speech_started = True
                    self._consecutive_silence_frames = 0

                    # Prepend pre-speech ring buffer to preserve initial phonemes
                    self._speech_buffer = list(self._ring_buffer) + self._candidate_speech_chunks
                    self._candidate_speech_chunks = []
                    logger.debug(
                        "Speech ONSET confirmed (RMS=%.2f, thresh=%.2f, ring_buffer_chunks=%d)",
                        rms,
                        threshold,
                        len(self._ring_buffer),
                    )
            else:
                # Energy dropped below threshold before reaching minimum speech frames:
                # Discard candidate chunks as noise spike (<64ms) and store chunk into ring buffer
                self._consecutive_speech_frames = 0
                self._candidate_speech_chunks.clear()
                self._ring_buffer.append(chunk)

        else:
            # Already inside an active speech segment
            self._speech_buffer.append(chunk)

            if is_frame_above_threshold:
                # Active speech continues, reset silence hangover
                self._consecutive_silence_frames = 0
            else:
                # Silence frame during speech, increment hangover counter
                self._consecutive_silence_frames += 1
                if self._consecutive_silence_frames >= self.config.hangover_chunks:
                    # Silence hangover expired: Confirmed speech end!
                    self._in_speech = False
                    speech_ended = True
                    self._consecutive_speech_frames = 0
                    self._consecutive_silence_frames = 0

                    # Extract full assembled speech audio
                    completed_audio = b"".join(self._speech_buffer)
                    self._speech_buffer.clear()
                    self._candidate_speech_chunks.clear()
                    self._ring_buffer.clear()
                    logger.debug(
                        "Speech ENDED confirmed (RMS=%.2f, total_bytes=%d)",
                        rms,
                        len(completed_audio),
                    )

        return VADFrameResult(
            is_speech=self._in_speech or is_frame_above_threshold,
            energy_rms=rms,
            threshold=threshold,
            noise_floor=noise_floor,
            speech_started=speech_started,
            speech_ended=speech_ended,
            speech_audio=completed_audio,
        )

    def process_chunk(self, chunk: bytes) -> Tuple[bool, float]:
        """VADProtocol compliance method returning (is_speech, energy_rms)."""
        result = self.process_frame(chunk)
        return (result.is_speech, result.energy_rms)

    def reset(self) -> None:
        """Reset internal buffers and counters."""
        self._energy_history.clear()
        self._ring_buffer.clear()
        self._speech_buffer.clear()
        self._candidate_speech_chunks.clear()
        self._in_speech = False
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        self._last_noise_floor = self.config.min_noise_floor
        self._last_threshold = self.config.min_noise_floor * self.config.energy_multiplier
        logger.debug("VADDetector reset")
