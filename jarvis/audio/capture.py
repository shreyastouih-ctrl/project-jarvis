"""Audio capture abstractions supporting live hardware and headless synthetic injection."""

from __future__ import annotations

import logging
import math
import queue
import struct
import threading
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from jarvis.config import AudioConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class AudioCaptureProtocol(Protocol):
    """Protocol for audio stream capture engines."""

    def start(self) -> None:
        """Start capturing audio."""
        ...

    def stop(self) -> None:
        """Stop capturing audio."""
        ...

    def read_chunk(self, timeout: float = 0.1) -> bytes:
        """Read a single chunk of PCM audio bytes. Returns empty bytes on timeout."""
        ...

    @property
    def is_active(self) -> bool:
        """Check if capture is currently active."""
        ...


class VirtualAudioPipeline(AudioCaptureProtocol):
    """Thread-safe virtual audio pipeline for headless automated testing and CI.

    Allows tests to inject deterministic synthetic audio, tone bursts, and silence chunks.
    """

    def __init__(self, config: Optional[AudioConfig] = None) -> None:
        self.config = config or AudioConfig()
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=self.config.buffer_queue_size)
        self._active = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Activate the virtual audio pipeline."""
        with self._lock:
            self._active = True
        logger.debug("VirtualAudioPipeline started")

    def stop(self) -> None:
        """Deactivate the virtual audio pipeline."""
        with self._lock:
            self._active = False
        logger.debug("VirtualAudioPipeline stopped")

    @property
    def is_active(self) -> bool:
        return self._active

    def clear(self) -> None:
        """Drain all buffered chunks."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def inject_pcm_data(self, pcm_data: bytes) -> int:
        """Inject raw PCM bytes into the pipeline, split into proper chunk sizes.

        Returns the number of chunks queued.
        """
        chunk_bytes = self.config.bytes_per_chunk
        chunks_added = 0
        offset = 0
        total_len = len(pcm_data)

        while offset < total_len:
            end = min(offset + chunk_bytes, total_len)
            chunk = pcm_data[offset:end]
            if len(chunk) < chunk_bytes:
                # Zero-pad incomplete final chunk
                chunk = chunk + b"\x00" * (chunk_bytes - len(chunk))
            try:
                self._queue.put(chunk, timeout=0.5)
                chunks_added += 1
            except queue.Full:
                logger.warning("VirtualAudioPipeline queue full, dropping chunk")
                break
            offset = end

        return chunks_added

    def inject_synthetic_audio(
        self,
        duration_ms: float,
        frequency: float = 440.0,
        amplitude: float = 0.5,
    ) -> int:
        """Generate a synthetic 16-bit mono sine wave and inject into pipeline.

        Args:
            duration_ms: Duration of the audio tone in milliseconds.
            frequency: Frequency of sine wave in Hertz (e.g. 440.0 Hz).
            amplitude: Signal amplitude between 0.0 and 1.0.

        Returns:
            Number of chunks enqueued.
        """
        sample_rate = self.config.sample_rate
        total_samples = int((duration_ms / 1000.0) * sample_rate)
        if total_samples <= 0:
            return 0

        # Generate sine wave samples
        t = np.arange(total_samples) / float(sample_rate)
        # Scaled to int16 range: [-32768, 32767]
        peak = int(32767 * max(0.0, min(1.0, amplitude)))
        waveform = (peak * np.sin(2.0 * np.pi * frequency * t)).astype(np.int16)
        raw_bytes = waveform.tobytes()

        return self.inject_pcm_data(raw_bytes)

    def inject_silence(self, duration_ms: float) -> int:
        """Inject zero-amplitude PCM silence chunks.

        Returns the number of chunks enqueued.
        """
        sample_rate = self.config.sample_rate
        total_samples = int((duration_ms / 1000.0) * sample_rate)
        if total_samples <= 0:
            return 0
        silence_bytes = b"\x00" * (total_samples * self.config.sample_width)
        return self.inject_pcm_data(silence_bytes)

    def read_chunk(self, timeout: float = 0.1) -> bytes:
        """Retrieve the next chunk of PCM audio.

        Returns empty bytes on timeout.
        """
        if not self._active:
            return b""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return b""


class SoundDeviceCapture(AudioCaptureProtocol):
    """Live microphone audio capture using the sounddevice library with automatic fallback."""

    def __init__(self, config: Optional[AudioConfig] = None) -> None:
        self.config = config or AudioConfig()
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=self.config.buffer_queue_size)
        self._stream = None
        self._active = False
        self._lock = threading.Lock()

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """Callback invoked by PortAudio / sounddevice for each input chunk."""
        if status:
            logger.debug("SoundDevice input status: %s", status)
        if not self._active:
            return

        # indata is numpy array or raw buffer of int16 samples
        raw_bytes = indata.tobytes() if hasattr(indata, "tobytes") else bytes(indata)
        try:
            self._queue.put_nowait(raw_bytes)
        except queue.Full:
            # Drop oldest to preserve real-time latency
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(raw_bytes)
            except (queue.Empty, queue.Full):
                pass

    def start(self) -> None:
        """Start sounddevice recording stream with automatic hardware fallback."""
        with self._lock:
            if self._active:
                return

            try:
                import sounddevice as sd
            except ImportError as err:
                raise RuntimeError("sounddevice is not installed. Install via pip.") from err

            device_index = self.config.input_device
            try:
                self._stream = sd.RawInputStream(
                    samplerate=self.config.sample_rate,
                    blocksize=self.config.chunk_size,
                    channels=self.config.channels,
                    dtype="int16",
                    device=device_index,
                    callback=self._audio_callback,
                )
                self._stream.start()
                self._active = True
                logger.info(
                    "SoundDeviceCapture active on device %s (sample_rate=%d, blocksize=%d)",
                    device_index if device_index is not None else "default",
                    self.config.sample_rate,
                    self.config.chunk_size,
                )
            except Exception as exc:
                logger.warning("Failed to start sounddevice on requested device (%s). Attempting default device: %s", device_index, exc)
                try:
                    self._stream = sd.RawInputStream(
                        samplerate=self.config.sample_rate,
                        blocksize=self.config.chunk_size,
                        channels=self.config.channels,
                        dtype="int16",
                        device=None,
                        callback=self._audio_callback,
                    )
                    self._stream.start()
                    self._active = True
                    logger.info("SoundDeviceCapture active on fallback default device.")
                except Exception as fallback_exc:
                    logger.error("SoundDeviceCapture failed to initialize any input audio device: %s", fallback_exc)
                    raise RuntimeError(f"Could not open audio capture stream: {fallback_exc}") from fallback_exc

    def stop(self) -> None:
        """Stop and close sounddevice stream."""
        with self._lock:
            self._active = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as exc:
                    logger.debug("Error stopping sounddevice stream: %s", exc)
                finally:
                    self._stream = None
            logger.info("SoundDeviceCapture stopped")

    @property
    def is_active(self) -> bool:
        return self._active

    def read_chunk(self, timeout: float = 0.1) -> bytes:
        """Read a captured chunk from the queue."""
        if not self._active:
            return b""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return b""
