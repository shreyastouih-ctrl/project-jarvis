"""Central typed configuration for the Jarvis AI Assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioConfig:
    """Audio capture and playback configuration."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 512  # 512 samples = 32ms frames at 16kHz
    sample_width: int = 2  # 16-bit PCM = 2 bytes per sample
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    buffer_queue_size: int = 200

    @property
    def bytes_per_chunk(self) -> int:
        """Returns byte count per audio chunk (512 * 1 * 2 = 1024 bytes)."""
        return self.chunk_size * self.channels * self.sample_width

    @property
    def chunk_duration_ms(self) -> float:
        """Duration of one chunk in milliseconds."""
        return (self.chunk_size / self.sample_rate) * 1000.0


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""

    sample_rate: int = 16000
    chunk_size: int = 512
    noise_floor_percentile: float = 25.0  # p25 percentile baseline
    energy_multiplier: float = 2.4       # threshold = max(min_noise_floor, p25 * multiplier)
    min_noise_floor: float = 8.0         # minimum clamped baseline RMS
    pre_speech_buffer_ms: int = 1000     # 1000ms pre-speech circular ring buffer
    min_speech_frames: int = 2           # >= 64ms required to trigger speech onset
    silence_hangover_ms: int = 300       # ~300ms silence hangover before speech end
    history_window_size: int = 100       # history buffer size for p25 noise estimation

    @property
    def pre_speech_chunks(self) -> int:
        """Number of chunks needed for pre-speech buffer."""
        chunk_ms = (self.chunk_size / self.sample_rate) * 1000.0
        return max(1, int(self.pre_speech_buffer_ms / chunk_ms))

    @property
    def hangover_chunks(self) -> int:
        """Number of silent chunks needed to trigger speech end."""
        chunk_ms = (self.chunk_size / self.sample_rate) * 1000.0
        return max(1, int(self.silence_hangover_ms / chunk_ms))


@dataclass
class TTSConfig:
    """Text-to-Speech configuration and Jarvis persona settings."""

    persona_name: str = "Jarvis"
    edge_voice: str = "en-GB-RyanNeural"
    pitch: str = "-2Hz"
    rate: str = "+5%"
    volume: str = "+0%"
    sapi_voice: str = "Microsoft David Desktop"
    enable_barge_in: bool = True
    barge_in_drain_timeout_ms: float = 30.0
    greeting_prefix: str = "At your service, sir."


@dataclass
class STTConfig:
    """Speech-to-Text configuration."""

    engine: str = "mock"  # "mock", "sapi", "vosk", "faster-whisper"
    model_name: str = "MS-1033-80-DESK"
    language: str = "en-US"


@dataclass
class SystemConfig:
    """Central system configuration aggregating all subsystems."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    headless: bool = False
    ui_port: int = 8000
    log_level: str = "INFO"


def load_config() -> SystemConfig:
    """Load configuration with environment variable overrides."""
    config = SystemConfig()
    if os.getenv("JARVIS_HEADLESS", "").lower() in ("1", "true", "yes"):
        config.headless = True
    if os.getenv("JARVIS_PORT"):
        try:
            config.ui_port = int(os.getenv("JARVIS_PORT", "8000"))
        except ValueError:
            pass
    if os.getenv("JARVIS_LOG_LEVEL"):
        config.log_level = os.getenv("JARVIS_LOG_LEVEL", "INFO")
    return config
