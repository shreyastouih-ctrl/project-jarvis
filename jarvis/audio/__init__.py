"""Audio capture, voice activity detection, STT, and TTS subsystems for Jarvis."""

from jarvis.audio.capture import AudioCaptureProtocol, SoundDeviceCapture, VirtualAudioPipeline
from jarvis.audio.stt import MockSTTEngine, SAPISTTEngine, STTEngineProtocol
from jarvis.audio.tts import EdgeTTSAudioEngine, MockTTSEngine, SAPITTSEngine, TTSEngineProtocol
from jarvis.audio.vad import VADDetector, VADFrameResult, VADProtocol

__all__ = [
    "AudioCaptureProtocol",
    "SoundDeviceCapture",
    "VirtualAudioPipeline",
    "VADProtocol",
    "VADDetector",
    "VADFrameResult",
    "STTEngineProtocol",
    "MockSTTEngine",
    "SAPISTTEngine",
    "TTSEngineProtocol",
    "MockTTSEngine",
    "EdgeTTSAudioEngine",
    "SAPITTSEngine",
]
