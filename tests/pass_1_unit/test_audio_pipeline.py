"""Comprehensive headless unit tests for Jarvis Audio Pipeline & Core State Subsystems.

Verifies:
1. VAD RMS energy calculation & dynamic noise floor adaptation.
2. Pre-speech circular ring buffer (verifying 1000ms audio retention on speech trigger).
3. Sub-30ms barge-in interruption (generation ID cancellation & audio queue draining).
4. State machine transitions (IDLE -> LISTENING -> PROCESSING -> SPEAKING -> IDLE / BUSY).
5. Event bus async dispatch, error isolation, and wait_for synchronization.
6. Headless end-to-end voice loop simulation using VirtualAudioPipeline, MockSTTEngine, and MockTTSEngine.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
import pytest

from jarvis.audio.capture import VirtualAudioPipeline
from jarvis.audio.stt import MockSTTEngine
from jarvis.audio.tts import MockTTSEngine, format_jarvis_speech
from jarvis.audio.vad import VADDetector, VADFrameResult
from jarvis.config import AudioConfig, SystemConfig, VADConfig
from jarvis.core.event_bus import Event, EventBus, EventType
from jarvis.core.orchestrator import JarvisOrchestrator
from jarvis.core.state_machine import AssistantState, AssistantStateMachine, InvalidStateTransitionError


# ==============================================================================
# 1. VAD RMS ENERGY CALCULATION & DYNAMIC NOISE FLOOR ADAPTATION
# ==============================================================================
class TestVADRMSAndAdaptiveNoiseFloor:
    """Test suite for RMS energy calculation and adaptive baseline tracking."""

    def test_rms_calculation_silence(self, vad_detector: VADDetector) -> None:
        """Verify RMS of pure zero-sample silence is exactly 0.0."""
        silence_chunk = b"\x00" * 1024  # 512 16-bit samples
        rms = vad_detector.calculate_rms(silence_chunk)
        assert rms == 0.0

    def test_rms_calculation_sine_wave(self, vad_detector: VADDetector) -> None:
        """Verify RMS calculation of a pure sine wave matches theoretical peak / sqrt(2)."""
        peak_amplitude = 10000.0
        t = np.linspace(0, 2 * np.pi, 512, endpoint=False)
        sine_samples = (peak_amplitude * np.sin(t)).astype(np.int16)
        sine_chunk = sine_samples.tobytes()

        rms = vad_detector.calculate_rms(sine_chunk)
        expected_rms = peak_amplitude / np.sqrt(2.0)
        # Allow 2% tolerance due to discrete sampling
        assert pytest.approx(rms, rel=0.02) == expected_rms

    def test_dynamic_noise_floor_baseline_tracking(self) -> None:
        """Verify that dynamic noise floor tracks 25th percentile of recent audio frames."""
        config = VADConfig(noise_floor_percentile=25.0, energy_multiplier=2.4, min_noise_floor=8.0)
        vad = VADDetector(config)

        # Feed 20 frames with amplitude ~100 (RMS ~70.7)
        for _ in range(20):
            tone = (np.sin(np.linspace(0, 2 * np.pi, 512)) * 100).astype(np.int16).tobytes()
            vad.process_frame(tone)

        # Baseline noise floor should be near ~70.7
        assert vad.current_noise_floor > 50.0
        # Adaptive threshold should equal noise_floor * multiplier (2.4)
        expected_threshold = vad.current_noise_floor * 2.4
        assert pytest.approx(vad.current_threshold, rel=0.01) == expected_threshold

    def test_noise_floor_upward_adaptation(self) -> None:
        """Verify noise floor baseline adapts upward in sustained high background noise."""
        config = VADConfig(noise_floor_percentile=25.0, energy_multiplier=2.4, history_window_size=50)
        vad = VADDetector(config)

        # Initially quiet: 10 frames at amplitude 50
        for _ in range(10):
            tone = (np.sin(np.linspace(0, 2 * np.pi, 512)) * 50).astype(np.int16).tobytes()
            vad.process_frame(tone)
        initial_floor = vad.current_noise_floor

        # Environment becomes noisier: 40 frames at amplitude 600
        for _ in range(40):
            noisy_tone = (np.sin(np.linspace(0, 2 * np.pi, 512)) * 600).astype(np.int16).tobytes()
            vad.process_frame(noisy_tone)

        adapted_floor = vad.current_noise_floor
        assert adapted_floor > initial_floor * 3.0, "Noise floor must adapt upward to prevent false triggers"

    def test_noise_floor_minimum_clamping(self) -> None:
        """Verify noise floor never drops below min_noise_floor even in dead silence."""
        config = VADConfig(min_noise_floor=12.5)
        vad = VADDetector(config)

        silence = b"\x00" * 1024
        for _ in range(15):
            res = vad.process_frame(silence)

        assert vad.current_noise_floor >= 12.5
        assert res.threshold >= 12.5 * config.energy_multiplier


# ==============================================================================
# 2. PRE-SPEECH CIRCULAR RING BUFFER & TIMING
# ==============================================================================
class TestPreSpeechRingBufferAndTiming:
    """Test suite for circular pre-speech buffer, onset trigger, and silence hangover."""

    def test_pre_speech_ring_buffer_capacity(self) -> None:
        """Verify circular ring buffer maintains 1000ms capacity (31 chunks)."""
        config = VADConfig(sample_rate=16000, chunk_size=512, pre_speech_buffer_ms=1000)
        vad = VADDetector(config)
        expected_capacity = vad.config.pre_speech_chunks  # 1000ms / 32ms = 31 chunks

        # Feed 50 distinct silence/low noise chunks
        for i in range(50):
            # Encode frame index in chunk for tracking
            chunk = i.to_bytes(4, "little") + b"\x00" * 1020
            vad.process_frame(chunk)

        # Ring buffer must retain exactly expected_capacity most recent frames
        ring = vad.pre_speech_buffer
        assert len(ring) == expected_capacity
        # Most recent frame index must be 49
        last_frame_idx = int.from_bytes(ring[-1][:4], "little")
        assert last_frame_idx == 49
        # Oldest retained frame index must be 49 - 31 + 1 = 19
        first_frame_idx = int.from_bytes(ring[0][:4], "little")
        assert first_frame_idx == (50 - expected_capacity)

    def test_pre_speech_audio_retention_on_trigger(self) -> None:
        """Verify all 1000ms of pre-speech audio are prepended to the speech buffer on onset."""
        config = VADConfig(
            sample_rate=16000,
            chunk_size=512,
            pre_speech_buffer_ms=1000,
            min_speech_frames=2,
            min_noise_floor=10.0,
        )
        vad = VADDetector(config)
        capacity = config.pre_speech_chunks

        # Feed pre-speech quiet chunks
        for i in range(capacity):
            quiet = (np.sin(np.linspace(0, 2 * np.pi, 512)) * 15).astype(np.int16).tobytes()
            vad.process_frame(quiet)

        # Now feed 2 high-energy speech chunks to trigger onset
        loud_chunk = (np.sin(np.linspace(0, 2 * np.pi, 512)) * 15000).astype(np.int16).tobytes()
        res1 = vad.process_frame(loud_chunk)
        assert not res1.speech_started, "Single frame must not trigger speech onset (<64ms)"

        res2 = vad.process_frame(loud_chunk)
        assert res2.speech_started, "Second frame (>=64ms) must confirm speech onset"

        # Speech buffer inside VAD must now contain pre-speech chunks + 2 speech chunks
        assert len(vad._speech_buffer) == capacity + 2

    def test_minimum_speech_duration_rejection(self) -> None:
        """Verify brief noise spike (<64ms / 1 frame) does not trigger speech."""
        config = VADConfig(min_speech_frames=2, min_noise_floor=10.0)
        vad = VADDetector(config)

        quiet = b"\x00" * 1024
        loud = (np.sin(np.linspace(0, 2 * np.pi, 512)) * 15000).astype(np.int16).tobytes()

        # Establish baseline
        for _ in range(5):
            vad.process_frame(quiet)

        # Spike of 1 frame
        res_spike = vad.process_frame(loud)
        assert not res_spike.speech_started

        # Followed by quiet
        res_quiet = vad.process_frame(quiet)
        assert not res_quiet.speech_started
        assert not vad.is_in_speech, "Transient spike must be discarded"

    def test_silence_hangover_and_offset_detection(self) -> None:
        """Verify speech ends only after silence persists for hangover duration (~300ms)."""
        config = VADConfig(
            min_speech_frames=2,
            silence_hangover_ms=300,  # ~9 frames
            min_noise_floor=10.0,
        )
        vad = VADDetector(config)
        hangover = config.hangover_chunks

        loud = (np.sin(np.linspace(0, 2 * np.pi, 512)) * 15000).astype(np.int16).tobytes()
        quiet = b"\x00" * 1024

        # Establish quiet baseline
        for _ in range(5):
            vad.process_frame(quiet)

        # Trigger speech
        vad.process_frame(loud)
        onset_res = vad.process_frame(loud)
        assert onset_res.speech_started
        assert vad.is_in_speech

        # Feed 5 frames of silence (< hangover_chunks)
        for _ in range(5):
            res = vad.process_frame(quiet)
            assert not res.speech_ended
            assert vad.is_in_speech

        # Feed 1 more loud frame: speech resumes, resetting hangover
        vad.process_frame(loud)
        assert vad.is_in_speech

        # Now feed full hangover duration of silence
        for idx in range(hangover - 1):
            res = vad.process_frame(quiet)
            assert not res.speech_ended
            assert vad.is_in_speech

        # Final frame triggers speech_ended!
        final_res = vad.process_frame(quiet)
        assert final_res.speech_ended
        assert not vad.is_in_speech
        assert len(final_res.speech_audio) > 0


# ==============================================================================
# 3. SUB-30MS BARGE-IN INTERRUPTION & TTS PROTOCOL
# ==============================================================================
class TestSub30msBargeInAndTTS:
    """Test suite for instant barge-in interruption, queue preemption, and TTS streaming."""

    def test_interruption_latency_under_30ms(self, mock_tts: MockTTSEngine) -> None:
        """Assert interrupt() completes in strictly under 30ms."""
        latency_ms = mock_tts.interrupt()
        assert latency_ms < 30.0, f"Barge-in latency was {latency_ms:.3f}ms, must be < 30ms"
        assert mock_tts.is_interrupted

    @pytest.mark.asyncio
    async def test_generation_id_cancellation_during_stream(self, mock_tts: MockTTSEngine) -> None:
        """Verify streaming TTS immediately halts yielding chunks when interrupted."""
        gen_id = 1
        chunks_received = []

        async def run_stream() -> None:
            async for chunk in mock_tts.synthesize_stream("This is a long response from Jarvis.", gen_id):
                chunks_received.append(chunk)
                if len(chunks_received) == 2:
                    # User barges in!
                    mock_tts.interrupt()

        await run_stream()
        # Stream should have terminated immediately on interruption, receiving only 2 chunks
        assert len(chunks_received) == 2
        assert mock_tts.is_interrupted
        assert mock_tts.active_generation_id > gen_id

    @pytest.mark.asyncio
    async def test_tts_ttft_tracking(self, mock_tts: MockTTSEngine) -> None:
        """Verify Time-To-First-Token (TTFT) latency is tracked accurately."""
        chunks = []
        async for chunk in mock_tts.synthesize_stream("Greetings sir.", generation_id=42):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert mock_tts.last_ttft_ms >= 0.0

    def test_persona_formatting(self) -> None:
        """Verify Jarvis persona formatter strips markdown and retains courteous brevity."""
        markdown_text = (
            "```python\ndef test(): pass\n```\n"
            "**Certainly!** Here is your *system* report:\n"
            "# System Status\n"
            "- Battery at 98%.\n"
            "- All shields nominal."
        )
        formatted = format_jarvis_speech(markdown_text)
        assert "```" not in formatted
        assert "**" not in formatted
        assert "#" not in formatted
        # Should be succinct 1-2 sentences
        sentences = formatted.split(".")
        assert len([s for s in sentences if s.strip()]) <= 2


# ==============================================================================
# 4. CENTRAL ASSISTANT STATE MACHINE
# ==============================================================================
class TestAssistantStateMachine:
    """Test suite for state transitions, validation, and history tracking."""

    @pytest.mark.asyncio
    async def test_canonical_voice_lifecycle(self, state_machine: AssistantStateMachine) -> None:
        """Verify full lifecycle: IDLE -> LISTENING -> PROCESSING -> SPEAKING -> IDLE."""
        assert state_machine.is_idle

        await state_machine.transition_to(AssistantState.LISTENING, reason="speech_onset")
        assert state_machine.is_listening

        await state_machine.transition_to(AssistantState.PROCESSING, reason="speech_ended")
        assert state_machine.is_processing

        await state_machine.transition_to(AssistantState.SPEAKING, reason="tts_started")
        assert state_machine.is_speaking

        await state_machine.transition_to(AssistantState.IDLE, reason="tts_finished")
        assert state_machine.is_idle

    @pytest.mark.asyncio
    async def test_barge_in_transition(self, state_machine: AssistantStateMachine) -> None:
        """Verify direct barge-in preemption transition: SPEAKING -> LISTENING."""
        await state_machine.transition_to(AssistantState.PROCESSING)
        await state_machine.transition_to(AssistantState.SPEAKING)
        assert state_machine.is_speaking

        # Barge-in occurs: user speaks while assistant is speaking
        await state_machine.transition_to(AssistantState.LISTENING, reason="barge_in")
        assert state_machine.is_listening

    @pytest.mark.asyncio
    async def test_busy_mode_transitions(self, state_machine: AssistantStateMachine) -> None:
        """Verify transitions into and out of BUSY mode."""
        await state_machine.transition_to(AssistantState.BUSY, reason="user_toggle")
        assert state_machine.is_busy

        await state_machine.transition_to(AssistantState.IDLE, reason="user_available")
        assert state_machine.is_idle

    @pytest.mark.asyncio
    async def test_invalid_transitions_rejected(self, state_machine: AssistantStateMachine) -> None:
        """Verify illegal transitions raise InvalidStateTransitionError."""
        # Illegal: LISTENING -> SPEAKING directly without processing
        await state_machine.transition_to(AssistantState.LISTENING)
        with pytest.raises(InvalidStateTransitionError):
            await state_machine.transition_to(AssistantState.SPEAKING)

        # Move to SPEAKING legally via PROCESSING
        await state_machine.transition_to(AssistantState.PROCESSING)
        await state_machine.transition_to(AssistantState.SPEAKING)

        # Illegal: SPEAKING -> PROCESSING directly
        with pytest.raises(InvalidStateTransitionError):
            await state_machine.transition_to(AssistantState.PROCESSING)

    @pytest.mark.asyncio
    async def test_state_history_tracking(self, state_machine: AssistantStateMachine) -> None:
        """Verify state history logs every transition with timestamps and reasons."""
        await state_machine.transition_to(AssistantState.LISTENING, reason="test_onset")
        await state_machine.transition_to(AssistantState.IDLE, reason="test_cancel")

        history = state_machine.get_history()
        assert len(history) >= 3  # init + 2 transitions
        assert history[-2][1] == AssistantState.LISTENING
        assert history[-2][2] == "test_onset"
        assert history[-1][1] == AssistantState.IDLE
        assert history[-1][2] == "test_cancel"


# ==============================================================================
# 5. ASYNC EVENT BUS DISPATCH & ERROR ISOLATION
# ==============================================================================
class TestEventBusAsync:
    """Test suite for event bus Pub/Sub, wildcard filtering, and error isolation."""

    @pytest.mark.asyncio
    async def test_typed_pub_sub(self, event_bus: EventBus) -> None:
        """Verify typed event delivery to registered subscribers."""
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        event_bus.subscribe(EventType.TRANSCRIPT_READY, handler)
        await event_bus.publish_event(EventType.TRANSCRIPT_READY, text="Hello Jarvis", confidence=0.99)

        assert len(received) == 1
        assert received[0].get("text") == "Hello Jarvis"
        assert received[0].get("confidence") == 0.99

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self, event_bus: EventBus) -> None:
        """Verify wildcard subscriber (None) receives all published events."""
        all_events: list[Event] = []

        def wildcard_handler(event: Event) -> None:
            all_events.append(event)

        event_bus.subscribe(None, wildcard_handler)

        await event_bus.publish_event(EventType.SPEECH_STARTED, timestamp=time.time())
        await event_bus.publish_event(EventType.STATE_CHANGED, old_state="IDLE", new_state="LISTENING")

        assert len(all_events) == 2
        assert all_events[0].event_type == EventType.SPEECH_STARTED
        assert all_events[1].event_type == EventType.STATE_CHANGED

    @pytest.mark.asyncio
    async def test_subscriber_error_isolation(self, event_bus: EventBus) -> None:
        """Verify an exception in one handler does NOT prevent other handlers from executing."""
        successful_calls: list[str] = []

        def failing_handler(event: Event) -> None:
            raise RuntimeError("Deliberate handler crash for error isolation testing")

        def surviving_handler(event: Event) -> None:
            successful_calls.append(event.get("marker", ""))

        event_bus.subscribe(EventType.DEVICE_STATE_CHANGED, failing_handler)
        event_bus.subscribe(EventType.DEVICE_STATE_CHANGED, surviving_handler)

        # Publishing must not raise or drop surviving_handler
        await event_bus.publish_event(EventType.DEVICE_STATE_CHANGED, marker="intact")
        assert successful_calls == ["intact"]

    @pytest.mark.asyncio
    async def test_wait_for_event_resolution(self, event_bus: EventBus) -> None:
        """Verify wait_for resolves correctly when matching event arrives."""
        async def publisher() -> None:
            await asyncio.sleep(0.05)
            await event_bus.publish_event(EventType.BARGE_IN_TRIGGERED, generation_id=7)

        asyncio.create_task(publisher())
        event = await event_bus.wait_for(EventType.BARGE_IN_TRIGGERED, timeout=1.0)
        assert event.get("generation_id") == 7

    @pytest.mark.asyncio
    async def test_wait_for_timeout_raises(self, event_bus: EventBus) -> None:
        """Verify wait_for raises TimeoutError on expiration."""
        with pytest.raises(TimeoutError):
            await event_bus.wait_for(EventType.CALL_INCOMING, timeout=0.1)


# ==============================================================================
# 6. HEADLESS END-TO-END VOICE LOOP SIMULATION
# ==============================================================================
class TestHeadlessEndToEndVoiceLoop:
    """Full voice loop integration using VirtualAudioPipeline, MockSTT, and MockTTS."""

    @pytest.mark.asyncio
    async def test_complete_voice_interaction_cycle(
        self,
        orchestrator: JarvisOrchestrator,
        virtual_pipeline: VirtualAudioPipeline,
        mock_stt: MockSTTEngine,
        event_bus: EventBus,
    ) -> None:
        """Simulate user speaking -> VAD onset -> VAD silence hangover -> STT -> TTS response -> IDLE."""
        mock_stt.set_next_transcript("Jarvis status report")
        await orchestrator.start()
        assert orchestrator.state_machine.is_idle

        # 1. Inject quiet frames to calibrate noise floor
        for _ in range(10):
            virtual_pipeline.inject_synthetic_audio(duration_ms=32, amplitude=0.01)

        # 2. Inject high-energy speech tone (4 frames = ~128ms)
        for _ in range(4):
            virtual_pipeline.inject_synthetic_audio(duration_ms=32, amplitude=0.8, frequency=300.0)

        # Wait for SPEECH_STARTED event
        speech_start_ev = await event_bus.wait_for(EventType.SPEECH_STARTED, timeout=2.0)
        assert speech_start_ev is not None

        # 3. Inject silence frames to trigger hangover expiration (12 frames = ~384ms > 300ms)
        for _ in range(12):
            virtual_pipeline.inject_silence(duration_ms=32)

        # Wait for SPEECH_ENDED event
        speech_end_ev = await event_bus.wait_for(EventType.SPEECH_ENDED, timeout=2.0)
        assert speech_end_ev is not None

        # Wait for TRANSCRIPT_READY
        transcript_ev = await event_bus.wait_for(EventType.TRANSCRIPT_READY, timeout=2.0)
        assert transcript_ev.get("text") == "Jarvis status report"

        # Wait for TTS audio chunk
        tts_chunk_ev = await event_bus.wait_for(EventType.TTS_AUDIO_CHUNK, timeout=2.0)
        assert len(tts_chunk_ev.get("chunk", b"")) > 0

        # Wait for speech to complete and return to IDLE
        await asyncio.sleep(0.5)
        assert orchestrator.state_machine.is_idle

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_barge_in_preempts_active_speech(
        self,
        orchestrator: JarvisOrchestrator,
        virtual_pipeline: VirtualAudioPipeline,
        event_bus: EventBus,
    ) -> None:
        """Simulate user barging in while Jarvis is actively speaking."""
        await orchestrator.start()

        # Trigger speech output directly
        speak_task = asyncio.create_task(
            orchestrator.speak("I am currently analyzing the telemetry feeds for any discrepancies, sir.")
        )
        # Give speak task a moment to begin streaming chunks
        await asyncio.sleep(0.05)
        assert orchestrator.state_machine.is_speaking

        # User interrupts by speaking loudly! Inject 3 loud frames (>64ms)
        for _ in range(3):
            virtual_pipeline.inject_synthetic_audio(duration_ms=32, amplitude=0.9, frequency=400.0)

        # Wait for BARGE_IN_TRIGGERED event
        barge_in_ev = await event_bus.wait_for(EventType.BARGE_IN_TRIGGERED, timeout=2.0)
        assert barge_in_ev is not None
        assert barge_in_ev.get("interruption_latency_ms") < 30.0

        # Assistant state must immediately shift from SPEAKING to LISTENING
        assert orchestrator.state_machine.is_listening

        await speak_task
        await orchestrator.stop()


# ==============================================================================
# 7. EXTENDED ORCHESTRATION, CONFIG & PIPELINE EDGE CASES
# ==============================================================================
class TestExtendedOrchestrationAndPipeline:
    """Test suite for custom response hooks, busy toggles, and pipeline edge cases."""

    @pytest.mark.asyncio
    async def test_custom_response_generator_hook(
        self,
        orchestrator: JarvisOrchestrator,
        mock_stt: MockSTTEngine,
        mock_tts: MockTTSEngine,
    ) -> None:
        """Verify orchestrator routes query through registered custom response generator."""
        async def custom_ai_router(query: str) -> str:
            return f"Custom analysis complete for: {query}"

        orchestrator.set_response_generator(custom_ai_router)
        mock_stt.set_next_transcript("What is the current trajectory?")

        # Trigger speech processing pipeline directly
        dummy_audio = b"\x01\x02" * 512
        await orchestrator._process_speech_pipeline(dummy_audio)

        assert len(mock_tts.history) > 0
        assert "Custom analysis complete for" in mock_tts.history[-1]

    @pytest.mark.asyncio
    async def test_empty_transcript_handling(
        self,
        orchestrator: JarvisOrchestrator,
        mock_stt: MockSTTEngine,
        mock_tts: MockTTSEngine,
    ) -> None:
        """Verify empty transcript transitions back to IDLE without synthesizing speech."""
        mock_stt.set_next_transcript("")
        await orchestrator.state_machine.transition_to(AssistantState.PROCESSING)

        await orchestrator._process_speech_pipeline(b"\x00" * 1024)
        assert orchestrator.state_machine.is_idle
        assert len(mock_tts.history) == 0

    @pytest.mark.asyncio
    async def test_orchestrator_busy_mode_toggle(
        self,
        orchestrator: JarvisOrchestrator,
    ) -> None:
        """Verify toggle_busy_mode toggles between BUSY and IDLE."""
        assert orchestrator.state_machine.is_idle

        await orchestrator.toggle_busy_mode(True)
        assert orchestrator.state_machine.is_busy

        await orchestrator.toggle_busy_mode(False)
        assert orchestrator.state_machine.is_idle

    def test_virtual_pipeline_padding_and_clear(self) -> None:
        """Verify virtual pipeline zero-pads incomplete final chunks and drains cleanly."""
        config = AudioConfig(chunk_size=512, channels=1, sample_width=2)  # 1024 bytes/chunk
        pipeline = VirtualAudioPipeline(config)
        pipeline.start()

        # Inject 1500 bytes (1 full chunk of 1024 + 1 partial chunk of 476 bytes)
        partial_data = b"\x05" * 1500
        chunks_added = pipeline.inject_pcm_data(partial_data)
        assert chunks_added == 2

        chunk1 = pipeline.read_chunk(timeout=0.1)
        assert len(chunk1) == 1024
        assert chunk1 == b"\x05" * 1024

        chunk2 = pipeline.read_chunk(timeout=0.1)
        assert len(chunk2) == 1024
        assert chunk2[:476] == b"\x05" * 476
        assert chunk2[476:] == b"\x00" * (1024 - 476)  # Zero padded!

        # Clear queue
        pipeline.inject_silence(duration_ms=64)
        pipeline.clear()
        assert pipeline.read_chunk(timeout=0.01) == b""
        pipeline.stop()

    def test_system_config_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify environment variables override configuration defaults."""
        from jarvis.config import load_config

        monkeypatch.setenv("JARVIS_HEADLESS", "true")
        monkeypatch.setenv("JARVIS_PORT", "9090")
        monkeypatch.setenv("JARVIS_LOG_LEVEL", "DEBUG")

        cfg = load_config()
        assert cfg.headless is True
        assert cfg.ui_port == 9090
        assert cfg.log_level == "DEBUG"
