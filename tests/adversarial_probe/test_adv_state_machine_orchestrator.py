"""Adversarial Concurrency & Stress Probe: State Machine & Orchestrator."""

import asyncio
import time
import pytest
from jarvis.audio.capture import VirtualAudioPipeline
from jarvis.audio.tts import MockTTSEngine
from jarvis.audio.vad import VADDetector
from jarvis.config import SystemConfig
from jarvis.core.event_bus import EventBus, EventType
from jarvis.core.orchestrator import JarvisOrchestrator
from jarvis.core.state_machine import (
    AssistantState,
    AssistantStateMachine,
    InvalidStateTransitionError,
)


class TestAdversarialStateMachineAndOrchestrator:
    """Stress testing state transitions, rapid barge-in preemption, and lifecycle races."""

    @pytest.mark.asyncio
    async def test_concurrent_out_of_order_transitions_on_state_machine(self):
        """Stress: 30 concurrent coroutines attempting competing valid and invalid transitions."""
        bus = EventBus()
        sm = AssistantStateMachine(event_bus=bus, initial_state=AssistantState.IDLE)

        successes = 0
        invalid_rejections = 0
        state_snapshots = []

        async def attempt_transition(target: AssistantState, reason: str):
            nonlocal successes, invalid_rejections
            try:
                res = await sm.transition_to(target, reason=reason)
                successes += 1
                state_snapshots.append(res)
            except InvalidStateTransitionError:
                invalid_rejections += 1

        # Mix of valid transitions and explicitly invalid transitions (e.g. SPEAKING from LISTENING)
        targets = [
            AssistantState.LISTENING,
            AssistantState.SPEAKING,    # Invalid from LISTENING
            AssistantState.PROCESSING,
            AssistantState.PROCESSING,  # Redundant / self
            AssistantState.SPEAKING,
            AssistantState.IDLE,
            AssistantState.LISTENING,
            AssistantState.SPEAKING,    # Invalid from LISTENING
        ] * 5  # 40 total attempts

        tasks = [asyncio.create_task(attempt_transition(t, f"stress_{i}")) for i, t in enumerate(targets)]
        await asyncio.gather(*tasks)

        # Assert no lock deadlock occurred, state machine remains in a valid enum state,
        # and invalid transitions were caught and rejected
        assert isinstance(sm.current_state, AssistantState)
        assert successes > 0
        assert invalid_rejections > 0, f"Expected some invalid transitions to be rejected, got {invalid_rejections}"
        assert len(sm.get_history()) <= sm._max_history
        await bus.close()

    @pytest.mark.asyncio
    async def test_rapid_barge_in_interruption_stress(self):
        """Stress: Rapid sequential barge-in interruptions during active TTS streaming (50 cycles)."""
        config = SystemConfig(headless=True)
        bus = EventBus()
        sm = AssistantStateMachine(event_bus=bus)
        mock_tts = MockTTSEngine(config.tts)
        mock_tts._chunk_delay_s = 0.005  # fast chunks for stress

        interruption_latencies = []

        for cycle in range(50):
            # Transition to speaking and start TTS
            await sm.transition_to(AssistantState.SPEAKING, reason=f"start_cycle_{cycle}", force=True)

            # Synthesize in background
            synth_task = asyncio.create_task(
                mock_tts.synthesize_stream("This is a synthesized test sentence for Jarvis barge-in.", cycle).__anext__()
            )

            # Wait tiny duration to ensure streaming started
            await asyncio.sleep(0.002)

            # Barge-in interruption
            lat = mock_tts.interrupt()
            interruption_latencies.append(lat)

            # State transition to listening
            await sm.transition_to(AssistantState.LISTENING, reason="barge_in")

            # Clean up task
            try:
                await synth_task
            except (StopAsyncIteration, asyncio.CancelledError):
                pass
            except Exception:
                pass

        # All interruptions must be strictly < 30ms per requirement
        max_lat = max(interruption_latencies)
        avg_lat = sum(interruption_latencies) / len(interruption_latencies)
        assert max_lat < 30.0, f"Barge-in latency violated sub-30ms guarantee: max={max_lat:.2f}ms"
        assert avg_lat < 10.0, f"Average barge-in latency too high: {avg_lat:.2f}ms"
        assert sm.current_state == AssistantState.LISTENING
        await bus.close()

    @pytest.mark.asyncio
    async def test_orchestrator_stop_task_leak_vulnerability(self):
        """Vulnerability probe: In-flight speech pipeline tasks leak and execute after orchestrator stop."""
        config = SystemConfig(headless=True)
        bus = EventBus()
        sm = AssistantStateMachine(event_bus=bus)
        orch = JarvisOrchestrator(config=config, event_bus=bus, state_machine=sm)

        # Slow down STT or response generator to simulate mid-flight model generation
        async def slow_response(transcript: str) -> str:
            await asyncio.sleep(0.2)
            return "Response after delay"

        orch.set_response_generator(slow_response)
        await orch.start()

        # Trigger speech pipeline task
        asyncio.create_task(orch._process_speech_pipeline(b"\x00" * 3200))
        await asyncio.sleep(0.05)

        # Stop orchestrator mid-flight
        await orch.stop()
        assert not orch.is_running
        history_len_at_stop = len(sm.get_history())

        # Wait for the slow response to finish post-stop
        await asyncio.sleep(0.3)
        history_len_after_wait = len(sm.get_history())

        # We probe whether background task continued executing and mutated state post-stop
        task_leaked = (history_len_after_wait > history_len_at_stop)
        await bus.close()
        # Verification: task must NOT leak and mutate state post-stop
        assert task_leaked is False, f"Background task leaked post-stop: history grew from {history_len_at_stop} to {history_len_after_wait}"

    @pytest.mark.asyncio
    async def test_speak_in_busy_mode_clobbers_busy_state(self):
        """Hardening verification: Calling speak() while in BUSY mode preserves BUSY state upon completion."""
        config = SystemConfig(headless=True)
        bus = EventBus()
        sm = AssistantStateMachine(event_bus=bus)
        orch = JarvisOrchestrator(config=config, event_bus=bus, state_machine=sm)

        # Put into busy mode
        await orch.toggle_busy_mode(True)
        assert sm.current_state == AssistantState.BUSY

        # Jarvis speaks a status response
        await orch.speak("Shields are at 100 percent, sir.")

        # Check if state returned to BUSY or was clobbered to IDLE
        state_after_speech = sm.current_state
        await bus.close()
        # Verification: state machine must preserve BUSY state after speaking
        assert state_after_speech == AssistantState.BUSY, f"Expected AssistantState.BUSY after speech, got {state_after_speech}"
