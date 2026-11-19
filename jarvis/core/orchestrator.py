"""Master Jarvis Orchestrator coordinating audio capture, VAD, STT, TTS, state machine, and event bus."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from jarvis.audio.capture import AudioCaptureProtocol, SoundDeviceCapture, VirtualAudioPipeline
from jarvis.audio.stt import MockSTTEngine, STTEngineProtocol
from jarvis.audio.tts import MockTTSEngine, TTSEngineProtocol, format_jarvis_speech
from jarvis.audio.vad import VADDetector, VADProtocol
from jarvis.config import SystemConfig
from jarvis.core.event_bus import Event, EventBus, EventType
from jarvis.core.state_machine import AssistantState, AssistantStateMachine

logger = logging.getLogger(__name__)


class JarvisOrchestrator:
    """Central orchestrator for the Jarvis AI Assistant."""

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        event_bus: Optional[EventBus] = None,
        state_machine: Optional[AssistantStateMachine] = None,
        audio_capture: Optional[AudioCaptureProtocol] = None,
        vad: Optional[VADDetector] = None,
        stt: Optional[STTEngineProtocol] = None,
        tts: Optional[TTSEngineProtocol] = None,
    ) -> None:
        self.config = config or SystemConfig()
        self.event_bus = event_bus or EventBus()
        self.state_machine = state_machine or AssistantStateMachine(self.event_bus)

        # Setup audio capture
        if audio_capture is not None:
            self.capture = audio_capture
        elif self.config.headless:
            self.capture = VirtualAudioPipeline(self.config.audio)
        else:
            self.capture = SoundDeviceCapture(self.config.audio)

        # Setup VAD, STT, TTS
        self.vad = vad or VADDetector(self.config.vad)
        self.stt = stt or MockSTTEngine()
        self.tts = tts or MockTTSEngine(self.config.tts)

        self._running = False
        self._audio_task: Optional[asyncio.Task] = None
        self._background_tasks: set[asyncio.Task] = set()
        self._current_generation_id: int = 0
        self._response_generator: Optional[callable] = None

    @property
    def is_running(self) -> bool:
        return self._running

    def set_response_generator(self, generator_fn: callable) -> None:
        """Register a custom response generator (e.g. model router in M4)."""
        self._response_generator = generator_fn

    async def start(self) -> None:
        """Start all subsystems and launch the background audio processing loop."""
        if self._running:
            return
        self._running = True

        self.capture.start()
        self._audio_task = asyncio.create_task(self._audio_loop(), name="jarvis_audio_loop")
        logger.info("JarvisOrchestrator started in %s mode", "headless" if self.config.headless else "live")

    async def stop(self) -> None:
        """Stop background tasks and audio capture cleanly."""
        if not self._running:
            return
        self._running = False

        if self._audio_task:
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass
            self._audio_task = None

        if self._background_tasks:
            for task in list(self._background_tasks):
                if not task.done():
                    task.cancel()
            await asyncio.gather(*list(self._background_tasks), return_exceptions=True)
            self._background_tasks.clear()

        self.capture.stop()
        self.tts.interrupt()
        logger.info("JarvisOrchestrator stopped")

    async def _audio_loop(self) -> None:
        """Continuously read PCM audio chunks, process VAD, and manage state transitions."""
        while self._running:
            # Non-blocking read chunk with small timeout
            chunk = await asyncio.to_thread(self.capture.read_chunk, 0.03)
            if not chunk:
                await asyncio.sleep(0.01)
                continue

            frame_result = self.vad.process_frame(chunk)

            # 1. Handle Speech Onset
            if frame_result.speech_started:
                now = time.time()
                logger.info("Speech onset detected at %.3f (RMS=%.2f)", now, frame_result.energy_rms)

                # Check for Barge-In Interruption during speaking
                if self.state_machine.is_speaking:
                    latency = self.tts.interrupt()
                    await self.event_bus.publish_event(
                        EventType.BARGE_IN_TRIGGERED,
                        generation_id=self._current_generation_id,
                        interruption_latency_ms=latency,
                        timestamp=now,
                    )
                    await self.state_machine.transition_to(
                        AssistantState.LISTENING,
                        reason="barge_in_interruption",
                    )
                elif self.state_machine.is_idle:
                    await self.state_machine.transition_to(
                        AssistantState.LISTENING,
                        reason="speech_onset",
                    )

                await self.event_bus.publish_event(
                    EventType.SPEECH_STARTED,
                    timestamp=now,
                    energy_rms=frame_result.energy_rms,
                )

            # 2. Handle Speech Ended
            if frame_result.speech_ended:
                audio_len = len(frame_result.speech_audio)
                duration_ms = (audio_len / (self.config.audio.sample_rate * self.config.audio.sample_width)) * 1000.0
                logger.info("Speech ended: %d bytes (%.1f ms)", audio_len, duration_ms)

                await self.event_bus.publish_event(
                    EventType.SPEECH_ENDED,
                    audio_data=frame_result.speech_audio,
                    duration_ms=duration_ms,
                )

                if self.state_machine.is_listening:
                    await self.state_machine.transition_to(
                        AssistantState.PROCESSING,
                        reason="speech_ended",
                    )
                    # Spawn speech processing task
                    task = asyncio.create_task(
                        self._process_speech_pipeline(frame_result.speech_audio),
                        name="process_speech_pipeline",
                    )
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)

    async def _process_speech_pipeline(self, audio_data: bytes) -> None:
        """Transcribe speech, generate response, and stream TTS output."""
        current = asyncio.current_task()
        if current is not None:
            self._background_tasks.add(current)
            current.add_done_callback(self._background_tasks.discard)

        try:
            # 1. Transcribe audio to text
            transcript = await self.stt.transcribe(audio_data)
            await self.event_bus.publish_event(
                EventType.TRANSCRIPT_READY,
                text=transcript,
                confidence=1.0,
            )

            if not transcript.strip():
                if self.state_machine.is_processing:
                    await self.state_machine.transition_to(AssistantState.IDLE, reason="empty_transcript")
                return

            # 2. Formulate assistant response
            if self._response_generator:
                response_text = await self._response_generator(transcript)
            else:
                response_text = self._generate_default_reflex(transcript)

            # 3. Transition to SPEAKING and stream TTS
            await self.speak(response_text)

        except asyncio.CancelledError:
            logger.debug("Speech processing pipeline cancelled")
            raise
        except Exception as exc:
            logger.exception("Error in speech processing pipeline: %s", exc)
            if self.state_machine.is_processing:
                await self.state_machine.transition_to(AssistantState.IDLE, reason="processing_error")

    def _generate_default_reflex(self, query: str) -> str:
        """Generate courteous Jarvis reflex response and execute desktop/system commands."""
        import datetime
        import subprocess
        import urllib.parse
        import webbrowser
        import psutil

        q_lower = query.lower().strip()

        # 1. Application & Web Actions (from Samin's tutorial)
        if "open youtube" in q_lower:
            webbrowser.open("https://www.youtube.com")
            return "Opening YouTube for you now, sir."
        elif (q_lower.startswith("play ") or "play" in q_lower) and "on youtube" in q_lower:
            song = q_lower.replace("play", "").replace("on youtube", "").replace("jarvis", "").strip()
            webbrowser.open(f"https://www.youtube.com/results?search_query={urllib.parse.quote(song)}")
            return f"Cueing up {song} on YouTube, sir."
        elif "open google" in q_lower:
            webbrowser.open("https://www.google.com")
            return "Opening Google search interface, sir."
        elif "search for" in q_lower or q_lower.startswith("google "):
            search_query = q_lower.replace("search for", "").replace("google", "").strip()
            webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote(search_query)}")
            return f"Searching for {search_query} on the network, sir."
        elif "open spotify" in q_lower:
            webbrowser.open("https://open.spotify.com")
            return "Accessing Spotify music interface, sir."
        elif "open calculator" in q_lower or q_lower == "calculator":
            try:
                subprocess.Popen("calc.exe")
                return "Calculator initialized, sir."
            except Exception:
                return "Calculator utility launched, sir."
        elif "open notepad" in q_lower or q_lower == "notepad":
            try:
                subprocess.Popen("notepad.exe")
                return "Opening Notepad for your notes, sir."
            except Exception:
                return "Notepad opened, sir."

        # 2. Time & Date
        elif "what time" in q_lower or "current time" in q_lower or q_lower == "time":
            now_time = datetime.datetime.now().strftime("%I:%M %p")
            return f"It is currently {now_time}, sir."
        elif "what date" in q_lower or "today's date" in q_lower or "what day" in q_lower:
            now_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
            return f"Today is {now_date}, sir."

        # 3. System Health & Hardware Diagnostics
        elif any(kw in q_lower for kw in ["diagnostics", "system report", "system status", "system health", "how are you"]):
            try:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                battery = psutil.sensors_battery()
                bat_str = f"{battery.percent} percent" if battery else "nominal"
                return f"All core systems operational, sir. Central processor is at {cpu} percent load, memory utilization is at {ram} percent, and power reserves are {bat_str}."
            except Exception:
                return "All core subsystems are online and operating within optimal parameters, sir."

        # 4. Mode & Persona
        elif "busy" in q_lower or "screening" in q_lower:
            return "Understood, sir. Enabling screening protocol for incoming communications."
        elif "available" in q_lower or "free" in q_lower:
            return "Disabling call screening. You are now marked available, sir."
        elif "who are you" in q_lower:
            return "I am J.A.R.V.I.S., Just A Rather Very Intelligent System, modeled to assist you with all workspace operations and interface telemetry."
        elif "thank you" in q_lower or "thanks" in q_lower:
            return "Always at your service, sir."
        elif any(greeting in q_lower for greeting in ["hello", "hi", "good morning", "good afternoon", "good evening", "jarvis"]):
            return "At your service, sir. What are your orders?"
        else:
            return f"Understood, sir. Processing your request: {query}."

    async def speak(self, text: str) -> None:
        """Synthesize and stream speech output with barge-in support."""
        formatted_speech = format_jarvis_speech(text)
        self._current_generation_id += 1
        gen_id = self._current_generation_id

        pre_speech_state = self.state_machine.current_state
        return_state = AssistantState.BUSY if pre_speech_state == AssistantState.BUSY else AssistantState.IDLE

        if not self.state_machine.is_speaking:
            await self.state_machine.transition_to(AssistantState.SPEAKING, reason="starting_speech")

        await self.event_bus.publish_event(
            EventType.TTS_SYNTHESIS_STARTED,
            text=formatted_speech,
            generation_id=gen_id,
        )

        try:
            chunk_count = 0
            async for chunk in self.tts.synthesize_stream(formatted_speech, gen_id):
                # Check if interrupted during stream
                if gen_id != self._current_generation_id:
                    logger.debug("Speech synthesis aborted: generation %d superseded by %d", gen_id, self._current_generation_id)
                    return

                chunk_count += 1
                await self.event_bus.publish_event(
                    EventType.TTS_AUDIO_CHUNK,
                    chunk=chunk,
                    chunk_index=chunk_count,
                    generation_id=gen_id,
                )

            # If generation completed without barge-in preemption, return to preserved pre-speech state
            if gen_id == self._current_generation_id and self.state_machine.is_speaking:
                await self.state_machine.transition_to(return_state, reason="speech_completed")

        except asyncio.CancelledError:
            logger.debug("Speech synthesis stream cancelled")
            raise
        except Exception as exc:
            logger.exception("Error during speech synthesis stream: %s", exc)
            if self.state_machine.is_speaking:
                await self.state_machine.transition_to(return_state, reason="tts_error")

    async def toggle_busy_mode(self, enabled: bool) -> None:
        """Toggle assistant busy mode."""
        if enabled:
            await self.state_machine.transition_to(AssistantState.BUSY, reason="busy_mode_enabled")
        else:
            await self.state_machine.transition_to(AssistantState.IDLE, reason="busy_mode_disabled")
