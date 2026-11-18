"""CLI Entry point for the Jarvis AI Assistant."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from typing import Optional

from jarvis.audio.capture import SoundDeviceCapture, VirtualAudioPipeline
from jarvis.audio.stt import MockSTTEngine, SAPISTTEngine
from jarvis.audio.tts import EdgeTTSAudioEngine, MockTTSEngine, SAPITTSEngine
from jarvis.config import SystemConfig, load_config
from jarvis.core.event_bus import Event, EventBus, EventType
from jarvis.core.orchestrator import JarvisOrchestrator
from jarvis.core.state_machine import AssistantState, AssistantStateMachine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jarvis.main")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Iron Man Jarvis AI Assistant")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headlessly without desktop GUI or hardware dependencies",
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Capture audio from physical soundcard microphone via sounddevice",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic mock STT and TTS backends for automated testing",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HUD web server (default: 8000)",
    )
    return parser.parse_args()


async def run_assistant(args: argparse.Namespace) -> None:
    """Initialize and run the Jarvis assistant."""
    config = load_config()
    config.ui_port = args.port
    if args.headless:
        config.headless = True

    event_bus = EventBus()
    state_machine = AssistantStateMachine(event_bus)

    # Choose audio capture
    if args.mic and not args.headless:
        audio_capture = SoundDeviceCapture(config.audio)
    else:
        audio_capture = VirtualAudioPipeline(config.audio)

    # Choose STT & TTS
    if args.mock:
        stt_engine = MockSTTEngine()
        tts_engine = MockTTSEngine(config.tts)
    else:
        stt_engine = SAPISTTEngine(config.stt)
        tts_engine = EdgeTTSAudioEngine(config.tts)

    orchestrator = JarvisOrchestrator(
        config=config,
        event_bus=event_bus,
        state_machine=state_machine,
        audio_capture=audio_capture,
        stt=stt_engine,
        tts=tts_engine,
    )

    # Log state transitions
    async def on_state_changed(event: Event) -> None:
        logger.info(
            "State changed: %s -> %s (reason: %s)",
            event.get("old_state"),
            event.get("new_state"),
            event.get("reason"),
        )

    event_bus.subscribe(EventType.STATE_CHANGED, on_state_changed)

    # Setup termination event
    stop_event = asyncio.Event()

    def signal_handler() -> None:
        logger.info("Termination signal received, shutting down Jarvis...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Signal handling on Windows event loop fallback
            pass

    logger.info("=" * 60)
    logger.info("      J.A.R.V.I.S. AI ASSISTANT INITIALIZED")
    logger.info("=" * 60)
    logger.info("Mode: %s", "Headless / Synthetic" if config.headless or not args.mic else "Live Microphone")
    logger.info("STT: %s | TTS: %s", stt_engine.__class__.__name__, tts_engine.__class__.__name__)
    logger.info("State: %s", state_machine.current_state.value)
    logger.info("Press Ctrl+C to stop.")

    await orchestrator.start()

    try:
        if args.headless and not args.mic:
            # In headless non-mic mode, run until stopped
            await stop_event.wait()
        else:
            await stop_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        await orchestrator.stop()
        await event_bus.close()
        logger.info("J.A.R.V.I.S. shutdown complete.")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    try:
        asyncio.run(run_assistant(args))
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
        sys.exit(0)


if __name__ == "__main__":
    main()
