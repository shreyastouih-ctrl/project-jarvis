"""Headless Phone Call Simulator for scripted multi-turn caller dialogue verification."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Union

from jarvis.telephony.call_engine import CallDebriefRecord, CallEngine


@dataclass
class ScriptedTurn:
    """Individual turn in a scripted call simulation."""

    utterance: str
    expected_urgency: Optional[str] = None
    trigger_takeover: bool = False
    delay_seconds: float = 0.0


@dataclass
class ScriptedCall:
    """Definition of a complete scripted phone call scenario."""

    call_id: str
    caller_name: str
    caller_number: str
    turns: list[Union[str, ScriptedTurn]]
    initial_busy_mode: bool = True
    takeover_after_turn: Optional[int] = None


class CallSimulator:
    """Headless driver for executing scripted phone calls against CallEngine."""

    def __init__(self, engine: CallEngine) -> None:
        self.engine = engine

    def run_simulation(self, script: ScriptedCall) -> CallDebriefRecord:
        """Synchronously execute a scripted call scenario."""
        self.engine.set_busy_mode(script.initial_busy_mode)
        self.engine.handle_incoming_call(script.call_id, script.caller_name, script.caller_number)

        for turn_idx, turn_item in enumerate(script.turns):
            if isinstance(turn_item, str):
                utterance = turn_item
                takeover = (script.takeover_after_turn == turn_idx)
            else:
                utterance = turn_item.utterance
                takeover = turn_item.trigger_takeover or (script.takeover_after_turn == turn_idx)

            if takeover:
                self.engine.take_over(script.call_id)
                break

            self.engine.process_caller_utterance(script.call_id, utterance)

        return self.engine.end_call(script.call_id)

    async def run_simulation_async(self, script: ScriptedCall) -> CallDebriefRecord:
        """Asynchronously execute a scripted call with simulated speech delays."""
        self.engine.set_busy_mode(script.initial_busy_mode)
        self.engine.handle_incoming_call(script.call_id, script.caller_name, script.caller_number)

        for turn_idx, turn_item in enumerate(script.turns):
            if isinstance(turn_item, str):
                utterance = turn_item
                takeover = (script.takeover_after_turn == turn_idx)
                delay = 0.005
            else:
                utterance = turn_item.utterance
                takeover = turn_item.trigger_takeover or (script.takeover_after_turn == turn_idx)
                delay = turn_item.delay_seconds or 0.005

            if delay > 0:
                await asyncio.sleep(delay)

            if takeover:
                self.engine.take_over(script.call_id)
                break

            self.engine.process_caller_utterance(script.call_id, utterance)

        return self.engine.end_call(script.call_id)


# Preset Iron Man Scenarios
SCENARIO_EMERGENCY_REACTOR_BREACH = ScriptedCall(
    call_id="call-sim-emerg-01",
    caller_name="Facility Guard",
    caller_number="+1-555-911-0000",
    turns=[
        "Critical emergency! Reactor breach detected in sector 4!",
    ],
)

SCENARIO_HIGH_URGENT_DEFENSE = ScriptedCall(
    call_id="call-sim-high-01",
    caller_name="Colonel Rhodes",
    caller_number="+1-555-0144",
    turns=[
        "Please have Tony review the urgent defense brief immediately.",
    ],
)

SCENARIO_NORMAL_STARK_EXPO = ScriptedCall(
    call_id="call-sim-normal-01",
    caller_name="Justin Hammer",
    caller_number="+1-555-0199",
    turns=[
        "I am calling about the Stark Expo contract.",
        "Please have him call me back when he is free.",
    ],
)

SCENARIO_SILENT_CALLER = ScriptedCall(
    call_id="call-sim-silent-01",
    caller_name="Unknown Caller",
    caller_number="+1-555-000-1111",
    turns=[
        "",
        "...",
    ],
)

SCENARIO_USER_TAKEOVER = ScriptedCall(
    call_id="call-sim-takeover-01",
    caller_name="Happy Hogan",
    caller_number="+1-555-123-4567",
    turns=[
        "Tony, I need to discuss the transport manifest.",
    ],
    takeover_after_turn=0,
)
