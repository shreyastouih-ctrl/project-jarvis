"""Jarvis Telephony and Phone Call Management Subsystem."""

from __future__ import annotations

from jarvis.telephony.call_engine import (
    ActiveCall,
    CallDebriefRecord,
    CallEngine,
    CallEngineProtocol,
    UrgencyClassifier,
)
from jarvis.telephony.simulator import (
    SCENARIO_EMERGENCY_REACTOR_BREACH,
    SCENARIO_HIGH_URGENT_DEFENSE,
    SCENARIO_NORMAL_STARK_EXPO,
    SCENARIO_SILENT_CALLER,
    SCENARIO_USER_TAKEOVER,
    CallSimulator,
    ScriptedCall,
    ScriptedTurn,
)
from jarvis.telephony.state import (
    InvalidTelephonyStateTransitionError,
    TelephonyState,
    TelephonyStateMachine,
    UrgencyLevel,
)

__all__ = [
    "TelephonyState",
    "UrgencyLevel",
    "InvalidTelephonyStateTransitionError",
    "TelephonyStateMachine",
    "ActiveCall",
    "CallDebriefRecord",
    "UrgencyClassifier",
    "CallEngineProtocol",
    "CallEngine",
    "ScriptedTurn",
    "ScriptedCall",
    "CallSimulator",
    "SCENARIO_EMERGENCY_REACTOR_BREACH",
    "SCENARIO_HIGH_URGENT_DEFENSE",
    "SCENARIO_NORMAL_STARK_EXPO",
    "SCENARIO_SILENT_CALLER",
    "SCENARIO_USER_TAKEOVER",
]
