# Jarvis AI Assistant (Iron Man Inspired Proof-of-Concept)

[![GitHub](https://img.shields.io/badge/GitHub-shreyastouih--ctrl%2Fproject--jarvis-blue)](https://github.com/shreyastouih-ctrl/project-jarvis.git)

An exploratory proof-of-concept AI assistant inspired by **J.A.R.V.I.S.** from Iron Man that feels almost living, is always listening in the background, features a Jarvis-themed home screen interface and voice, integrates with smart devices and system operations, handles phone calls when the user is busy, and benchmarks models to select the fastest high-quality responses.

**Repository**: [https://github.com/shreyastouih-ctrl/project-jarvis.git](https://github.com/shreyastouih-ctrl/project-jarvis.git)

---

## Architecture Overview

The system is designed with a modular, asynchronous, event-driven architecture decoupling hardware I/O from core intelligence:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   JARVIS CONVERSATIONAL HUD                            │
│  • Arc Reactor Orbit Rings & Dynamic Waveform Visualizer               │
│  • Telemetry: CPU, RAM, Network, Battery, Audio Level                  │
│  • States: IDLE, LISTENING, PROCESSING, SPEAKING, BUSY                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
┌──────────────────────────────┐          ┌──────────────────────────────┐
│ R1: BACKGROUND VOICE SERVICE │          │ R3: DEVICE & CALL CONTROLLER │
│ • 16kHz InputStream & VAD    │          │ • Smart Device Registry      │
│ • Sub-30ms Barge-In Preempt  │          │ • Natural Language Parser    │
│ • en-GB-RyanNeural / SAPI    │          │ • Busy-Mode Call Screening   │
│ • Jarvis Persona Formatter   │          │ • Debriefing Card & History  │
└──────────────┬───────────────┘          └──────────────┬───────────────┘
               │                                         │
               └────────────────────┬────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             R4: MODEL CATEGORIZATION & DYNAMIC LATENCY SELECTOR        │
│  • Tier 0: Direct Reflex (<50ms)                                       │
│  • Tier 1: Ultra-Fast Conversational (<150ms)                          │
│  • Tier 2: Balanced Reasoning (<500ms)                                 │
│  • Tier 3: Deep Analytical Reasoning (>500ms)                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Core Subsystems:
1. **`jarvis.core`**: Async Publish/Subscribe Event Bus, Assistant State Machine (`IDLE`, `LISTENING`, `PROCESSING`, `SPEAKING`, `BUSY`), and Central Orchestrator.
2. **`jarvis.audio`**: Dual-backend audio capture (`SoundDeviceCapture` and headless `VirtualAudioPipeline`), streaming adaptive RMS Voice Activity Detection (VAD) with 1000ms circular ring buffer, speech onset detection (<64ms), silence hangover (~300ms), and sub-30ms barge-in preemption.
3. **`jarvis.ui`**: Holographic Iron Man HUD (cyan `#00E5FF`, electric blue `#0077B6`, Stark gold `#FFD700`, strictly NOT Ultron red), animated arc-reactor orbital visualizer, live telemetry gauges.
4. **`jarvis.devices`**: HomeAssistant JSON entity schema, device registry, natural language command parser, and simulated IoT bridge.
5. **`jarvis.telephony`**: Busy-mode toggle, autonomous AI call screening proxy, urgency classification, user call takeover, and structured debrief records.
6. **`jarvis.models`**: 4-Tier model taxonomy, dynamic latency benchmark probe (TTFT/TAT), fast-path selector, and mock provider.

---

## Voice Presets & Persona

- **Primary Persona**: Courteous, witty, respectful British male AI assistant modeled after Paul Bettany's performance as J.A.R.V.I.S.
- **Primary Voice Engine**: `edge-tts` with voice preset `en-GB-RyanNeural`:
  - **Pitch**: `-2Hz` (lowers tone for distinguished gravitas)
  - **Rate**: `+5%` (crisp, brisk British butler cadence)
- **Local SAPI Fallback**: Windows `Microsoft David Desktop`
- **Barge-in Interruption**: Sub-30ms instant queue draining and generation ID preemption whenever user speech onset is detected.

---

## Quickstart & CLI Flags

Run the assistant entry point:

```bash
# Run with live microphone capture
python run_jarvis.py --mic

# Run in headless mode with synthetic audio pipeline
python run_jarvis.py --headless --mock

# Specify custom HUD web port
python run_jarvis.py --port 8080
```

### CLI Arguments:
- `--headless`: Run without desktop GUI or hardware requirements.
- `--mic`: Use physical soundcard microphone capture (`sounddevice`).
- `--mock`: Use deterministic mock STT and TTS engines for testing and verification.
- `--port`: Specify HTTP/WebSocket port for the HUD interface (default: `8000`).

---

## Testing & Verification

Run the comprehensive unit test suite:

```bash
python -m pytest tests/pass_1_unit/test_audio_pipeline.py -v
```

All tests execute completely headlessly without requiring physical audio hardware or display servers.
