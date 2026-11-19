class JarvisSFX {
    constructor() {
        this.ctx = null;
    }
    init() {
        if (!this.ctx) {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) this.ctx = new AudioCtx();
        }
    }
    play(type = 'startup') {
        try {
            this.init();
            if (!this.ctx) return;
            if (this.ctx.state === 'suspended') this.ctx.resume();
            const now = this.ctx.currentTime;
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.connect(gain);
            gain.connect(this.ctx.destination);

            if (type === 'startup') {
                osc.type = 'sine';
                osc.frequency.setValueAtTime(220, now);
                osc.frequency.exponentialRampToValueAtTime(880, now + 0.35);
                gain.gain.setValueAtTime(0.01, now);
                gain.gain.linearRampToValueAtTime(0.12, now + 0.08);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.6);
                osc.start(now);
                osc.stop(now + 0.6);
            } else if (type === 'listening') {
                osc.type = 'triangle';
                osc.frequency.setValueAtTime(587.33, now);
                osc.frequency.setValueAtTime(880.0, now + 0.07);
                gain.gain.setValueAtTime(0.08, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.22);
                osc.start(now);
                osc.stop(now + 0.22);
            } else if (type === 'done') {
                osc.type = 'sine';
                osc.frequency.setValueAtTime(880, now);
                osc.frequency.setValueAtTime(440, now + 0.05);
                gain.gain.setValueAtTime(0.07, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.18);
                osc.start(now);
                osc.stop(now + 0.18);
            }
        } catch (e) {
            console.warn('[JarvisSFX] Audio error:', e);
        }
    }
}

/**
 * Dynamic Audio Oscilloscope & Telemetry Visualizer Bridge
 * Manages WebSocket connectivity (/ws/hud and fallback /ws/telemetry),
 * 32-band FFT audio oscilloscope rendering, live hardware telemetry gauges,
 * smart home device cards, and call screening widget. Includes standalone mock preview mode.
 */
class VisualizerBridge {
    constructor() {
        this.canvas = document.getElementById('oscilloscope');
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.arcReactor = null;
        this.sfx = new JarvisSFX();

        // 32-Band Oscilloscope State
        this.bandCount = 32;
        this.bands = new Float32Array(this.bandCount);
        this.peaks = new Float32Array(this.bandCount);
        this.decayRate = 0.88;
        this.audioLevel = 0.0;

        // Telemetry State
        this.telemetry = {
            cpu: 0.0,
            ram: 0.0,
            net_ping: 0.0,
            battery: 100.0,
            audio_level: 0.0,
            state: 'IDLE'
        };

        // WebSocket Connection Management
        this.ws = null;
        this.wsEndpoint = '/ws/hud';
        this.retryDelay = 1000;
        this.maxRetryDelay = 15000;
        this.retryCount = 0;
        this.isMockMode = false;

        this.initDOM();
        this.initOscilloscope();
        this.connectWebSocket();
        this.startClock();
    }

    setArcReactor(reactorInstance) {
        this.arcReactor = reactorInstance;
    }

    initDOM() {
        // Busy Mode Toggle Button
        const busyBtn = document.getElementById('busy-toggle-btn');
        if (busyBtn) {
            busyBtn.addEventListener('click', () => {
                const isCurrentlyBusy = (this.telemetry.state === 'BUSY');
                const targetState = isCurrentlyBusy ? 'IDLE' : 'BUSY';
                this.sendAction('toggle_busy', { busy: !isCurrentlyBusy });
                this.sendAction('set_state', { state: targetState });
            });
        }

        // Call Takeover Button
        const takeoverBtn = document.getElementById('call-takeover-btn');
        if (takeoverBtn) {
            takeoverBtn.addEventListener('click', () => {
                this.sendAction('takeover_call', {});
                takeoverBtn.disabled = true;
                takeoverBtn.textContent = 'CALL TRANSFERRED';
            });
        }

        // Terminal Clear Button
        const clearBtn = document.getElementById('terminal-clear-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                const term = document.getElementById('transcript-terminal');
                if (term) term.innerHTML = '';
            });
        }
    }

    initOscilloscope() {
        if (!this.canvas || !this.ctx) return;
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = (rect.width || 640) * dpr;
        this.canvas.height = (rect.height || 120) * dpr;
        this.ctx.scale(dpr, dpr);
        this.width = rect.width || 640;
        this.height = rect.height || 120;

        const renderLoop = () => {
            this.drawOscilloscope();
            requestAnimationFrame(renderLoop);
        };
        requestAnimationFrame(renderLoop);
    }

    connectWebSocket() {
        if (window.location.protocol === 'file:') {
            console.log('[Visualizer] Standalone file mode detected. Activating Mock Mode.');
            this.startMockMode();
            return;
        }

        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}${this.wsEndpoint}`;
        console.log(`[Visualizer] Connecting to WebSocket: ${wsUrl}`);

        try {
            this.ws = new WebSocket(wsUrl);
        } catch (e) {
            this.handleReconnect();
            return;
        }

        this.ws.onopen = () => {
            console.log('[Visualizer] WebSocket connection established.');
            this.retryDelay = 1000;
            this.retryCount = 0;
            this.updateConnectionStatus(true);
            this.ws.send(JSON.stringify({ action: 'ping' }));
        };

        this.ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                this.handleIncomingMessage(msg);
            } catch (err) {
                console.error('[Visualizer] Malformed WS message:', err);
            }
        };

        this.ws.onerror = (err) => {
            console.warn('[Visualizer] WebSocket error:', err);
            this.ws.close();
        };

        this.ws.onclose = () => {
            this.updateConnectionStatus(false);
            // Fallback between /ws/hud and /ws/telemetry on initial failure
            if (this.retryCount === 1 && this.wsEndpoint === '/ws/hud') {
                this.wsEndpoint = '/ws/telemetry';
                console.log('[Visualizer] Attempting fallback to /ws/telemetry...');
            }
            this.handleReconnect();
        };
    }

    handleReconnect() {
        this.retryCount++;
        if (this.retryCount > 4 && !this.isMockMode) {
            console.log('[Visualizer] Max retries reached. Activating Mock Fallback Mode.');
            this.startMockMode();
            return;
        }

        const jitter = Math.random() * 500;
        const delay = Math.min(this.maxRetryDelay, this.retryDelay) + jitter;
        this.retryDelay *= 1.5;

        setTimeout(() => this.connectWebSocket(), delay);
    }

    handleIncomingMessage(msg) {
        const type = msg.type;
        const data = msg.data || msg;

        if (type === 'state' || type === 'state_change') {
            const newState = data.state || data.new_state;
            if (newState) this.updateState(newState);
        } else if (type === 'telemetry') {
            this.updateTelemetry({
                cpu: data.cpu !== undefined ? data.cpu : data.cpu_percent,
                ram: data.ram !== undefined ? data.ram : data.ram_percent,
                net_ping: data.net_ping !== undefined ? data.net_ping : (data.ping !== undefined ? data.ping : 15.0),
                battery: data.battery !== undefined ? data.battery : (data.battery_percent !== undefined ? data.battery_percent : 100.0),
                audio_level: data.audio_level !== undefined ? data.audio_level : 0.0
            });
        } else if (type === 'transcript') {
            this.appendTranscript(data.speaker || 'JARVIS', data.text || '');
        } else if (type === 'device_update') {
            this.updateDevice(data);
        } else if (type === 'call_update') {
            this.updateCallScreening(data);
        }
    }

    sendAction(action, payload) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ action, ...payload }));
        } else if (this.isMockMode) {
            console.log(`[MockAction] Dispatched ${action}:`, payload);
            if (action === 'set_state' && payload.state) {
                this.updateState(payload.state);
            }
        }
    }

    updateState(state) {
        const prevState = this.telemetry.state;
        this.telemetry.state = state;
        const badge = document.getElementById('assistant-state');
        if (badge) badge.textContent = state;

        if (this.sfx && prevState !== state) {
            if (state === 'LISTENING') this.sfx.play('listening');
            else if (state === 'IDLE' && prevState === 'SPEAKING') this.sfx.play('done');
        }

        const busyBtn = document.getElementById('busy-toggle-btn');
        const busyLabel = document.getElementById('busy-label');
        if (busyBtn && busyLabel) {
            if (state === 'BUSY') {
                busyBtn.classList.add('active');
                busyLabel.textContent = 'MODE: BUSY (SHIELD ON)';
            } else {
                busyBtn.classList.remove('active');
                busyLabel.textContent = 'MODE: AVAILABLE';
            }
        }

        if (this.arcReactor) {
            this.arcReactor.setState(state);
        }
    }

    updateTelemetry(telem) {
        // Clamp metrics safely
        const cpu = Math.max(0.0, Math.min(100.0, telem.cpu || 0.0));
        const ram = Math.max(0.0, Math.min(100.0, telem.ram || 0.0));
        const ping = telem.net_ping !== undefined ? telem.net_ping : 0.0;
        const battery = Math.max(0.0, Math.min(100.0, telem.battery !== undefined ? telem.battery : 100.0));
        const audio = Math.max(0.0, Math.min(1.0, telem.audio_level || 0.0));

        this.telemetry.cpu = cpu;
        this.telemetry.ram = ram;
        this.telemetry.net_ping = ping;
        this.telemetry.battery = battery;
        this.audioLevel = audio;

        if (this.arcReactor) {
            this.arcReactor.setAudioLevel(audio);
        }

        // Update Gauges & Labels
        this.setGauge('cpu-val', 'cpu-bar', cpu, 100);
        this.setGauge('ram-val', 'ram-bar', ram, 100);
        this.setGauge('ping-val', 'ping-bar', Math.max(0, ping), 200);
        this.setGauge('battery-val', 'battery-bar', battery, 100);

        const rmsLabel = document.getElementById('audio-rms-val');
        if (rmsLabel) rmsLabel.textContent = audio.toFixed(3);

        // Modulate 32-band FFT bars with incoming audio level
        this.synthesizeSpectrum(audio);
    }

    setGauge(valId, barId, value, maxVal) {
        const valElem = document.getElementById(valId);
        const barElem = document.getElementById(barId);
        if (valElem) valElem.textContent = value.toFixed(value < 10 ? 1 : 0);
        if (barElem) {
            const circumference = 264; // 2 * pi * 42
            const percent = Math.min(1.0, value / maxVal);
            barElem.style.strokeDashoffset = (circumference * (1 - percent));
        }
    }

    synthesizeSpectrum(audioEnergy) {
        for (let i = 0; i < this.bandCount; i++) {
            // Harmonics curve: higher sensitivity on low-to-mid voice frequencies
            const freqFactor = Math.sin((i / this.bandCount) * Math.PI);
            const targetHeight = (audioEnergy * 0.9 * freqFactor) + (Math.random() * audioEnergy * 0.2);

            if (targetHeight > this.bands[i]) {
                this.bands[i] = targetHeight;
            } else {
                this.bands[i] *= this.decayRate;
            }

            // Peak-hold caps
            if (this.bands[i] > this.peaks[i]) {
                this.peaks[i] = this.bands[i];
            } else {
                this.peaks[i] -= 0.015;
                if (this.peaks[i] < 0) this.peaks[i] = 0;
            }
        }
    }

    drawOscilloscope() {
        if (!this.ctx) return;
        const ctx = this.ctx;
        const w = this.width;
        const h = this.height;

        ctx.clearRect(0, 0, w, h);

        const barWidth = (w / this.bandCount) - 3;
        for (let i = 0; i < this.bandCount; i++) {
            const barHeight = Math.max(3, this.bands[i] * (h - 20));
            const x = i * (barWidth + 3);
            const y = h - barHeight;

            // Stark Cyan bar
            ctx.fillStyle = 'rgba(0, 229, 255, 0.65)';
            ctx.fillRect(x, y, barWidth, barHeight);

            // Stark Gold peak cap
            const peakY = h - Math.max(3, this.peaks[i] * (h - 20));
            ctx.fillStyle = '#FFD700';
            ctx.fillRect(x, peakY - 2, barWidth, 2);
        }

        // Overlay central audio sine wave
        ctx.strokeStyle = '#00E5FF';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        const sliceWidth = w / 64;
        let sx = 0;
        for (let j = 0; j < 64; j++) {
            const waveAmp = (this.audioLevel * 25) * Math.sin(j * 0.35 + performance.now() * 0.01);
            const sy = (h / 2) + waveAmp;
            if (j === 0) ctx.moveTo(sx, sy);
            else ctx.lineTo(sx, sy);
            sx += sliceWidth;
        }
        ctx.stroke();
    }

    appendTranscript(speaker, text) {
        const terminal = document.getElementById('transcript-terminal');
        if (!terminal || !text) return;

        const entry = document.createElement('div');
        entry.className = `log-entry ${speaker.toLowerCase()}`;

        const now = new Date().toTimeString().split(' ')[0];
        entry.innerHTML = `
            <span class="timestamp">[${now}]</span>
            <span class="sender">${speaker.toUpperCase()}:</span>
            <span class="message">${text}</span>
        `;
        terminal.appendChild(entry);
        terminal.scrollTop = terminal.scrollHeight;
    }

    updateDevice(deviceData) {
        if (!deviceData || !deviceData.device_id) return;
        const card = document.querySelector(`.device-card[data-id="${deviceData.device_id}"]`);
        if (card) {
            const stateLabel = card.querySelector('.device-state');
            if (stateLabel) stateLabel.textContent = `${deviceData.state.toUpperCase()}`;
            if (deviceData.state === 'on' || deviceData.state === 'locked') {
                card.classList.add('active');
            } else {
                card.classList.remove('active');
            }
        }
    }

    updateCallScreening(callData) {
        const callerElem = document.getElementById('caller-name');
        const urgencyElem = document.getElementById('call-urgency');
        const takeoverBtn = document.getElementById('call-takeover-btn');
        const transcriptBox = document.getElementById('call-transcript-box');

        if (callerElem) callerElem.textContent = callData.caller_name || callData.caller || 'UNKNOWN CALLER';
        if (urgencyElem) {
            urgencyElem.textContent = (callData.urgency || 'NORMAL').toUpperCase();
        }
        if (takeoverBtn) takeoverBtn.disabled = false;

        if (transcriptBox && callData.last_utterance) {
            const p = document.createElement('p');
            p.className = 'transcript-line';
            p.textContent = `${callData.speaker || 'CALLER'}: ${callData.last_utterance}`;
            transcriptBox.appendChild(p);
            transcriptBox.scrollTop = transcriptBox.scrollHeight;
        }
    }

    updateConnectionStatus(isOnline) {
        const dot = document.getElementById('connection-dot');
        const label = document.getElementById('connection-status');
        if (dot && label) {
            if (isOnline) {
                dot.className = 'status-dot online';
                label.textContent = 'ONLINE // WS LINKED';
            } else {
                dot.className = 'status-dot';
                label.textContent = 'OFFLINE // RECONNECTING...';
            }
        }
    }

    startClock() {
        const clockElem = document.getElementById('hud-clock');
        setInterval(() => {
            if (clockElem) {
                const now = new Date();
                clockElem.textContent = `${now.toISOString().substring(11, 19)} UTC`;
            }
        }, 1000);
    }

    startMockMode() {
        this.isMockMode = true;
        this.updateConnectionStatus(true);
        const label = document.getElementById('connection-status');
        if (label) label.textContent = 'PREVIEW // STANDALONE MOCK';

        const states = ['IDLE', 'LISTENING', 'PROCESSING', 'SPEAKING'];
        let stateIdx = 0;

        setInterval(() => {
            stateIdx = (stateIdx + 1) % states.length;
            const targetState = states[stateIdx];
            this.updateState(targetState);

            if (targetState === 'SPEAKING') {
                this.appendTranscript('JARVIS', 'All systems are operating at peak efficiency, sir.');
            } else if (targetState === 'LISTENING') {
                this.appendTranscript('USER', 'Jarvis, system diagnostics.');
            }
        }, 6000);

        setInterval(() => {
            const isSpeaking = (this.telemetry.state === 'SPEAKING');
            this.updateTelemetry({
                cpu: 18.0 + Math.sin(Date.now() * 0.001) * 8.0,
                ram: 44.5 + Math.cos(Date.now() * 0.0005) * 2.0,
                net_ping: 14.0 + Math.random() * 4.0,
                battery: 98.0,
                audio_level: isSpeaking ? (0.3 + Math.random() * 0.5) : (Math.random() * 0.05)
            });
        }, 100);
    }
}

// Global Auto-Initialization
document.addEventListener('DOMContentLoaded', () => {
    const reactor = new ArcReactorVisualizer('orbital-rings');
    const visualizer = new VisualizerBridge();
    visualizer.setArcReactor(reactor);
});
