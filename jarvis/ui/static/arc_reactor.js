/**
 * Arc Reactor Orbital Ring Visualizer
 * High-performance HTML5 Canvas renderer simulating Tony Stark's palladium/vibranium core.
 * Strictly adheres to Iron Man aesthetic: Stark Cyan (#00E5FF), Electric Blue (#0077B6),
 * Stark Gold (#FFD700) on Obsidian Void (#050B14). STRICTLY NOT ULTRON RED.
 */
class ArcReactorVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.warn(`[ArcReactor] Canvas #${canvasId} not found.`);
            return;
        }
        this.ctx = this.canvas.getContext('2d');
        this.state = 'IDLE';
        this.audioLevel = 0.0;

        // Dynamic State Interpolation Targets
        this.rotationOuter = 0.0;
        this.rotationInner = 0.0;
        this.rotationMid = 0.0;

        this.targetSpeedOuter = 0.5;
        this.targetSpeedInner = -0.4;
        this.currentSpeedOuter = 0.5;
        this.currentSpeedInner = -0.4;

        this.pulsePhase = 0.0;
        this.baseRadius = 42.0;
        this.currentRadius = 42.0;

        this.primaryColor = '#00E5FF';
        this.secondaryColor = '#0077B6';
        this.glowColor = 'rgba(0, 229, 255, 0.45)';

        this.initHiDPI();
        this.startLoop();
    }

    initHiDPI() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        const displayWidth = rect.width || 460;
        const displayHeight = rect.height || 460;

        this.canvas.width = displayWidth * dpr;
        this.canvas.height = displayHeight * dpr;
        this.ctx.scale(dpr, dpr);
        this.width = displayWidth;
        this.height = displayHeight;
    }

    setState(newState) {
        if (!['IDLE', 'LISTENING', 'PROCESSING', 'SPEAKING', 'BUSY'].includes(newState)) {
            console.warn(`[ArcReactor] Invalid state transition: ${newState}`);
            return;
        }
        this.state = newState;

        switch (newState) {
            case 'IDLE':
                this.targetSpeedOuter = 0.5;
                this.targetSpeedInner = -0.4;
                this.primaryColor = '#00E5FF';
                this.secondaryColor = '#0077B6';
                this.glowColor = 'rgba(0, 229, 255, 0.4)';
                break;
            case 'LISTENING':
                this.targetSpeedOuter = 2.4;
                this.targetSpeedInner = -2.0;
                this.primaryColor = '#0077B6';
                this.secondaryColor = '#00E5FF';
                this.glowColor = 'rgba(0, 119, 182, 0.6)';
                break;
            case 'PROCESSING':
                this.targetSpeedOuter = 4.2;
                this.targetSpeedInner = -3.8;
                this.primaryColor = '#00E5FF';
                this.secondaryColor = '#0077B6';
                this.glowColor = 'rgba(0, 229, 255, 0.8)';
                break;
            case 'SPEAKING':
                this.targetSpeedOuter = 1.6;
                this.targetSpeedInner = -1.2;
                this.primaryColor = '#00E5FF';
                this.secondaryColor = '#0077B6';
                this.glowColor = 'rgba(0, 229, 255, 0.7)';
                break;
            case 'BUSY':
                this.targetSpeedOuter = 0.2;
                this.targetSpeedInner = -0.15;
                this.primaryColor = '#FFD700'; // Stark Gold
                this.secondaryColor = '#FFB703'; // Deep Amber
                this.glowColor = 'rgba(255, 215, 0, 0.55)';
                break;
        }
    }

    setAudioLevel(level) {
        // Clamped float [0.0, 1.0]
        this.audioLevel = Math.max(0.0, Math.min(1.0, level || 0.0));
    }

    startLoop() {
        const renderFrame = () => {
            this.update();
            this.draw();
            requestAnimationFrame(renderFrame);
        };
        requestAnimationFrame(renderFrame);
    }

    update() {
        // Smooth lerp speed transitions
        const lerpFactor = 0.08;
        this.currentSpeedOuter += (this.targetSpeedOuter - this.currentSpeedOuter) * lerpFactor;
        this.currentSpeedInner += (this.targetSpeedInner - this.currentSpeedInner) * lerpFactor;

        // Speaking modulation: velocity and radius expand with speech energy
        let effectiveOuterSpeed = this.currentSpeedOuter;
        if (this.state === 'SPEAKING') {
            effectiveOuterSpeed += this.audioLevel * 3.5;
        }

        this.rotationOuter += effectiveOuterSpeed * 0.02;
        this.rotationInner += this.currentSpeedInner * 0.02;
        this.rotationMid += (this.currentSpeedOuter * 0.5) * 0.02;

        this.pulsePhase += 0.05;
        const targetRadius = this.baseRadius + (this.state === 'SPEAKING' ? this.audioLevel * 25 : Math.sin(this.pulsePhase) * 3);
        this.currentRadius += (targetRadius - this.currentRadius) * 0.15;
    }

    draw() {
        const ctx = this.ctx;
        const cx = this.width / 2;
        const cy = this.height / 2;

        ctx.clearRect(0, 0, this.width, this.height);

        // 1. Central Core Radial Flare
        const coreGradient = ctx.createRadialGradient(cx, cy, 2, cx, cy, this.currentRadius * 1.8);
        coreGradient.addColorStop(0, '#FFFFFF');
        coreGradient.addColorStop(0.25, this.primaryColor);
        coreGradient.addColorStop(0.7, this.secondaryColor);
        coreGradient.addColorStop(1, 'transparent');

        ctx.save();
        ctx.fillStyle = coreGradient;
        ctx.beginPath();
        ctx.arc(cx, cy, this.currentRadius * 1.8, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        // 2. Central Unitor Ring
        ctx.save();
        ctx.strokeStyle = this.primaryColor;
        ctx.lineWidth = 3;
        ctx.shadowColor = this.glowColor;
        ctx.shadowBlur = 15;
        ctx.beginPath();
        ctx.arc(cx, cy, this.currentRadius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();

        // 3. Middle Stator: 10 Power Coils
        const coilCount = 10;
        const coilRadius = 90;
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(this.rotationMid);
        for (let i = 0; i < coilCount; i++) {
            const angle = (i * Math.PI * 2) / coilCount;
            ctx.save();
            ctx.rotate(angle);
            ctx.translate(0, -coilRadius);

            // Staggered coil glow
            const isHighlighted = (this.state === 'PROCESSING')
                ? ((Math.floor(this.pulsePhase * 3) % coilCount) === i)
                : true;

            ctx.fillStyle = isHighlighted ? this.primaryColor : this.secondaryColor;
            ctx.shadowColor = this.glowColor;
            ctx.shadowBlur = isHighlighted ? 12 : 4;

            // Trapezoidal power coil
            ctx.beginPath();
            ctx.moveTo(-10, -8);
            ctx.lineTo(10, -8);
            ctx.lineTo(7, 8);
            ctx.lineTo(-7, 8);
            ctx.closePath();
            ctx.fill();

            // Inner copper/gold winding lines
            ctx.strokeStyle = (this.state === 'BUSY') ? '#FFD700' : 'rgba(255, 255, 255, 0.6)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(-5, 0);
            ctx.lineTo(5, 0);
            ctx.stroke();

            ctx.restore();
        }
        ctx.restore();

        // 4. Inner Acceleration Ring (Segmented dashes counter-rotating)
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(this.rotationInner);
        ctx.strokeStyle = this.secondaryColor;
        ctx.lineWidth = 2;
        ctx.setLineDash([8, 14]);
        ctx.beginPath();
        ctx.arc(0, 0, 125, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();

        // 5. Outer Targeting Orbital Ring (Ticks & degree reticles)
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(this.rotationOuter);
        ctx.strokeStyle = this.primaryColor;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([]);
        ctx.shadowColor = this.glowColor;
        ctx.shadowBlur = 8;

        // Outer smooth thin ring
        ctx.beginPath();
        ctx.arc(0, 0, 155, 0, Math.PI * 2);
        ctx.stroke();

        // 36 Degree Tick Marks
        for (let j = 0; j < 36; j++) {
            const tickAngle = (j * Math.PI * 2) / 36;
            const tickLength = (j % 9 === 0) ? 12 : 5;
            const rStart = 155;
            const rEnd = rStart + tickLength;

            const x1 = Math.cos(tickAngle) * rStart;
            const y1 = Math.sin(tickAngle) * rStart;
            const x2 = Math.cos(tickAngle) * rEnd;
            const y2 = Math.sin(tickAngle) * rEnd;

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }
        ctx.restore();

        // 6. Tri-Reactor Gold Security Shields (Emphasized when BUSY)
        if (this.state === 'BUSY') {
            ctx.save();
            ctx.translate(cx, cy);
            ctx.rotate(this.rotationMid * 0.3);
            ctx.strokeStyle = '#FFD700';
            ctx.lineWidth = 4;
            ctx.shadowColor = 'rgba(255, 215, 0, 0.7)';
            ctx.shadowBlur = 18;

            for (let k = 0; k < 3; k++) {
                const shieldAngle = (k * Math.PI * 2) / 3;
                ctx.save();
                ctx.rotate(shieldAngle);
                ctx.beginPath();
                ctx.arc(0, 0, 175, -0.4, 0.4);
                ctx.stroke();
                ctx.restore();
            }
            ctx.restore();
        }
    }
}
