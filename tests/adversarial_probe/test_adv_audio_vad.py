"""Adversarial Stress Probe: Audio Capture & VAD."""

import math
import threading
import time
import numpy as np
import pytest
from jarvis.audio.capture import VirtualAudioPipeline
from jarvis.audio.vad import VADDetector
from jarvis.config import AudioConfig, VADConfig


class TestAdversarialAudioAndVAD:
    """Stress testing VAD edge cases, odd buffers, NaN/Inf, extreme clipping, and thread safety."""

    def test_zero_and_sub_minimum_byte_buffers(self):
        """Boundary: 0-byte, 1-byte buffers passed to calculate_rms and process_frame."""
        vad = VADDetector()

        for chunk in [b"", b"\x00", b"\xff"]:
            rms = vad.calculate_rms(chunk)
            assert rms == 0.0, f"Expected 0.0 RMS for len {len(chunk)}, got {rms}"
            res = vad.process_frame(chunk)
            assert res.energy_rms == 0.0
            assert res.is_speech is False

    def test_odd_byte_length_buffer_vulnerability(self):
        """Boundary/Hardening: Odd byte length PCM buffers (3 bytes, 5 bytes, 513 bytes) are safely truncated without crashing."""
        vad = VADDetector()

        odd_chunks = [b"\x00\x01\x02", b"\x10\x20\x30\x40\x50", b"\xaa" * 513]
        crashes = []
        for chunk in odd_chunks:
            try:
                rms = vad.calculate_rms(chunk)
                assert rms >= 0.0
            except ValueError as exc:
                crashes.append((len(chunk), str(exc)))

        # Truncation ensures 0 crashes on odd-byte buffers
        assert len(crashes) == 0, f"Expected 0 crashes on odd-byte chunks, got {len(crashes)}: {crashes}"

    def test_extreme_dc_offset_and_full_scale_clipping(self):
        """Boundary: Full-scale positive DC offset (+32767), negative DC (-32768), and square wave."""
        vad = VADDetector()

        # 1. Full-scale positive DC offset (all +32767)
        pos_dc = np.full(512, 32767, dtype=np.int16).tobytes()
        rms_pos = vad.calculate_rms(pos_dc)
        assert not math.isnan(rms_pos)
        assert not math.isinf(rms_pos)
        assert abs(rms_pos - 32767.0) < 1.0

        # 2. Full-scale negative DC offset (all -32768)
        neg_dc = np.full(512, -32768, dtype=np.int16).tobytes()
        rms_neg = vad.calculate_rms(neg_dc)
        assert not math.isnan(rms_neg)
        assert not math.isinf(rms_neg)
        assert abs(rms_neg - 32768.0) < 1.0

        # 3. Maximum square wave clipping (+32767, -32768 alternating)
        square = np.array([32767, -32768] * 256, dtype=np.int16).tobytes()
        rms_square = vad.calculate_rms(square)
        assert not math.isnan(rms_square)
        assert not math.isinf(rms_square)
        assert rms_square > 32000.0

        # 4. Feed 20 frames of constant high DC offset to test adaptive noise floor behavior
        vad.reset()
        for _ in range(20):
            res = vad.process_frame(pos_dc)

        # Baseline noise floor should adapt upward towards 32767, threshold should exceed 32767
        # so it should NOT continuously report speech
        assert vad.current_noise_floor > 1000.0
        assert vad.current_threshold > vad.current_noise_floor

    def test_nan_and_inf_handling_in_synthetic_audio(self):
        """Boundary: Passing NaN and Inf to VirtualAudioPipeline generators."""
        pipeline = VirtualAudioPipeline()

        # NaN duration raises ValueError
        with pytest.raises(ValueError):
            pipeline.inject_synthetic_audio(duration_ms=float("nan"))

        # Inf duration raises OverflowError
        with pytest.raises(OverflowError):
            pipeline.inject_synthetic_audio(duration_ms=float("inf"))

        # NaN silence raises ValueError
        with pytest.raises(ValueError):
            pipeline.inject_silence(duration_ms=float("nan"))

        # NaN amplitude coerced via min/max
        added = pipeline.inject_synthetic_audio(duration_ms=10.0, amplitude=float("nan"))
        assert added >= 0

    def test_concurrent_multithreaded_pipeline_injection(self):
        """Concurrency: 5 threads concurrently injecting audio while main thread reads chunks."""
        config = AudioConfig(buffer_queue_size=50)
        pipeline = VirtualAudioPipeline(config)
        pipeline.start()

        num_threads = 5
        chunks_per_thread = 20
        sample_chunk = b"\x01\x00" * 512  # exactly 1024 bytes = 1 chunk

        errors = []

        def producer():
            try:
                for _ in range(chunks_per_thread):
                    pipeline.inject_pcm_data(sample_chunk)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=producer) for _ in range(num_threads)]
        for t in threads:
            t.start()

        # Consumer loop
        chunks_read = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.0 and chunks_read < (num_threads * chunks_per_thread):
            c = pipeline.read_chunk(timeout=0.05)
            if c:
                chunks_read += 1

        for t in threads:
            t.join()

        pipeline.stop()
        assert len(errors) == 0, f"Producer thread errors: {errors}"
        assert chunks_read > 0, "No chunks read from multithreaded pipeline"
