"""Comprehensive test suite for AI-Based Adaptive Noise Removal for Clear Speech."""

from __future__ import annotations

import io
import os
import sys
import tempfile

import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import JOBS, app as flask_app  # noqa: E402


def _synthesize_speech_like_signal(sr: int = 16000, duration: float = 2.0) -> np.ndarray:
    """Create a speech-like signal with voiced harmonic bursts and silent pauses."""
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    f0 = 145.0 + 25.0 * np.sin(2.0 * np.pi * 2.5 * t)
    voiced = (
        0.55 * np.sin(2.0 * np.pi * f0 * t)
        + 0.30 * np.sin(2.0 * np.pi * 2.0 * f0 * t)
        + 0.15 * np.sin(2.0 * np.pi * 3.0 * f0 * t)
    )
    # Speech envelope with clear pause gaps so noise floor estimation is meaningful
    envelope = np.zeros_like(t)
    envelope[(t >= 0.25) & (t <= 0.85)] = 1.0
    envelope[(t >= 1.15) & (t <= 1.75)] = 0.9
    signal = (voiced * envelope).astype(np.float32)
    return signal * 0.6


def _add_noise(clean: np.ndarray, noise_type: str, seed: int = 42, sr: int = 16000) -> np.ndarray:
    """Add stationary or non-stationary background noise to a clean signal."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, len(clean) / sr, len(clean), endpoint=False, dtype=np.float32)

    if noise_type == "white":
        noise = 0.12 * rng.standard_normal(len(clean)).astype(np.float32)
    elif noise_type == "hum":
        # Low-frequency AC mains hum + harmonic buzz + light hiss
        noise = (
            0.14 * np.sin(2.0 * np.pi * 60.0 * t)
            + 0.08 * np.sin(2.0 * np.pi * 180.0 * t)
            + 0.05 * rng.standard_normal(len(clean))
        ).astype(np.float32)
    elif noise_type == "pink_fan":
        white = rng.standard_normal(len(clean))
        # Simple 1/f spectral shaping via FFT
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(len(clean), d=1.0 / sr)
        scale = 1.0 / np.maximum(np.sqrt(freqs), 1.0)
        pink = np.fft.irfft(spectrum * scale, n=len(clean))
        pink = pink / (np.max(np.abs(pink)) + 1e-8)
        noise = (0.18 * pink).astype(np.float32)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return np.clip(clean + noise, -0.99, 0.99).astype(np.float32)


def _save_temp_audio(samples: np.ndarray, sr: int, fmt: str = "wav") -> bytes:
    """Serialize waveform into WAV or MP3 bytes in memory."""
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format=fmt.upper())
    buf.seek(0)
    return buf.read()


@pytest.fixture()
def client(tmp_path):
    """Configure isolated upload/output/image folders for each test."""
    JOBS.clear()
    upload_dir = tmp_path / "uploads"
    output_dir = tmp_path / "outputs"
    images_dir = tmp_path / "images"
    upload_dir.mkdir()
    output_dir.mkdir()
    images_dir.mkdir()

    flask_app.config["TESTING"] = True
    flask_app.config["UPLOAD_FOLDER"] = str(upload_dir)
    flask_app.config["OUTPUT_FOLDER"] = str(output_dir)
    flask_app.config["IMAGES_FOLDER"] = str(images_dir)

    with flask_app.test_client() as test_client:
        yield test_client


def test_homepage_and_health(client):
    home_res = client.get("/")
    assert home_res.status_code == 200
    assert b"AI-Based Adaptive Noise Removal for Clear Speech" in home_res.data

    health_res = client.get("/health")
    assert health_res.status_code == 200
    data = health_res.get_json()
    assert data["status"] == "ok"
    assert "primary_model" in data
    assert "fallback_model" in data


def test_empty_and_invalid_uploads(client):
    # 1. Missing audio field
    res = client.post("/upload", data={}, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "error" in res.get_json()

    # 2. Empty filename
    res = client.post(
        "/upload",
        data={"audio": (io.BytesIO(b""), "")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400

    # 3. Unsupported extension (.txt)
    res = client.post(
        "/upload",
        data={"audio": (io.BytesIO(b"not audio"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400
    assert "Unsupported file format" in res.get_json()["error"]

    # 4. Empty 0-byte WAV file
    res = client.post(
        "/upload",
        data={"audio": (io.BytesIO(b""), "empty.wav")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400
    assert "empty" in res.get_json()["error"].lower()

    # 5. Corrupted WAV bytes
    res = client.post(
        "/upload",
        data={"audio": (io.BytesIO(b"corrupted-binary-data"), "broken.wav")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400


def test_noisy_speech_wav_deep_learning_pipeline_and_download(client):
    clean = _synthesize_speech_like_signal(sr=16000, duration=2.0)
    noisy = _add_noise(clean, "white", seed=7, sr=16000)

    noisy_bytes = _save_temp_audio(noisy, 16000, fmt="wav")
    clean_bytes = _save_temp_audio(clean, 16000, fmt="wav")

    upload_res = client.post(
        "/upload",
        data={
            "audio": (io.BytesIO(noisy_bytes), "noisy_speech.wav"),
            "reference": (io.BytesIO(clean_bytes), "clean_ref.wav"),
        },
        content_type="multipart/form-data",
    )
    assert upload_res.status_code == 200
    up_data = upload_res.get_json()
    job_id = up_data["job_id"]
    assert up_data["original_filename"] == "noisy_speech.wav"
    assert up_data["duration_seconds"] == pytest.approx(2.0, abs=0.05)

    # Confirm original audio can be streamed for playback
    orig_play = client.get(up_data["original_audio_url"])
    assert orig_play.status_code == 200
    assert len(orig_play.data) > 0

    proc_res = client.post(
        "/process",
        json={"job_id": job_id, "method": "deep_learning"},
    )
    assert proc_res.status_code == 200
    proc_data = proc_res.get_json()

    metrics = proc_data["metrics"]
    assert metrics["original_duration_seconds"] == pytest.approx(2.0, abs=0.05)
    assert metrics["enhanced_duration_seconds"] == pytest.approx(2.0, abs=0.05)
    assert metrics["processing_time_seconds"] > 0.0
    assert metrics["estimated_noise_reduction_db"] > 0.0
    assert metrics["snr_available"] is True
    assert metrics["snr_after_db"] > metrics["snr_before_db"]

    # Confirm all 4 visualization PNGs were generated on disk
    for plot_url in proc_data["plots"].values():
        filename = os.path.basename(plot_url)
        assert os.path.exists(os.path.join(flask_app.config["IMAGES_FOLDER"], filename))

    # Confirm enhanced audio playback route works
    enh_play = client.get(proc_data["enhanced_audio_url"])
    assert enh_play.status_code == 200
    assert len(enh_play.data) > 44

    # Confirm download route works with job_id and without job_id
    dl_res = client.get(proc_data["download_url"])
    assert dl_res.status_code == 200
    assert dl_res.headers["Content-Disposition"].startswith("attachment")
    assert dl_res.data[:4] == b"RIFF"

    dl_latest = client.get("/download")
    assert dl_latest.status_code == 200
    assert dl_latest.data[:4] == b"RIFF"


def test_mp3_upload_and_spectral_gating_fallback(client):
    clean = _synthesize_speech_like_signal(sr=16000, duration=1.5)
    noisy = _add_noise(clean, "pink_fan", seed=19, sr=16000)
    mp3_bytes = _save_temp_audio(noisy, 16000, fmt="mp3")

    upload_res = client.post(
        "/upload",
        data={"audio": (io.BytesIO(mp3_bytes), "noisy_sample.mp3")},
        content_type="multipart/form-data",
    )
    assert upload_res.status_code == 200
    job_id = upload_res.get_json()["job_id"]

    proc_res = client.post(
        "/process",
        json={"job_id": job_id, "method": "spectral_gating"},
    )
    assert proc_res.status_code == 200
    data = proc_res.get_json()
    assert "Spectral Gating" in data["metrics"]["method_used"]
    assert data["metrics"]["snr_available"] is False
    assert data["metrics"]["estimated_noise_reduction_db"] > 0.0


def test_clear_speech_and_multiple_noise_types(client):
    clean = _synthesize_speech_like_signal(sr=16000, duration=1.5)

    # 1. Clear speech test: should remain valid and uncorrupted
    clean_bytes = _save_temp_audio(clean, 16000, fmt="wav")
    up_clean = client.post(
        "/upload",
        data={"audio": (io.BytesIO(clean_bytes), "clear_speech.wav")},
        content_type="multipart/form-data",
    )
    assert up_clean.status_code == 200
    proc_clean = client.post(
        "/process",
        json={"job_id": up_clean.get_json()["job_id"], "method": "auto"},
    )
    assert proc_clean.status_code == 200
    assert proc_clean.get_json()["metrics"]["enhanced_duration_seconds"] == pytest.approx(1.5, abs=0.05)

    # 2. Different background noise types: AC hum + pink fan noise
    for noise_name in ("hum", "pink_fan"):
        noisy = _add_noise(clean, noise_name, seed=31, sr=16000)
        wav_bytes = _save_temp_audio(noisy, 16000, fmt="wav")
        up_res = client.post(
            "/upload",
            data={"audio": (io.BytesIO(wav_bytes), f"{noise_name}_speech.wav")},
            content_type="multipart/form-data",
        )
        assert up_res.status_code == 200
        proc_res = client.post(
            "/process",
            json={"job_id": up_res.get_json()["job_id"], "method": "auto"},
        )
        assert proc_res.status_code == 200
        metrics = proc_res.get_json()["metrics"]
        assert metrics["estimated_noise_reduction_db"] > 0.0


def test_download_missing_returns_404(client):
    res = client.get("/download")
    assert res.status_code == 404
    res_bad = client.get("/download?job_id=nonexistent")
    assert res_bad.status_code == 404
