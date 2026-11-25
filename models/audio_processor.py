"""Audio I/O, validation, and objective performance metrics."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import librosa
import numpy as np
import soundfile as sf

ALLOWED_EXTENSIONS = {".wav", ".mp3"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


def validate_audio_file(file_path: str, original_filename: str | None = None) -> Tuple[bool, str]:
    """Validate extension, non-empty size, and decodable audio content."""
    filename = original_filename or os.path.basename(file_path)
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        return False, "Unsupported file format. Please upload a .wav or .mp3 audio file."

    if not os.path.exists(file_path):
        return False, "Uploaded file could not be found on disk."

    size = os.path.getsize(file_path)
    if size == 0:
        return False, "Uploaded audio file is empty (0 bytes)."
    if size > MAX_FILE_SIZE_BYTES:
        return False, "File exceeds the 50 MB maximum size limit."

    try:
        audio, sr = load_audio(file_path)
        if audio.size == 0 or sr <= 0:
            return False, "Audio file contains no valid audio samples."
    except Exception as exc:
        return False, f"Invalid or corrupted audio file: {exc}"

    return True, "Valid audio file."


def load_audio(file_path: str, target_sr: int | None = None) -> Tuple[np.ndarray, int]:
    """Load WAV or MP3 as mono float32 waveform."""
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return np.asarray(audio, dtype=np.float32), int(sr)


def save_wav(file_path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Save waveform as 16-bit PCM WAV file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    sf.write(file_path, clipped, sample_rate, subtype="PCM_16")


def _estimate_noise_floor_rms(audio: np.ndarray, sample_rate: int) -> float:
    """Estimate background noise floor RMS from the lowest-energy 20% of frames."""
    frame_length = max(256, int(0.025 * sample_rate))
    hop_length = max(128, int(0.010 * sample_rate))
    if len(audio) < frame_length:
        return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))

    rms_frames = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]
    if rms_frames.size == 0:
        return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))

    # Lowest 20% of frames approximate background noise regions between speech
    threshold = np.percentile(rms_frames, 20)
    quiet_frames = rms_frames[rms_frames <= threshold]
    if quiet_frames.size == 0:
        quiet_frames = rms_frames
    return float(np.mean(quiet_frames) + 1e-9)


def _compute_snr_db(clean_ref: np.ndarray, signal: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio (dB) against a clean reference signal."""
    min_len = min(len(clean_ref), len(signal))
    if min_len == 0:
        return 0.0
    ref = clean_ref[:min_len]
    sig = signal[:min_len]
    noise = sig - ref
    ref_power = float(np.sum(np.square(ref))) + 1e-12
    noise_power = float(np.sum(np.square(noise))) + 1e-12
    return float(10.0 * np.log10(ref_power / noise_power))


def compute_metrics(
    original_audio: np.ndarray,
    original_sr: int,
    enhanced_audio: np.ndarray,
    enhanced_sr: int,
    processing_time_seconds: float,
    method_used: str,
    reference_audio_path: str | None = None,
) -> Dict[str, Any]:
    """Compute non-fabricated audio and enhancement performance metrics."""
    orig_duration = float(len(original_audio) / original_sr)
    enh_duration = float(len(enhanced_audio) / enhanced_sr)

    # Align sample rates for fair noise-floor comparison
    if original_sr != enhanced_sr:
        orig_aligned = librosa.resample(
            original_audio, orig_sr=original_sr, target_sr=enhanced_sr
        )
    else:
        orig_aligned = original_audio

    noise_before = _estimate_noise_floor_rms(orig_aligned, enhanced_sr)
    noise_after = _estimate_noise_floor_rms(enhanced_audio, enhanced_sr)
    estimated_noise_reduction_db = float(
        20.0 * np.log10(max(noise_before, 1e-9) / max(noise_after, 1e-9))
    )

    metrics: Dict[str, Any] = {
        "original_duration_seconds": round(orig_duration, 3),
        "enhanced_duration_seconds": round(enh_duration, 3),
        "processing_time_seconds": round(float(processing_time_seconds), 3),
        "estimated_noise_reduction_db": round(estimated_noise_reduction_db, 2),
        "method_used": method_used,
        "snr_available": False,
        "snr_before_db": None,
        "snr_after_db": None,
        "snr_improvement_db": None,
        "snr_note": "SNR requires a clean reference signal.",
    }

    if reference_audio_path and os.path.exists(reference_audio_path):
        try:
            ref_audio, _ = load_audio(reference_audio_path, target_sr=enhanced_sr)
            if ref_audio.size > 0:
                snr_before = _compute_snr_db(ref_audio, orig_aligned)
                snr_after = _compute_snr_db(ref_audio, enhanced_audio)
                metrics["snr_available"] = True
                metrics["snr_before_db"] = round(snr_before, 2)
                metrics["snr_after_db"] = round(snr_after, 2)
                metrics["snr_improvement_db"] = round(snr_after - snr_before, 2)
                metrics["snr_note"] = "Computed against uploaded clean reference audio."
        except Exception:
            metrics["snr_note"] = "Could not evaluate reference audio for SNR."

    return metrics
