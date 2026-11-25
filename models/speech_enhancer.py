"""AI Speech Enhancement Pipeline.

Primary engine: Pretrained PyTorch Demucs DNS64 speech enhancement model
(`denoiser.pretrained.dns64`), which operates directly on 16 kHz mono waveforms
to suppress stationary and non-stationary background noise while preserving
human speech harmonics.

Fallback engine: Adaptive Spectral Gating using `noisereduce` and `librosa`
short-time Fourier transform (STFT) noise-floor estimation.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Tuple

import librosa
import noisereduce as nr
import numpy as np
import torch

logger = logging.getLogger(__name__)


class SpeechEnhancer:
    """Thread-safe speech enhancement engine with Deep Learning & Spectral Gating."""

    TARGET_SAMPLE_RATE = 16000
    CHUNK_SECONDS = 15.0
    OVERLAP_SECONDS = 0.5

    def __init__(self, model_cache_dir: str | None = None) -> None:
        self._lock = threading.Lock()
        self._dns64_model = None
        self._dns64_checked = False
        self._dns64_error: str | None = None
        self.device = torch.device("cpu")
        if model_cache_dir:
            os.makedirs(model_cache_dir, exist_ok=True)
            torch.hub.set_dir(model_cache_dir)

    def _load_dns64(self):
        """Lazily load and cache the pretrained Facebook Demucs DNS64 model."""
        if self._dns64_checked:
            return self._dns64_model

        with self._lock:
            if self._dns64_checked:
                return self._dns64_model
            try:
                from denoiser import pretrained

                model = pretrained.dns64().to(self.device)
                model.eval()
                self._dns64_model = model
                self._dns64_error = None
                logger.info("Loaded pretrained Demucs DNS64 model successfully.")
            except Exception as exc:  # pragma: no cover - defensive fallback
                self._dns64_model = None
                self._dns64_error = str(exc)
                logger.warning(
                    "Could not load Demucs DNS64 model (%s). Will use Spectral Gating fallback.",
                    exc,
                )
            finally:
                self._dns64_checked = True
        return self._dns64_model

    def is_deep_learning_available(self) -> bool:
        """Return True if the pretrained PyTorch DNS64 model is ready."""
        return self._load_dns64() is not None

    def primary_model_label(self) -> str:
        """Human-readable label describing the active primary model."""
        if self.is_deep_learning_available():
            return "PyTorch Demucs DNS64"
        return "Adaptive Spectral Gating (Librosa)"

    def enhance(
        self,
        audio: np.ndarray,
        sample_rate: int,
        method: str = "auto",
    ) -> Tuple[np.ndarray, int, str]:
        """Enhance speech audio by suppressing background noise.

        Args:
            audio: 1-D float32 mono waveform in [-1.0, 1.0].
            sample_rate: Original sample rate of `audio`.
            method: One of "auto", "deep_learning", or "spectral_gating".

        Returns:
            Tuple of (enhanced_waveform, output_sample_rate, method_used_label).
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        audio = np.asarray(audio, dtype=np.float32)

        if audio.size == 0:
            raise ValueError("Cannot process an empty audio signal.")

        normalized_method = (method or "auto").strip().lower()
        if normalized_method not in {"auto", "deep_learning", "spectral_gating"}:
            raise ValueError(
                f"Unsupported enhancement method '{method}'. "
                "Choose 'auto', 'deep_learning', or 'spectral_gating'."
            )

        if normalized_method in {"auto", "deep_learning"}:
            model = self._load_dns64()
            if model is not None:
                enhanced = self._enhance_with_dns64(audio, sample_rate, model)
                return (
                    enhanced,
                    self.TARGET_SAMPLE_RATE,
                    "Deep Learning (PyTorch Demucs DNS64)",
                )
            if normalized_method == "deep_learning":
                logger.warning(
                    "Deep learning model unavailable (%s); falling back to Spectral Gating.",
                    self._dns64_error,
                )

        enhanced = self._enhance_with_spectral_gating(audio, sample_rate)
        fallback_note = (
            "Adaptive Spectral Gating (Fallback)"
            if normalized_method == "deep_learning"
            else "Adaptive Spectral Gating (Librosa + NoiseReduce)"
        )
        return enhanced, sample_rate, fallback_note

    def _enhance_with_dns64(
        self,
        audio: np.ndarray,
        sample_rate: int,
        model,
    ) -> np.ndarray:
        """Run pretrained Demucs DNS64 enhancement in overlapping chunks."""
        if sample_rate != self.TARGET_SAMPLE_RATE:
            audio_16k = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=self.TARGET_SAMPLE_RATE
            )
        else:
            audio_16k = audio

        audio_16k = np.asarray(audio_16k, dtype=np.float32)
        peak = float(np.max(np.abs(audio_16k)))
        if peak > 1.0:
            audio_16k = audio_16k / peak

        chunk_samples = int(self.CHUNK_SECONDS * self.TARGET_SAMPLE_RATE)
        overlap_samples = int(self.OVERLAP_SECONDS * self.TARGET_SAMPLE_RATE)

        if len(audio_16k) <= chunk_samples:
            enhanced = self._infer_chunk(audio_16k, model)
        else:
            enhanced = self._infer_chunked(
                audio_16k, model, chunk_samples, overlap_samples
            )

        # Gentle peak protection to avoid clipping when exporting WAV
        out_peak = float(np.max(np.abs(enhanced)))
        if out_peak > 0.99:
            enhanced = enhanced * (0.98 / out_peak)

        return np.asarray(enhanced, dtype=np.float32)

    def _infer_chunk(self, chunk: np.ndarray, model) -> np.ndarray:
        """Run a single forward pass through Demucs DNS64 without gradients."""
        tensor = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            estimate = model(tensor)
        out = estimate.squeeze(0).squeeze(0).cpu().numpy()
        # Match input chunk length exactly
        if len(out) > len(chunk):
            out = out[: len(chunk)]
        elif len(out) < len(chunk):
            out = np.pad(out, (0, len(chunk) - len(out)))
        return out

    def _infer_chunked(
        self,
        audio_16k: np.ndarray,
        model,
        chunk_samples: int,
        overlap_samples: int,
    ) -> np.ndarray:
        """Process long recordings using overlap-add crossfading."""
        step = max(1, chunk_samples - overlap_samples)
        output = np.zeros_like(audio_16k, dtype=np.float32)
        weight = np.zeros_like(audio_16k, dtype=np.float32)

        for start in range(0, len(audio_16k), step):
            end = min(start + chunk_samples, len(audio_16k))
            chunk = audio_16k[start:end]
            denoised = self._infer_chunk(chunk, model)

            win = np.ones(len(denoised), dtype=np.float32)
            if start > 0 and overlap_samples > 0:
                fade_len = min(overlap_samples, len(win))
                win[:fade_len] = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            if end < len(audio_16k) and overlap_samples > 0:
                fade_len = min(overlap_samples, len(win))
                win[-fade_len:] = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

            output[start:end] += denoised * win
            weight[start:end] += win

            if end == len(audio_16k):
                break

        nonzero = weight > 1e-8
        output[nonzero] /= weight[nonzero]
        return output

    def _enhance_with_spectral_gating(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Adaptive non-stationary + stationary spectral gating fallback."""
        # Non-stationary adaptive noise reduction preserves speech transients
        reduced = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=False,
            prop_decrease=0.88,
            n_fft=1024,
            hop_length=256,
        )
        # Light stationary pass to clean residual constant hum/hiss
        reduced = nr.reduce_noise(
            y=reduced,
            sr=sample_rate,
            stationary=True,
            prop_decrease=0.55,
            n_fft=1024,
            hop_length=256,
        )
        out_peak = float(np.max(np.abs(reduced)))
        if out_peak > 0.99:
            reduced = reduced * (0.98 / out_peak)
        return np.asarray(reduced, dtype=np.float32)
