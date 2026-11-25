"""Waveform and Spectrogram visualization generator using Matplotlib."""

from __future__ import annotations

import os
from typing import Dict

import librosa
import librosa.display
import matplotlib
import numpy as np

# Use non-interactive backend for safe server-side image rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


BG_COLOR = "#090e1f"
PANEL_COLOR = "#0f172e"
TEXT_COLOR = "#e2e8f0"
GRID_COLOR = "#1e293b"


def _style_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(PANEL_COLOR)
    ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color="#94a3b8", fontsize=9)
    ax.set_ylabel(ylabel, color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.7)


def _plot_waveform(
    audio: np.ndarray,
    sample_rate: int,
    output_path: str,
    title: str,
    line_color: str,
    y_limit: float,
) -> None:
    # Downsample long waveforms purely for crisp, fast plotting
    max_points = 12000
    if len(audio) > max_points:
        step = len(audio) // max_points
        plot_y = audio[::step]
        times = np.linspace(0.0, len(audio) / sample_rate, num=len(plot_y))
    else:
        plot_y = audio
        times = np.linspace(0.0, len(audio) / sample_rate, num=len(plot_y))

    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=130)
    fig.patch.set_facecolor(BG_COLOR)
    ax.plot(times, plot_y, color=line_color, linewidth=0.85, alpha=0.95)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlim(0.0, max(times[-1] if times.size else 0.1, 0.01))
    _style_axis(ax, title, "Time (seconds)", "Amplitude")
    fig.tight_layout()
    fig.savefig(output_path, facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)


def _plot_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    output_path: str,
    title: str,
    ref_peak: float,
) -> None:
    n_fft = min(1024, max(256, len(audio)))
    hop_length = max(64, n_fft // 4)
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    db_spec = librosa.amplitude_to_db(np.abs(stft), ref=max(ref_peak, 1e-6), top_db=80.0)

    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=130)
    fig.patch.set_facecolor(BG_COLOR)
    img = librosa.display.specshow(
        db_spec,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        vmin=-80.0,
        vmax=0.0,
        ax=ax,
    )
    _style_axis(ax, title, "Time (seconds)", "Frequency (Hz)")
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.tick_params(colors="#94a3b8", labelsize=8)
    cbar.outline.set_edgecolor("#334155")
    fig.tight_layout()
    fig.savefig(output_path, facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)


def generate_comparison_plots(
    original_audio: np.ndarray,
    original_sr: int,
    enhanced_audio: np.ndarray,
    enhanced_sr: int,
    output_dir: str,
    job_id: str,
) -> Dict[str, str]:
    """Generate waveform and spectrogram PNG files for original and enhanced audio."""
    os.makedirs(output_dir, exist_ok=True)

    shared_peak = max(
        float(np.max(np.abs(original_audio))) if original_audio.size else 0.1,
        float(np.max(np.abs(enhanced_audio))) if enhanced_audio.size else 0.1,
        0.05,
    )
    y_limit = min(1.05, shared_peak * 1.1)

    # Shared STFT reference peak so both spectrograms use the same dB scale
    orig_stft_peak = float(np.max(np.abs(librosa.stft(original_audio)))) if original_audio.size else 1.0
    enh_stft_peak = float(np.max(np.abs(librosa.stft(enhanced_audio)))) if enhanced_audio.size else 1.0
    shared_spec_ref = max(orig_stft_peak, enh_stft_peak, 1e-6)

    filenames = {
        "original_waveform": f"{job_id}_orig_waveform.png",
        "enhanced_waveform": f"{job_id}_enh_waveform.png",
        "original_spectrogram": f"{job_id}_orig_spectrogram.png",
        "enhanced_spectrogram": f"{job_id}_enh_spectrogram.png",
    }

    _plot_waveform(
        original_audio,
        original_sr,
        os.path.join(output_dir, filenames["original_waveform"]),
        "Original Noisy Speech — Waveform",
        line_color="#60a5fa",
        y_limit=y_limit,
    )
    _plot_waveform(
        enhanced_audio,
        enhanced_sr,
        os.path.join(output_dir, filenames["enhanced_waveform"]),
        "AI-Enhanced Speech — Waveform",
        line_color="#a78bfa",
        y_limit=y_limit,
    )
    _plot_spectrogram(
        original_audio,
        original_sr,
        os.path.join(output_dir, filenames["original_spectrogram"]),
        "Original Noisy Speech — Spectrogram",
        ref_peak=shared_spec_ref,
    )
    _plot_spectrogram(
        enhanced_audio,
        enhanced_sr,
        os.path.join(output_dir, filenames["enhanced_spectrogram"]),
        "AI-Enhanced Speech — Spectrogram",
        ref_peak=shared_spec_ref,
    )

    return filenames
