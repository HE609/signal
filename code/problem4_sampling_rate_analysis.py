# -*- coding: utf-8 -*-
"""
Problem 4(2): sampling-frequency analysis for a 3-second speech signal.

The original assignment says to record 3-second speech using different sampling
frequencies. In this Python implementation, we use the same recorded female
speech and resample it to several sampling rates. This keeps speech content
fixed and changes only the sampling frequency.

Input:
  audio/female_speech.wav

Outputs:
  audio/sampling_16000.wav
  audio/sampling_12000.wav
  audio/sampling_8000.wav
  audio/sampling_4000.wav
  audio/sampling_2000.wav
  images/problem4_sampling_rate_comparison.png
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal


ROOT_DIR = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT_DIR / "audio"
IMAGE_DIR = ROOT_DIR / "images"

SOURCE_SR = 16_000
DURATION_SECONDS = 3.0
SAMPLING_RATES = [16_000, 12_000, 8_000, 4_000, 2_000]
EPS = 1e-12


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 320


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, always_2d=True)
    mono = data.astype(np.float32).mean(axis=1)
    return mono, int(sr)


def resample_to(x: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return x.astype(np.float32)
    gcd = math.gcd(source_sr, target_sr)
    return signal.resample_poly(x, target_sr // gcd, source_sr // gcd).astype(np.float32)


def peak_normalize(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = float(np.max(np.abs(x)) + EPS)
    if max_abs > peak:
        x = x / max_abs * peak
    return x.astype(np.float32)


def plot_waveform(ax: plt.Axes, x: np.ndarray, sr: int, title: str, quality: str, color: str) -> None:
    t = np.arange(len(x)) / sr
    ax.plot(t, x, color=color, linewidth=0.55)
    ax.set_xlim(0, min(DURATION_SECONDS, t[-1] if len(t) else DURATION_SECONDS))
    ax.set_title(f"{title} | {quality}", fontweight="bold")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)


def plot_spectrum(ax: plt.Axes, x: np.ndarray, sr: int, title: str, color: str) -> None:
    nperseg = min(1024, len(x))
    freqs, pxx = signal.welch(x, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    db = 10.0 * np.log10(pxx + EPS)
    ax.plot(freqs, db, color=color, linewidth=0.9)
    ax.axvline(sr / 2, color="black", linestyle="--", linewidth=1.0, alpha=0.5, label="Nyquist")
    ax.set_xlim(0, min(8000, max(SAMPLING_RATES) / 2))
    ax.set_ylim(np.max(db) - 90, np.max(db) + 5)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Frequency / Hz")
    ax.set_ylabel("PSD / dB")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")


def quality_label(sr: int) -> tuple[str, str]:
    if sr >= 8_000:
        return "基本不失真", "#2563eb"
    if sr >= 4_000:
        return "高频明显损失", "#d97706"
    return "严重失真", "#dc2626"


def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    source_path = AUDIO_DIR / "female_speech.wav"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source speech: {source_path}")

    source, sr = read_mono(source_path)
    source = resample_to(source, sr, SOURCE_SR)
    n = int(DURATION_SECONDS * SOURCE_SR)
    if len(source) < n:
        source = np.pad(source, (0, n - len(source)))
    speech_3s = peak_normalize(source[:n])

    fig, axes = plt.subplots(len(SAMPLING_RATES), 2, figsize=(16, 3.2 * len(SAMPLING_RATES)), constrained_layout=True)
    fig.suptitle("Problem 4(2): Waveforms and spectra under different sampling frequencies", fontsize=16, fontweight="bold")

    for row, target_sr in enumerate(SAMPLING_RATES):
        sampled = resample_to(speech_3s, SOURCE_SR, target_sr)
        sampled = peak_normalize(sampled)
        sf.write(AUDIO_DIR / f"sampling_{target_sr}.wav", sampled, target_sr)

        quality, color = quality_label(target_sr)
        plot_waveform(axes[row, 0], sampled, target_sr, f"fs = {target_sr} Hz waveform", quality, color)
        plot_spectrum(axes[row, 1], sampled, target_sr, f"fs = {target_sr} Hz spectrum", color)

    fig.savefig(IMAGE_DIR / "problem4_sampling_rate_comparison.png", bbox_inches="tight")
    plt.close(fig)

    print("Saved sampling audio files:")
    for target_sr in SAMPLING_RATES:
        print(f"  audio/sampling_{target_sr}.wav")
    print("Saved image: images/problem4_sampling_rate_comparison.png")
    print("Conclusion: speech is usually intelligible without obvious distortion at about 8000 Hz; 4000 Hz and below lose high-frequency detail.")


if __name__ == "__main__":
    main()
