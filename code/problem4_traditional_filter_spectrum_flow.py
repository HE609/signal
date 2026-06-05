# -*- coding: utf-8 -*-
"""
Visualize how a traditional low-pass/high-pass filter pair acts on spectra.

Outputs:
  images/problem4_traditional_filter_spectrum_flow.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal


ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "audio"
IMAGES = ROOT / "images"

SR = 16_000
EPS = 1e-12


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 320


def read_mono(name: str) -> np.ndarray:
    x, _ = sf.read(AUDIO / name, always_2d=True)
    return x.mean(axis=1).astype(np.float64)


def psd_db(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f, p = signal.welch(x, fs=SR, nperseg=1024, noverlap=768)
    return f, 10.0 * np.log10(p + EPS)


def plot_psd(ax: plt.Axes, x: np.ndarray, label: str, color: str, alpha: float = 1.0, lw: float = 1.5) -> None:
    f, db = psd_db(x)
    ax.plot(f, db, label=label, color=color, alpha=alpha, linewidth=lw)
    ax.set_xlim(0, 5000)
    ax.set_ylim(-95, -25)
    ax.set_xlabel("Frequency / Hz")
    ax.set_ylabel("PSD / dB")
    ax.grid(True, alpha=0.25)


def main() -> None:
    male = read_mono("male_speech_ref.wav")
    female = read_mono("female_speech_ref.wav")
    mix = read_mono("mix_speech.wav")

    # Use the tuned best Butterworth cutoff pair from the grid-search result.
    male_cutoff = 900.0
    female_cutoff = 1000.0
    sos_low = signal.butter(6, male_cutoff, btype="lowpass", fs=SR, output="sos")
    sos_high = signal.butter(6, female_cutoff, btype="highpass", fs=SR, output="sos")
    est_male = signal.sosfiltfilt(sos_low, mix)
    est_female = signal.sosfiltfilt(sos_high, mix)

    freqs = np.linspace(0, 5000, 1600)
    _, h_low = signal.sosfreqz(sos_low, worN=freqs, fs=SR)
    _, h_high = signal.sosfreqz(sos_high, worN=freqs, fs=SR)
    h_low_db = 20.0 * np.log10(np.abs(h_low) + EPS)
    h_high_db = 20.0 * np.log10(np.abs(h_high) + EPS)

    fig, axes = plt.subplots(3, 2, figsize=(14.0, 11.0), constrained_layout=True)
    fig.suptitle("Traditional filter separation in the frequency domain", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    plot_psd(ax, male, "male reference", "#2563eb")
    plot_psd(ax, female, "female reference", "#db2777")
    ax.set_title("1) Original reference spectra", fontweight="bold")
    ax.legend()

    ax = axes[0, 1]
    plot_psd(ax, mix, "mixture = male + female", "#111827", lw=1.7)
    plot_psd(ax, male, "male reference", "#2563eb", alpha=0.45, lw=1.0)
    plot_psd(ax, female, "female reference", "#db2777", alpha=0.45, lw=1.0)
    ax.set_title("2) Mixed spectrum: spectra are added, not chemically fused", fontweight="bold")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(freqs, h_low_db, color="#2563eb", linewidth=2.0, label=f"LPF for male, fc={male_cutoff:.0f} Hz")
    ax.plot(freqs, h_high_db, color="#db2777", linewidth=2.0, label=f"HPF for female, fc={female_cutoff:.0f} Hz")
    ax.axvline(male_cutoff, color="#2563eb", linestyle="--", alpha=0.75)
    ax.axvline(female_cutoff, color="#db2777", linestyle="--", alpha=0.75)
    ax.set_xlim(0, 5000)
    ax.set_ylim(-75, 5)
    ax.set_xlabel("Frequency / Hz")
    ax.set_ylabel("Gain / dB")
    ax.set_title("3) Filter responses: soft attenuation around cutoff", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    f, mix_db = psd_db(mix)
    mix_interp = np.interp(freqs, f, mix_db)
    ax.plot(freqs, mix_interp, color="#111827", linewidth=1.25, label="mixture PSD")
    ax.plot(freqs, mix_interp + h_low_db, color="#2563eb", linewidth=1.6, label="mixture PSD × LPF")
    ax.plot(freqs, mix_interp + h_high_db, color="#db2777", linewidth=1.6, label="mixture PSD × HPF")
    ax.set_xlim(0, 5000)
    ax.set_ylim(-110, -25)
    ax.set_xlabel("Frequency / Hz")
    ax.set_ylabel("Approx. output PSD / dB")
    ax.set_title("4) Filtering means multiplying the mixed spectrum by H(f)", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[2, 0]
    plot_psd(ax, male, "male reference", "#93a4b7", alpha=0.85, lw=1.2)
    plot_psd(ax, est_male, "filtered male estimate", "#2563eb", lw=1.8)
    ax.set_title("5) Male estimate after LPF", fontweight="bold")
    ax.legend()

    ax = axes[2, 1]
    plot_psd(ax, female, "female reference", "#c9a7a2", alpha=0.85, lw=1.2)
    plot_psd(ax, est_female, "filtered female estimate", "#db2777", lw=1.8)
    ax.set_title("6) Female estimate after HPF", fontweight="bold")
    ax.legend()

    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES / "problem4_traditional_filter_spectrum_flow.png", bbox_inches="tight")
    plt.close(fig)
    print(IMAGES / "problem4_traditional_filter_spectrum_flow.png")


if __name__ == "__main__":
    main()
