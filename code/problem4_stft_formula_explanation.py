# -*- coding: utf-8 -*-
"""
Create an STFT formula and process explanation figure.

Output:
  images/problem4_stft_formula_explanation.png
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
N_FFT = 1024
HOP = 256
EPS = 1e-10


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 320


def read_mono(name: str) -> np.ndarray:
    x, sr = sf.read(AUDIO / name, always_2d=True)
    if sr != SR:
        raise ValueError(f"{name} has sample rate {sr}, expected {SR}")
    return x.mean(axis=1).astype(np.float64)


def stft_np(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs, times, spec = signal.stft(
        x,
        fs=SR,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP,
        window="hann",
        boundary="zeros",
    )
    return freqs, times, spec


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    x = read_mono("mix_speech.wav")
    freqs, times, spec = stft_np(x)
    mag_db = 20.0 * np.log10(np.abs(spec) + EPS)
    mag_db = np.clip(mag_db, np.max(mag_db) - 80.0, np.max(mag_db))

    selected_time = 2.10
    frame_index = int(np.argmin(np.abs(times - selected_time)))
    start = max(0, int(times[frame_index] * SR) - N_FFT // 2)
    end = min(len(x), start + N_FFT)
    frame = np.zeros(N_FFT, dtype=np.float64)
    frame[: end - start] = x[start:end]
    window = signal.windows.hann(N_FFT, sym=False)
    windowed = frame * window
    frame_fft = np.fft.rfft(windowed, n=N_FFT)
    frame_freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    frame_db = 20.0 * np.log10(np.abs(frame_fft) + EPS)

    fig = plt.figure(figsize=(15.0, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    fig.suptitle("Short-Time Fourier Transform (STFT): from waveform to time-frequency map", fontsize=16, fontweight="bold")

    # 1. Waveform and sliding frames.
    ax0 = fig.add_subplot(gs[0, :2])
    t = np.arange(len(x)) / SR
    step = max(1, len(x) // 35000)
    ax0.plot(t[::step], x[::step], color="#111827", linewidth=0.55)
    frame_t0 = start / SR
    frame_t1 = (start + N_FFT) / SR
    hop_time = HOP / SR
    ax0.axvspan(frame_t0, frame_t1, color="#9DB7C7", alpha=0.28, label=f"one frame: N={N_FFT}")
    for offset in [-2, -1, 0, 1, 2]:
        center = times[frame_index] + offset * hop_time
        ax0.axvline(center, color="#C8A27A", linestyle="--", linewidth=0.8, alpha=0.75)
    ax0.set_title("1) Slice the waveform into overlapping short frames", fontweight="bold")
    ax0.set_xlabel("Time / s")
    ax0.set_ylabel("Amplitude")
    ax0.grid(True, alpha=0.22)
    ax0.legend(loc="upper right")

    # 2. Formula panel.
    ax1 = fig.add_subplot(gs[0, 2])
    ax1.axis("off")
    ax1.text(0.02, 0.90, "STFT formula", fontsize=15, fontweight="bold", transform=ax1.transAxes)
    ax1.text(
        0.02,
        0.72,
        "X[m,k] = sum_{n=0}^{N-1}\n"
        "         x[n+mH] w[n] exp(-j*2*pi*k*n/N)",
        fontsize=9.2,
        family="monospace",
        linespacing=1.25,
        transform=ax1.transAxes,
    )
    ax1.text(
        0.02,
        0.38,
        "m: time-frame index\n"
        "k: frequency-bin index\n"
        "w[n]: Hann window\n"
        "N = 1024, H = 256",
        fontsize=10.4,
        linespacing=1.22,
        transform=ax1.transAxes,
    )
    ax1.text(
        0.02,
        0.02,
        "time: t_m = mH / fs\n"
        "freq: f_k = k fs / N",
        fontsize=10.0,
        family="monospace",
        linespacing=1.2,
        transform=ax1.transAxes,
    )

    # 3. Windowed frame.
    ax2 = fig.add_subplot(gs[1, 0])
    local_time = np.arange(N_FFT) / SR * 1000.0
    ax2.plot(local_time, frame, color="#B9B2A7", linewidth=1.0, label="raw frame")
    ax2.plot(local_time, windowed, color="#6F8FA3", linewidth=1.4, label="frame * Hann window")
    ax2.plot(local_time, window * np.max(np.abs(frame) + EPS), color="#C8A27A", linewidth=1.0, alpha=0.8, label="scaled Hann window")
    ax2.set_title("2) Multiply each frame by a Hann window", fontweight="bold")
    ax2.set_xlabel("Local time / ms")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.22)
    ax2.legend(fontsize=8)

    # 4. FFT of selected frame.
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(frame_freqs, frame_db, color="#6F8FA3", linewidth=1.2)
    ax3.set_xlim(0, 5000)
    ax3.set_ylim(np.max(frame_db) - 80.0, np.max(frame_db) + 3.0)
    ax3.set_title("3) FFT gives one frame's frequency content", fontweight="bold")
    ax3.set_xlabel("Frequency / Hz")
    ax3.set_ylabel("Magnitude / dB")
    ax3.grid(True, alpha=0.22)

    # 5. Interpretation panel.
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    ax4.text(0.02, 0.84, "What STFT stores", fontsize=15, fontweight="bold", transform=ax4.transAxes)
    ax4.text(
        0.02,
        0.62,
        r"$X[m,k]$ is complex:",
        fontsize=13,
        transform=ax4.transAxes,
    )
    ax4.text(
        0.08,
        0.43,
        r"$|X[m,k]|$  magnitude / energy" + "\n" + r"$\angle X[m,k]$  phase",
        fontsize=13,
        transform=ax4.transAxes,
    )
    ax4.text(
        0.02,
        0.18,
        "Separation masks usually change the magnitude\n"
        "while reusing the mixture phase for iSTFT.",
        fontsize=11.5,
        transform=ax4.transAxes,
    )

    # 6. Full spectrogram.
    ax5 = fig.add_subplot(gs[2, :])
    im = ax5.imshow(
        mag_db,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap="magma",
        vmin=np.max(mag_db) - 75.0,
        vmax=np.max(mag_db),
    )
    ax5.axvline(times[frame_index], color="#9DB7C7", linestyle="--", linewidth=1.8, label="selected frame")
    ax5.set_ylim(0, 5000)
    ax5.set_title("4) Stacking all frame spectra forms a time-frequency map", fontweight="bold")
    ax5.set_xlabel("Time / s")
    ax5.set_ylabel("Frequency / Hz")
    ax5.legend(loc="upper right")
    fig.colorbar(im, ax=ax5, fraction=0.022, pad=0.012, label="Magnitude / dB")

    fig.savefig(IMAGES / "problem4_stft_formula_explanation.png", bbox_inches="tight")
    plt.close(fig)
    print(IMAGES / "problem4_stft_formula_explanation.png")


if __name__ == "__main__":
    main()
