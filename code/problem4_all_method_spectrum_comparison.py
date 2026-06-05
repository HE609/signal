# -*- coding: utf-8 -*-
"""
Compare every separated male/female output spectrum with the original references.

Outputs:
  images/problem4_all_methods_spectrum_comparison.png
  images/spectrum_comparisons/*.png
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
DETAIL_DIR = IMAGES / "spectrum_comparisons"

SR = 16_000
EPS = 1e-12


METHODS = [
    ("Fixed Butterworth", "butter_male.wav", "butter_female.wav", "fixed_butterworth"),
    ("Fixed Chebyshev-I", "cheby1_male.wav", "cheby1_female.wav", "fixed_chebyshev1"),
    ("Fixed Elliptic", "ellip_male.wav", "ellip_female.wav", "fixed_elliptic"),
    ("Fixed FIR-Hamming", "fir_male.wav", "fir_female.wav", "fixed_fir_hamming"),
    ("Tuned Butterworth", "tuned_butter_male.wav", "tuned_butter_female.wav", "tuned_butterworth"),
    ("Tuned Chebyshev-I", "tuned_cheby1_male.wav", "tuned_cheby1_female.wav", "tuned_chebyshev1"),
    ("Tuned Elliptic", "tuned_ellip_male.wav", "tuned_ellip_female.wav", "tuned_elliptic"),
    ("Tuned FIR-Hamming", "tuned_fir_male.wav", "tuned_fir_female.wav", "tuned_fir_hamming"),
    ("Ref-spectrum Butterworth", "refspec_butter_male.wav", "refspec_butter_female.wav", "refspec_butterworth"),
    ("Ref-spectrum Wiener", "refspec_wiener_male.wav", "refspec_wiener_female.wav", "refspec_wiener"),
    ("Reference-guided", "ref_male.wav", "ref_female.wav", "reference_guided"),
    ("Differentiable filter", "diff_male.wav", "diff_female.wav", "differentiable_filter"),
    ("Trainable ref-guided", "tref_male.wav", "tref_female.wav", "trainable_ref_guided"),
]


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 320


def read_mono(name: str) -> np.ndarray:
    x, sr = sf.read(AUDIO / name, always_2d=True)
    mono = x.mean(axis=1).astype(np.float64)
    if sr != SR:
        raise ValueError(f"{name} has sample rate {sr}, expected {SR}")
    return mono


def match_length(ref: np.ndarray, est: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(ref), len(est))
    return ref[:n], est[:n]


def psd_db(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freqs, pxx = signal.welch(x, fs=SR, nperseg=1024, noverlap=768)
    return freqs, 10.0 * np.log10(pxx + EPS)


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference, estimate = match_length(reference, estimate)
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    scale = np.dot(estimate, reference) / (np.dot(reference, reference) + EPS)
    target = scale * reference
    noise = estimate - target
    return float(10.0 * np.log10((np.sum(target * target) + EPS) / (np.sum(noise * noise) + EPS)))


def corrcoef(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference, estimate = match_length(reference, estimate)
    if np.std(reference) < 1e-8 or np.std(estimate) < 1e-8:
        return 0.0
    return float(np.corrcoef(reference, estimate)[0, 1])


def plot_pair(ax: plt.Axes, reference: np.ndarray, estimate: np.ndarray, title: str, color: str) -> None:
    f_ref, db_ref = psd_db(reference)
    f_est, db_est = psd_db(estimate)
    ax.plot(f_ref, db_ref, color="#9ca3af", linewidth=1.15, alpha=0.95, label="original reference")
    ax.plot(f_est, db_est, color=color, linewidth=1.45, label="separated estimate")
    ax.set_xlim(0, 5000)
    ax.set_ylim(-100, -25)
    ax.set_xlabel("Frequency / Hz")
    ax.set_ylabel("PSD / dB")
    ax.set_title(title, fontweight="bold", fontsize=10.5)
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=8)


def save_detail_figures(male_ref: np.ndarray, female_ref: np.ndarray) -> None:
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    for method, male_file, female_file, slug in METHODS:
        male = read_mono(male_file)
        female = read_mono(female_file)
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
        plot_pair(
            axes[0],
            male_ref,
            male,
            f"{method} male | SI-SDR={si_sdr(male_ref, male):.2f} dB, r={corrcoef(male_ref, male):.3f}",
            "#2563eb",
        )
        plot_pair(
            axes[1],
            female_ref,
            female,
            f"{method} female | SI-SDR={si_sdr(female_ref, female):.2f} dB, r={corrcoef(female_ref, female):.3f}",
            "#db2777",
        )
        fig.suptitle(f"Spectrum comparison: {method}", fontsize=14, fontweight="bold")
        fig.savefig(DETAIL_DIR / f"{slug}.png", bbox_inches="tight")
        plt.close(fig)


def save_combined_figure(male_ref: np.ndarray, female_ref: np.ndarray) -> None:
    rows = len(METHODS)
    fig, axes = plt.subplots(rows, 2, figsize=(14.0, 2.45 * rows), constrained_layout=True)
    fig.suptitle("All separation methods: estimated spectra vs original references", fontsize=17, fontweight="bold")

    for row, (method, male_file, female_file, _) in enumerate(METHODS):
        male = read_mono(male_file)
        female = read_mono(female_file)
        plot_pair(
            axes[row, 0],
            male_ref,
            male,
            f"{row + 1}. {method} male | SI-SDR={si_sdr(male_ref, male):.2f} dB, r={corrcoef(male_ref, male):.3f}",
            "#2563eb",
        )
        plot_pair(
            axes[row, 1],
            female_ref,
            female,
            f"{row + 1}. {method} female | SI-SDR={si_sdr(female_ref, female):.2f} dB, r={corrcoef(female_ref, female):.3f}",
            "#db2777",
        )

    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES / "problem4_all_methods_spectrum_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    male_ref = read_mono("male_speech_ref.wav")
    female_ref = read_mono("female_speech_ref.wav")
    save_combined_figure(male_ref, female_ref)
    save_detail_figures(male_ref, female_ref)
    print(IMAGES / "problem4_all_methods_spectrum_comparison.png")
    print(DETAIL_DIR)


if __name__ == "__main__":
    main()
