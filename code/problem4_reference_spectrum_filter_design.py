# -*- coding: utf-8 -*-
"""
Reference-spectrum traditional filter design for Problem 4.

This script uses the known original male/female speech references to estimate
their average spectra, then designs traditional DSP filters from those spectra.
It does not train or change any neural model.

Outputs:
  audio/refspec_butter_male.wav
  audio/refspec_butter_female.wav
  audio/refspec_wiener_male.wav
  audio/refspec_wiener_female.wav
  images/problem4_reference_spectrum_filter_metrics.csv
  images/problem4_reference_spectrum_filter_response.png
  images/problem4_reference_spectrum_filter_comparison.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from problem4_speech_method_comparison import (
    AUDIO_DIR,
    EPS,
    HOP,
    IMAGE_DIR,
    N_FFT,
    SR,
    corrcoef,
    peak_limit,
    prepare_reference_mix,
    si_sdr,
    snr_db,
    spectral_error,
    write_wav,
)


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 320


def smooth_1d(x: np.ndarray, bins: int) -> np.ndarray:
    bins = max(3, int(bins))
    if bins % 2 == 0:
        bins += 1
    kernel = np.ones(bins, dtype=np.float64) / bins
    return np.convolve(x, kernel, mode="same")


def reference_psd(x: np.ndarray, smooth_bins: int = 21) -> tuple[np.ndarray, np.ndarray]:
    freqs, psd = signal.welch(x, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
    log_psd = np.log(psd + EPS)
    smoothed = np.exp(smooth_1d(log_psd, smooth_bins))
    return freqs, smoothed.astype(np.float64)


def estimate_crossover(freqs: np.ndarray, male_psd: np.ndarray, female_psd: np.ndarray) -> float:
    ratio_db = 10.0 * np.log10((male_psd + EPS) / (female_psd + EPS))
    search = (freqs >= 300.0) & (freqs <= 3500.0)
    f = freqs[search]
    r = smooth_1d(ratio_db[search], 9)

    sign_change = np.where(np.diff(np.signbit(r)))[0]
    if sign_change.size:
        idx = int(sign_change[0])
        f0, f1 = float(f[idx]), float(f[idx + 1])
        r0, r1 = float(r[idx]), float(r[idx + 1])
        if abs(r1 - r0) < EPS:
            return f0
        return f0 - r0 * (f1 - f0) / (r1 - r0)

    return float(f[np.argmin(np.abs(r))])


def reference_crossover_butterworth(
    mix: np.ndarray,
    cutoff: float,
    order: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    sos_low = signal.butter(order, cutoff, btype="lowpass", fs=SR, output="sos")
    sos_high = signal.butter(order, cutoff, btype="highpass", fs=SR, output="sos")
    male = peak_limit(signal.sosfiltfilt(sos_low, mix).astype(np.float32))
    female = peak_limit(signal.sosfiltfilt(sos_high, mix).astype(np.float32))
    return male, female


def reference_wiener_filter(
    mix: np.ndarray,
    psd_freqs: np.ndarray,
    male_psd: np.ndarray,
    female_psd: np.ndarray,
    alpha: float = 0.8,
    mask_smooth_bins: int = 17,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    freqs, _, mix_stft = signal.stft(mix, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="zeros")
    male_interp = np.interp(freqs, psd_freqs, male_psd)
    female_interp = np.interp(freqs, psd_freqs, female_psd)
    male_weight = np.power(male_interp + EPS, alpha)
    female_weight = np.power(female_interp + EPS, alpha)
    male_mask = male_weight / (male_weight + female_weight + EPS)
    male_mask = smooth_1d(male_mask, mask_smooth_bins)
    male_mask = np.clip(male_mask, 0.02, 0.98)
    female_mask = 1.0 - male_mask

    _, male = signal.istft(mix_stft * male_mask[:, None], fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
    _, female = signal.istft(mix_stft * female_mask[:, None], fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
    male = peak_limit(male[: len(mix)].astype(np.float32))
    female = peak_limit(female[: len(mix)].astype(np.float32))
    return male, female, freqs, male_mask, female_mask


def metric_rows(
    male_ref: np.ndarray,
    female_ref: np.ndarray,
    mix: np.ndarray,
    outputs: list[tuple[str, np.ndarray, np.ndarray]],
    cutoff: float,
) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for method, male_est, female_est in outputs:
        rec_err = float(np.linalg.norm(mix - (male_est + female_est)) / (np.linalg.norm(mix) + EPS))
        for stem, ref, est in [("male", male_ref, male_est), ("female", female_ref, female_est)]:
            rows.append(
                {
                    "method": method,
                    "stem": stem,
                    "reference_crossover_hz": cutoff,
                    "si_sdr_db": si_sdr(ref, est),
                    "snr_db": snr_db(ref, est),
                    "corr": corrcoef(ref, est),
                    "mse": float(np.mean(np.square(ref - est))),
                    "spectral_l1": spectral_error(ref, est),
                    "reconstruction_error": rec_err,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_response(
    freqs: np.ndarray,
    male_psd: np.ndarray,
    female_psd: np.ndarray,
    cutoff: float,
    mask_freqs: np.ndarray,
    male_mask: np.ndarray,
    female_mask: np.ndarray,
) -> None:
    ratio_db = 10.0 * np.log10((male_psd + EPS) / (female_psd + EPS))

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.2), constrained_layout=True)
    axes[0].plot(freqs, 10.0 * np.log10(male_psd + EPS), label="male reference PSD", color="#2563eb", linewidth=1.8)
    axes[0].plot(freqs, 10.0 * np.log10(female_psd + EPS), label="female reference PSD", color="#db2777", linewidth=1.8)
    axes[0].axvline(cutoff, color="#111827", linestyle="--", linewidth=1.2, label=f"crossover = {cutoff:.1f} Hz")
    axes[0].set_xlim(0, 5000)
    axes[0].set_title("Reference spectra and estimated crossover frequency", fontweight="bold")
    axes[0].set_xlabel("Frequency / Hz")
    axes[0].set_ylabel("Power / dB")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(freqs, ratio_db, label="male/female PSD ratio", color="#0f766e", linewidth=1.4)
    axes[1].plot(mask_freqs, male_mask, label="reference Wiener male response", color="#2563eb", linewidth=1.8)
    axes[1].plot(mask_freqs, female_mask, label="reference Wiener female response", color="#db2777", linewidth=1.8)
    axes[1].axvline(cutoff, color="#111827", linestyle="--", linewidth=1.2)
    axes[1].axhline(0.0, color="#6b7280", linestyle=":", linewidth=0.9)
    axes[1].set_xlim(0, 5000)
    axes[1].set_title("Reference-derived filter response", fontweight="bold")
    axes[1].set_xlabel("Frequency / Hz")
    axes[1].set_ylabel("Ratio / Mask")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.savefig(IMAGE_DIR / "problem4_reference_spectrum_filter_response.png", bbox_inches="tight")
    plt.close(fig)


def plot_comparison(outputs: list[tuple[str, np.ndarray, np.ndarray]], male_ref: np.ndarray, female_ref: np.ndarray) -> None:
    fig, axes = plt.subplots(len(outputs), 2, figsize=(12.0, 4.0 * len(outputs)), constrained_layout=True)
    if len(outputs) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (method, male_est, female_est) in enumerate(outputs):
        for col, stem, ref, est, color in [
            (0, "male", male_ref, male_est, "#2563eb"),
            (1, "female", female_ref, female_est, "#db2777"),
        ]:
            ax = axes[row, col]
            idx = np.linspace(0, len(ref) - 1, min(len(ref), 20_000)).astype(int)
            t = idx / SR
            ax.plot(t, ref[idx], color="#9ca3af", linewidth=0.45, label="reference")
            ax.plot(t, est[idx], color=color, linewidth=0.55, alpha=0.85, label="estimate")
            ax.set_title(f"{method} {stem}: SI-SDR={si_sdr(ref, est):.2f} dB, Corr={corrcoef(ref, est):.3f}", fontweight="bold")
            ax.set_xlabel("Time / s")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.2)
            ax.legend()

    fig.savefig(IMAGE_DIR / "problem4_reference_spectrum_filter_comparison.png", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design traditional filters from reference male/female spectra.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Reference Wiener mask sharpness.")
    parser.add_argument("--smooth-bins", type=int, default=21, help="Reference PSD smoothing bins.")
    parser.add_argument("--mask-smooth-bins", type=int, default=17, help="Wiener response smoothing bins.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    male_ref, female_ref, mix = prepare_reference_mix()
    freqs, male_psd = reference_psd(male_ref, args.smooth_bins)
    _, female_psd = reference_psd(female_ref, args.smooth_bins)
    cutoff = estimate_crossover(freqs, male_psd, female_psd)

    butter_male, butter_female = reference_crossover_butterworth(mix, cutoff)
    wiener_male, wiener_female, mask_freqs, male_mask, female_mask = reference_wiener_filter(
        mix,
        freqs,
        male_psd,
        female_psd,
        alpha=args.alpha,
        mask_smooth_bins=args.mask_smooth_bins,
    )

    write_wav(AUDIO_DIR / "refspec_butter_male.wav", butter_male)
    write_wav(AUDIO_DIR / "refspec_butter_female.wav", butter_female)
    write_wav(AUDIO_DIR / "refspec_wiener_male.wav", wiener_male)
    write_wav(AUDIO_DIR / "refspec_wiener_female.wav", wiener_female)
    write_wav(AUDIO_DIR / "refspec_wiener_reconstructed_mix.wav", wiener_male + wiener_female)

    outputs = [
        ("ReferenceCrossoverButterworth", butter_male, butter_female),
        ("ReferenceSpectrumWiener", wiener_male, wiener_female),
    ]
    rows = metric_rows(male_ref, female_ref, mix, outputs, cutoff)
    write_csv(IMAGE_DIR / "problem4_reference_spectrum_filter_metrics.csv", rows)
    plot_response(freqs, male_psd, female_psd, cutoff, mask_freqs, male_mask, female_mask)
    plot_comparison(outputs, male_ref, female_ref)

    print(f"Estimated reference crossover cutoff: {cutoff:.2f} Hz")
    for method, male, female in outputs:
        print(
            f"{method}: male SI-SDR={si_sdr(male_ref, male):.2f} dB, Corr={corrcoef(male_ref, male):.4f}; "
            f"female SI-SDR={si_sdr(female_ref, female):.2f} dB, Corr={corrcoef(female_ref, female):.4f}"
        )
    print(f"Saved metrics: {IMAGE_DIR / 'problem4_reference_spectrum_filter_metrics.csv'}")


if __name__ == "__main__":
    main()
