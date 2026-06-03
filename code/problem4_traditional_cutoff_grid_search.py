# -*- coding: utf-8 -*-
"""
Grid-search traditional filter cutoff frequencies for Problem 4.

This script does not change the neural methods. It only evaluates traditional
low-pass/high-pass filter pairs under many male/female cutoff combinations.

Outputs:
  images/problem4_traditional_cutoff_grid_search.csv
  images/problem4_traditional_cutoff_best.csv
  images/problem4_traditional_cutoff_heatmap.png
  images/problem4_traditional_cutoff_best_bars.png
  audio/tuned_<method>_male.wav
  audio/tuned_<method>_female.wav
  audio/tuned_best_male.wav
  audio/tuned_best_female.wav
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
    IMAGE_DIR,
    SR,
    corrcoef,
    peak_limit,
    prepare_reference_mix,
    si_sdr,
    snr_db,
    spectral_error,
    write_wav,
)


IIR_METHODS = ("Butterworth", "Chebyshev-I", "Elliptic")
ALL_METHODS = ("Butterworth", "Chebyshev-I", "Elliptic", "FIR-Hamming")
PREFIX = {
    "Butterworth": "butter",
    "Chebyshev-I": "cheby1",
    "Elliptic": "ellip",
    "FIR-Hamming": "fir",
}


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 320


def cutoff_values(start: float, stop: float, step: float) -> np.ndarray:
    count = int(np.floor((stop - start) / step)) + 1
    return np.round(start + step * np.arange(count), 6)


def separate_with_filter(method: str, mix: np.ndarray, male_cutoff: float, female_cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    if method == "Butterworth":
        low = signal.butter(6, male_cutoff, btype="lowpass", fs=SR, output="sos")
        high = signal.butter(6, female_cutoff, btype="highpass", fs=SR, output="sos")
        male = signal.sosfiltfilt(low, mix)
        female = signal.sosfiltfilt(high, mix)
    elif method == "Chebyshev-I":
        low = signal.cheby1(6, 1.0, male_cutoff, btype="lowpass", fs=SR, output="sos")
        high = signal.cheby1(6, 1.0, female_cutoff, btype="highpass", fs=SR, output="sos")
        male = signal.sosfiltfilt(low, mix)
        female = signal.sosfiltfilt(high, mix)
    elif method == "Elliptic":
        low = signal.ellip(6, 1.0, 60.0, male_cutoff, btype="lowpass", fs=SR, output="sos")
        high = signal.ellip(6, 1.0, 60.0, female_cutoff, btype="highpass", fs=SR, output="sos")
        male = signal.sosfiltfilt(low, mix)
        female = signal.sosfiltfilt(high, mix)
    elif method == "FIR-Hamming":
        taps_low = signal.firwin(801, male_cutoff, fs=SR, pass_zero="lowpass", window="hamming")
        taps_high = signal.firwin(801, female_cutoff, fs=SR, pass_zero="highpass", window="hamming")
        male = signal.filtfilt(taps_low, [1.0], mix)
        female = signal.filtfilt(taps_high, [1.0], mix)
    else:
        raise ValueError(f"Unknown method: {method}")
    return peak_limit(male.astype(np.float32)), peak_limit(female.astype(np.float32))


def evaluate_pair(
    method: str,
    male_ref: np.ndarray,
    female_ref: np.ndarray,
    mix: np.ndarray,
    male_cutoff: float,
    female_cutoff: float,
    full_spectral: bool = False,
) -> dict[str, float | str]:
    male_est, female_est = separate_with_filter(method, mix, male_cutoff, female_cutoff)
    reconstructed = male_est + female_est
    rec_err = float(np.linalg.norm(mix - reconstructed) / (np.linalg.norm(mix) + EPS))

    male_si = si_sdr(male_ref, male_est)
    female_si = si_sdr(female_ref, female_est)
    male_corr = corrcoef(male_ref, male_est)
    female_corr = corrcoef(female_ref, female_est)
    avg_si = 0.5 * (male_si + female_si)
    min_si = min(male_si, female_si)
    avg_corr = 0.5 * (male_corr + female_corr)

    # Balanced score prefers both speakers to improve, instead of sacrificing
    # one speaker to maximize only the average.
    balanced_score = avg_si + 0.50 * min_si + 2.0 * avg_corr - 2.0 * rec_err

    return {
        "method": method,
        "male_cutoff_hz": float(male_cutoff),
        "female_cutoff_hz": float(female_cutoff),
        "male_si_sdr_db": male_si,
        "female_si_sdr_db": female_si,
        "avg_si_sdr_db": avg_si,
        "min_si_sdr_db": min_si,
        "male_snr_db": snr_db(male_ref, male_est),
        "female_snr_db": snr_db(female_ref, female_est),
        "male_corr": male_corr,
        "female_corr": female_corr,
        "avg_corr": avg_corr,
        "male_spectral_l1": spectral_error(male_ref, male_est) if full_spectral else float("nan"),
        "female_spectral_l1": spectral_error(female_ref, female_est) if full_spectral else float("nan"),
        "reconstruction_error": rec_err,
        "balanced_score": balanced_score,
    }


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_best_audio(best_rows: list[dict[str, float | str]], male_ref: np.ndarray, female_ref: np.ndarray, mix: np.ndarray) -> None:
    best_overall = max(best_rows, key=lambda row: float(row["balanced_score"]))
    for row in best_rows:
        method = str(row["method"])
        male, female = separate_with_filter(method, mix, float(row["male_cutoff_hz"]), float(row["female_cutoff_hz"]))
        prefix = PREFIX[method]
        write_wav(AUDIO_DIR / f"tuned_{prefix}_male.wav", male)
        write_wav(AUDIO_DIR / f"tuned_{prefix}_female.wav", female)

    male, female = separate_with_filter(
        str(best_overall["method"]),
        mix,
        float(best_overall["male_cutoff_hz"]),
        float(best_overall["female_cutoff_hz"]),
    )
    write_wav(AUDIO_DIR / "tuned_best_male.wav", male)
    write_wav(AUDIO_DIR / "tuned_best_female.wav", female)


def plot_heatmap(rows: list[dict[str, float | str]], methods: tuple[str, ...], male_values: np.ndarray, female_values: np.ndarray) -> None:
    n = len(methods)
    cols = 2 if n > 1 else 1
    rows_count = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(6.8 * cols, 4.5 * rows_count), constrained_layout=True)
    fig.suptitle("Traditional filter cutoff grid search: balanced score", fontsize=15, fontweight="bold")
    axes_array = np.atleast_1d(axes).ravel()

    for ax, method in zip(axes_array, methods):
        grid = np.full((len(female_values), len(male_values)), np.nan, dtype=np.float64)
        for row in rows:
            if row["method"] != method:
                continue
            mi = int(np.where(np.isclose(male_values, float(row["male_cutoff_hz"])))[0][0])
            fi = int(np.where(np.isclose(female_values, float(row["female_cutoff_hz"])))[0][0])
            grid[fi, mi] = float(row["balanced_score"])

        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[male_values[0], male_values[-1], female_values[0], female_values[-1]],
            cmap="YlGnBu",
        )
        ax.set_title(method, fontweight="bold")
        ax.set_xlabel("Male LPF cutoff / Hz")
        ax.set_ylabel("Female HPF cutoff / Hz")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes_array[len(methods):]:
        ax.axis("off")

    fig.savefig(IMAGE_DIR / "problem4_traditional_cutoff_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_best_bars(best_rows: list[dict[str, float | str]]) -> None:
    labels = [str(row["method"]) for row in best_rows]
    male_si = [float(row["male_si_sdr_db"]) for row in best_rows]
    female_si = [float(row["female_si_sdr_db"]) for row in best_rows]
    avg_corr = [float(row["avg_corr"]) for row in best_rows]
    score = [float(row["balanced_score"]) for row in best_rows]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), constrained_layout=True)
    axes[0].bar(x - 0.18, male_si, width=0.36, label="male", color="#6F8FA3")
    axes[0].bar(x + 0.18, female_si, width=0.36, label="female", color="#C9A7A2")
    axes[0].axhline(0.0, color="#8c857d", linewidth=0.8)
    axes[0].set_title("Best cutoff SI-SDR by method", fontweight="bold")
    axes[0].set_ylabel("SI-SDR / dB")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    ax2 = axes[1]
    ax2.plot(x, avg_corr, marker="o", linewidth=2.0, label="average correlation", color="#A8B8A0")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("Average correlation")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.grid(axis="y", alpha=0.25)
    ax3 = ax2.twinx()
    ax3.plot(x, score, marker="s", linewidth=2.0, label="balanced score", color="#C8A27A")
    ax3.set_ylabel("Balanced score")
    ax2.set_title("Best cutoff summary", fontweight="bold")

    lines, names = [], []
    for axis in (ax2, ax3):
        line, name = axis.get_legend_handles_labels()
        lines.extend(line)
        names.extend(name)
    ax2.legend(lines, names, loc="lower right")

    fig.savefig(IMAGE_DIR / "problem4_traditional_cutoff_best_bars.png", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search traditional speech-separation filter cutoffs.")
    parser.add_argument("--male-start", type=float, default=600.0)
    parser.add_argument("--male-stop", type=float, default=1500.0)
    parser.add_argument("--female-start", type=float, default=1100.0)
    parser.add_argument("--female-stop", type=float, default=2600.0)
    parser.add_argument("--step", type=float, default=100.0)
    parser.add_argument("--min-gap", type=float, default=150.0, help="Require female_cutoff - male_cutoff >= min_gap.")
    parser.add_argument("--include-fir", action="store_true", help="Also evaluate FIR-Hamming. This is much slower.")
    parser.add_argument("--full-spectral", action="store_true", help="Compute spectral L1 for every grid point. Slower.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    male_ref, female_ref, mix = prepare_reference_mix()
    male_values = cutoff_values(args.male_start, args.male_stop, args.step)
    female_values = cutoff_values(args.female_start, args.female_stop, args.step)
    methods = ALL_METHODS if args.include_fir else IIR_METHODS

    rows: list[dict[str, float | str]] = []
    total = 0
    for method in methods:
        for male_cutoff in male_values:
            for female_cutoff in female_values:
                if female_cutoff - male_cutoff < args.min_gap:
                    continue
                total += 1
                rows.append(
                    evaluate_pair(
                        method,
                        male_ref,
                        female_ref,
                        mix,
                        float(male_cutoff),
                        float(female_cutoff),
                        full_spectral=args.full_spectral,
                    )
                )

    rows.sort(key=lambda row: (str(row["method"]), -float(row["balanced_score"])))
    write_csv(IMAGE_DIR / "problem4_traditional_cutoff_grid_search.csv", rows)

    best_rows = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        best_rows.append(max(method_rows, key=lambda row: float(row["balanced_score"])))
    best_rows.sort(key=lambda row: -float(row["balanced_score"]))
    write_csv(IMAGE_DIR / "problem4_traditional_cutoff_best.csv", best_rows)
    save_best_audio(best_rows, male_ref, female_ref, mix)
    plot_heatmap(rows, methods, male_values, female_values)
    plot_best_bars(best_rows)

    print(f"Evaluated {total} traditional cutoff settings.")
    print("Best settings by balanced score:")
    for row in best_rows:
        print(
            f"{row['method']:13s} | male={row['male_cutoff_hz']:6.1f} Hz | "
            f"female={row['female_cutoff_hz']:6.1f} Hz | "
            f"male SI-SDR={row['male_si_sdr_db']:6.2f} dB | "
            f"female SI-SDR={row['female_si_sdr_db']:6.2f} dB | "
            f"avg corr={row['avg_corr']:.4f} | score={row['balanced_score']:.2f}"
        )
    print(f"\nSaved CSV: {IMAGE_DIR / 'problem4_traditional_cutoff_grid_search.csv'}")
    print(f"Saved best summary: {IMAGE_DIR / 'problem4_traditional_cutoff_best.csv'}")
    print(f"Saved figures: {IMAGE_DIR / 'problem4_traditional_cutoff_heatmap.png'}")
    print(f"Saved figures: {IMAGE_DIR / 'problem4_traditional_cutoff_best_bars.png'}")
    print("Saved tuned audio files under audio/tuned_*.wav")


if __name__ == "__main__":
    main()
