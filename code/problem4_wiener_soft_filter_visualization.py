# -*- coding: utf-8 -*-
"""
Visualize the reference-spectrum Wiener soft filter.

Outputs:
  images/problem4_wiener_soft_filter_design_flow.png
  images/problem4_wiener_soft_filter_mask_application.png
  images/problem4_wiener_soft_filter_weight_allocation.png
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
EPS = 1e-12

ALPHA = 0.8
SMOOTH_BINS = 21
MASK_SMOOTH_BINS = 17


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 320


def read_mono(name: str) -> np.ndarray:
    x, sr = sf.read(AUDIO / name, always_2d=True)
    if sr != SR:
        raise ValueError(f"{name} has sample rate {sr}, expected {SR}")
    return x.mean(axis=1).astype(np.float64)


def smooth_1d(x: np.ndarray, bins: int) -> np.ndarray:
    bins = max(3, int(bins))
    if bins % 2 == 0:
        bins += 1
    kernel = np.ones(bins, dtype=np.float64) / bins
    return np.convolve(x, kernel, mode="same")


def reference_psd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freqs, psd = signal.welch(x, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
    log_psd = np.log(psd + EPS)
    smoothed = np.exp(smooth_1d(log_psd, SMOOTH_BINS))
    return freqs, smoothed.astype(np.float64)


def stft_np(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return signal.stft(x, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="zeros")


def mag_db(spec: np.ndarray) -> np.ndarray:
    db = 20.0 * np.log10(np.abs(spec) + EPS)
    return np.clip(db, np.max(db) - 80.0, np.max(db))


def wiener_components(male: np.ndarray, female: np.ndarray, mix: np.ndarray) -> dict[str, np.ndarray]:
    psd_freqs, male_psd = reference_psd(male)
    _, female_psd = reference_psd(female)

    freqs, times, mix_stft = stft_np(mix)
    male_interp = np.interp(freqs, psd_freqs, male_psd)
    female_interp = np.interp(freqs, psd_freqs, female_psd)

    male_power = np.power(male_interp + EPS, ALPHA)
    female_power = np.power(female_interp + EPS, ALPHA)
    male_mask = male_power / (male_power + female_power + EPS)
    male_mask = smooth_1d(male_mask, MASK_SMOOTH_BINS)
    male_mask = np.clip(male_mask, 0.02, 0.98)
    female_mask = 1.0 - male_mask

    male_est_stft = mix_stft * male_mask[:, None]
    female_est_stft = mix_stft * female_mask[:, None]
    return {
        "psd_freqs": psd_freqs,
        "male_psd": male_psd,
        "female_psd": female_psd,
        "freqs": freqs,
        "times": times,
        "mix_stft": mix_stft,
        "male_power": male_power,
        "female_power": female_power,
        "male_mask": male_mask,
        "female_mask": female_mask,
        "male_est_stft": male_est_stft,
        "female_est_stft": female_est_stft,
    }


def plot_spectrogram(ax: plt.Axes, times: np.ndarray, freqs: np.ndarray, spec: np.ndarray, title: str, cmap: str = "magma"):
    db = mag_db(spec)
    im = ax.imshow(
        db,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap=cmap,
        vmin=np.max(db) - 75.0,
        vmax=np.max(db),
    )
    ax.set_ylim(0, 5000)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Frequency / Hz")
    ax.set_title(title, fontweight="bold")
    return im


def save_design_flow(comp: dict[str, np.ndarray]) -> None:
    psd_freqs = comp["psd_freqs"]
    freqs = comp["freqs"]
    valid_psd = psd_freqs <= 5000
    valid = freqs <= 5000

    fig = plt.figure(figsize=(15.0, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    fig.suptitle("Reference-spectrum Wiener soft filter: from reference PSD to H(f)", fontsize=16, fontweight="bold")

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(psd_freqs[valid_psd], 10 * np.log10(comp["male_psd"][valid_psd] + EPS), color="#2563eb", linewidth=2.0, label="male PSD")
    ax0.plot(psd_freqs[valid_psd], 10 * np.log10(comp["female_psd"][valid_psd] + EPS), color="#db2777", linewidth=2.0, label="female PSD")
    ax0.set_title("1) Estimate reference power spectra", fontweight="bold")
    ax0.set_xlabel("Frequency / Hz")
    ax0.set_ylabel("Power / dB")
    ax0.set_xlim(0, 5000)
    ax0.grid(True, alpha=0.24)
    ax0.legend()

    ax1 = fig.add_subplot(gs[0, 1])
    norm_region = valid & (freqs >= 80)
    male_scale = np.max(comp["male_power"][norm_region]) + EPS
    female_scale = np.max(comp["female_power"][norm_region]) + EPS
    male_norm = np.clip(comp["male_power"][valid] / male_scale, 0.0, 1.05)
    female_norm = np.clip(comp["female_power"][valid] / female_scale, 0.0, 1.05)
    ax1.plot(freqs[valid], male_norm, color="#2563eb", linewidth=2.0, label=r"$P_m(f)^\alpha$")
    ax1.plot(freqs[valid], female_norm, color="#db2777", linewidth=2.0, label=r"$P_f(f)^\alpha$")
    ax1.set_title(r"2) Apply soft sharpness $\alpha=0.8$", fontweight="bold")
    ax1.set_xlabel("Frequency / Hz")
    ax1.set_ylabel("Normalized powered PSD")
    ax1.set_xlim(0, 5000)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.24)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    ax2.text(0.02, 0.82, "Wiener-style soft split", fontsize=15, fontweight="bold", transform=ax2.transAxes)
    ax2.text(
        0.02,
        0.58,
        r"$H_m(f)=\frac{P_m(f)^\alpha}{P_m(f)^\alpha+P_f(f)^\alpha}$",
        fontsize=16,
        transform=ax2.transAxes,
    )
    ax2.text(0.02, 0.38, r"$H_f(f)=1-H_m(f)$", fontsize=16, transform=ax2.transAxes)
    ax2.text(
        0.02,
        0.16,
        "No training. No time-varying network.\n"
        "It is a fixed frequency response H(f).",
        fontsize=12,
        transform=ax2.transAxes,
    )

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.stackplot(
        freqs[valid],
        comp["male_mask"][valid],
        comp["female_mask"][valid],
        labels=["male weight", "female weight"],
        colors=["#93c5fd", "#f9a8d4"],
        alpha=0.88,
    )
    ax3.plot(freqs[valid], comp["male_mask"][valid], color="#2563eb", linewidth=2.0)
    ax3.plot(freqs[valid], comp["female_mask"][valid], color="#db2777", linewidth=2.0)
    ax3.set_title(r"3) Complementary weights: $H_m(f)+H_f(f)=1$", fontweight="bold")
    ax3.set_xlabel("Frequency / Hz")
    ax3.set_ylabel("Allocated weight")
    ax3.set_xlim(0, 5000)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.24)
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    mix_db = 10 * np.log10(signal.welch(read_mono("mix_speech.wav"), fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP)[1] + EPS)
    ax4.plot(psd_freqs[valid_psd], mix_db[valid_psd], color="#111827", linewidth=1.6, label="mixture PSD")
    ax4.plot(freqs[valid], mix_db[: len(freqs)][valid] + 20 * np.log10(comp["male_mask"][valid] + EPS), color="#2563eb", linewidth=1.6, label="mixture * male weight")
    ax4.plot(freqs[valid], mix_db[: len(freqs)][valid] + 20 * np.log10(comp["female_mask"][valid] + EPS), color="#db2777", linewidth=1.6, label="mixture * female weight")
    ax4.set_title("4) The same mixed spectrum is softly split", fontweight="bold")
    ax4.set_xlabel("Frequency / Hz")
    ax4.set_ylabel("Approx. PSD / dB")
    ax4.set_xlim(0, 5000)
    ax4.set_ylim(-105, -25)
    ax4.grid(True, alpha=0.24)
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    ax5.text(0.02, 0.78, "Apply to STFT", fontsize=15, fontweight="bold", transform=ax5.transAxes)
    ax5.text(
        0.02,
        0.54,
        r"$\hat{S}_m(t,f)=H_m(f)\cdot X(t,f)$" + "\n" + r"$\hat{S}_f(t,f)=H_f(f)\cdot X(t,f)$",
        fontsize=15,
        transform=ax5.transAxes,
    )
    ax5.text(
        0.02,
        0.24,
        "Because H_m + H_f = 1:\n"
        "male estimate + female estimate\n"
        "reconstructs the mixture almost exactly.",
        fontsize=12,
        transform=ax5.transAxes,
    )

    fig.savefig(IMAGES / "problem4_wiener_soft_filter_design_flow.png", bbox_inches="tight")
    plt.close(fig)


def save_mask_application(comp: dict[str, np.ndarray]) -> None:
    freqs = comp["freqs"]
    times = comp["times"]
    male_mask_img = np.repeat(comp["male_mask"][:, None], len(times), axis=1)
    female_mask_img = np.repeat(comp["female_mask"][:, None], len(times), axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.8), constrained_layout=True)
    fig.suptitle("Wiener soft filter application in the STFT domain", fontsize=16, fontweight="bold")

    im0 = plot_spectrogram(axes[0, 0], times, freqs, comp["mix_stft"], "1) Mixture STFT |X(t,f)|")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.045, pad=0.02)

    im1 = axes[0, 1].imshow(
        male_mask_img,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    axes[0, 1].set_ylim(0, 5000)
    axes[0, 1].set_xlabel("Time / s")
    axes[0, 1].set_ylabel("Frequency / Hz")
    axes[0, 1].set_title("2) Male response H_m(f), repeated over time", fontweight="bold")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.045, pad=0.02)

    im2 = axes[0, 2].imshow(
        female_mask_img,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap="Reds",
        vmin=0,
        vmax=1,
    )
    axes[0, 2].set_ylim(0, 5000)
    axes[0, 2].set_xlabel("Time / s")
    axes[0, 2].set_ylabel("Frequency / Hz")
    axes[0, 2].set_title("3) Female response H_f(f), repeated over time", fontweight="bold")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.045, pad=0.02)

    im3 = plot_spectrogram(axes[1, 0], times, freqs, comp["male_est_stft"], r"4) $\hat{S}_m(t,f)=H_m(f)X(t,f)$", cmap="Blues")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.045, pad=0.02)

    im4 = plot_spectrogram(axes[1, 1], times, freqs, comp["female_est_stft"], r"5) $\hat{S}_f(t,f)=H_f(f)X(t,f)$", cmap="Reds")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.045, pad=0.02)

    recon_error = np.abs(comp["mix_stft"] - (comp["male_est_stft"] + comp["female_est_stft"]))
    im5 = axes[1, 2].imshow(
        20 * np.log10(recon_error + EPS),
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap="Greys",
        vmin=-160,
        vmax=-60,
    )
    axes[1, 2].set_ylim(0, 5000)
    axes[1, 2].set_xlabel("Time / s")
    axes[1, 2].set_ylabel("Frequency / Hz")
    axes[1, 2].set_title(r"6) Reconstruction residual: $X-(\hat{S}_m+\hat{S}_f)$", fontweight="bold")
    fig.colorbar(im5, ax=axes[1, 2], fraction=0.045, pad=0.02)

    fig.savefig(IMAGES / "problem4_wiener_soft_filter_mask_application.png", bbox_inches="tight")
    plt.close(fig)


def save_weight_allocation(comp: dict[str, np.ndarray]) -> None:
    freqs = comp["freqs"]
    valid = freqs <= 5000
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    ax.stackplot(
        freqs[valid],
        comp["male_mask"][valid],
        comp["female_mask"][valid],
        labels=["male allocation H_m(f)", "female allocation H_f(f)"],
        colors=["#93c5fd", "#f9a8d4"],
        alpha=0.9,
    )
    ax.plot(freqs[valid], comp["male_mask"][valid], color="#2563eb", linewidth=2.0)
    ax.plot(freqs[valid], comp["female_mask"][valid], color="#db2777", linewidth=2.0)
    ax.set_xlim(0, 5000)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Frequency / Hz")
    ax.set_ylabel("Weight")
    ax.set_title("Wiener soft filter weight allocation by frequency", fontweight="bold")
    ax.grid(True, alpha=0.24)
    ax.legend(loc="center right")
    ax.text(
        0.03,
        0.08,
        "The split is soft: each frequency is partly assigned to male and partly to female.\n"
        "The response is fixed over time, so this is filtering H(f), not a time-varying neural mask M(t,f).",
        transform=ax.transAxes,
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )
    fig.savefig(IMAGES / "problem4_wiener_soft_filter_weight_allocation.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    male = read_mono("male_speech_ref.wav")
    female = read_mono("female_speech_ref.wav")
    mix = read_mono("mix_speech.wav")
    comp = wiener_components(male, female, mix)
    save_design_flow(comp)
    save_mask_application(comp)
    save_weight_allocation(comp)
    print(IMAGES / "problem4_wiener_soft_filter_design_flow.png")
    print(IMAGES / "problem4_wiener_soft_filter_mask_application.png")
    print(IMAGES / "problem4_wiener_soft_filter_weight_allocation.png")


if __name__ == "__main__":
    main()
