# -*- coding: utf-8 -*-
"""
Create intuitive visualizations for STFT, reference similarity, and mask allocation.

Outputs:
  images/problem4_stft_spectrogram_overview.png
  images/problem4_reference_guided_similarity_mask_flow.png
  images/problem4_mask_weight_allocation_slices.png
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
EPS = 1e-8


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
        boundary="zeros",
    )
    return freqs, times, spec


def mag_db(spec: np.ndarray) -> np.ndarray:
    db = 20.0 * np.log10(np.abs(spec) + EPS)
    return np.clip(db, np.max(db) - 80.0, np.max(db))


def show_spectrogram(ax: plt.Axes, times: np.ndarray, freqs: np.ndarray, spec: np.ndarray, title: str, cmap: str = "magma") -> None:
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


def spectral_template(spec: np.ndarray, smooth_bins: int = 41) -> np.ndarray:
    mag = np.log1p(np.abs(spec))
    active = np.mean(mag, axis=0) > max(1e-4, 0.08 * float(np.max(np.mean(mag, axis=0))))
    if np.any(active):
        env = np.mean(mag[:, active], axis=1)
    else:
        env = np.mean(mag, axis=1)
    kernel = np.ones(smooth_bins, dtype=np.float64) / smooth_bins
    env = np.convolve(env, kernel, mode="same")
    env = (env - np.mean(env)) / (np.std(env) + EPS)
    return env.astype(np.float64)


def reference_guided_components(
    mix: np.ndarray,
    male_ref: np.ndarray,
    female_ref: np.ndarray,
    male_cutoff: float = 900.0,
    female_cutoff: float = 1900.0,
    temperature: float = 2.6,
    prior_strength: float = 0.55,
) -> dict[str, np.ndarray]:
    freqs, times, mix_spec = stft_np(mix)
    _, _, male_spec = stft_np(male_ref)
    _, _, female_spec = stft_np(female_ref)

    male_template = spectral_template(male_spec)
    female_template = spectral_template(female_spec)
    mix_log = np.log1p(np.abs(mix_spec))
    mix_norm = (mix_log - np.mean(mix_log, axis=0, keepdims=True)) / (np.std(mix_log, axis=0, keepdims=True) + EPS)

    # Similarity score at each time-frequency bin. Positive gap means the bin
    # better matches the male reference template; negative means female.
    male_score = mix_norm * male_template[:, None]
    female_score = mix_norm * female_template[:, None]

    f = np.maximum(freqs, 1.0)
    order = 4.0
    male_prior = 1.0 / np.sqrt(1.0 + (f / male_cutoff) ** (2.0 * order))
    female_prior = 1.0 / np.sqrt(1.0 + (female_cutoff / f) ** (2.0 * order))
    male_prior_logit = np.log(np.clip(male_prior, 1e-4, 1 - 1e-4) / np.clip(1.0 - male_prior, 1e-4, 1.0))
    female_prior_logit = np.log(np.clip(female_prior, 1e-4, 1 - 1e-4) / np.clip(1.0 - female_prior, 1e-4, 1.0))

    male_logit = temperature * male_score + prior_strength * male_prior_logit[:, None]
    female_logit = temperature * female_score + prior_strength * female_prior_logit[:, None]
    logits = np.stack([male_logit, female_logit], axis=0)
    logits = logits - np.max(logits, axis=0, keepdims=True)
    masks = np.exp(logits)
    masks = masks / (np.sum(masks, axis=0, keepdims=True) + EPS)

    male_est_spec = mix_spec * masks[0]
    female_est_spec = mix_spec * masks[1]
    return {
        "freqs": freqs,
        "times": times,
        "mix_spec": mix_spec,
        "male_template": male_template,
        "female_template": female_template,
        "male_score": male_score,
        "female_score": female_score,
        "score_gap": male_score - female_score,
        "male_prior": male_prior,
        "female_prior": female_prior,
        "male_mask": masks[0],
        "female_mask": masks[1],
        "male_est_spec": male_est_spec,
        "female_est_spec": female_est_spec,
    }


def save_stft_overview(male: np.ndarray, female: np.ndarray, mix: np.ndarray) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14.0, 10.0), constrained_layout=True)
    fig.suptitle("STFT turns waveforms into time-frequency maps", fontsize=16, fontweight="bold")

    for row, (name, x, color) in enumerate(
        [
            ("Male reference", male, "#2563eb"),
            ("Female reference", female, "#db2777"),
            ("Mixture = male + female", mix, "#111827"),
        ]
    ):
        t = np.arange(len(x)) / SR
        step = max(1, len(x) // 25000)
        axes[row, 0].plot(t[::step], x[::step], color=color, linewidth=0.55)
        axes[row, 0].set_title(f"{name} waveform", fontweight="bold")
        axes[row, 0].set_xlabel("Time / s")
        axes[row, 0].set_ylabel("Amplitude")
        axes[row, 0].grid(True, alpha=0.22)

        freqs, times, spec = stft_np(x)
        im = show_spectrogram(axes[row, 1], times, freqs, spec, f"{name} STFT magnitude")
        fig.colorbar(im, ax=axes[row, 1], fraction=0.035, pad=0.02, label="Magnitude / dB")

    fig.savefig(IMAGES / "problem4_stft_spectrogram_overview.png", bbox_inches="tight")
    plt.close(fig)


def save_reference_guided_flow(comp: dict[str, np.ndarray]) -> None:
    freqs = comp["freqs"]
    times = comp["times"]
    valid_f = freqs <= 5000

    fig = plt.figure(figsize=(15.0, 11.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    fig.suptitle("Reference-guided separation: template similarity + DSP prior -> soft masks", fontsize=16, fontweight="bold")

    ax0 = fig.add_subplot(gs[0, 0])
    im0 = show_spectrogram(ax0, times, freqs, comp["mix_spec"], "1) Mix STFT |X(t,f)|")
    fig.colorbar(im0, ax=ax0, fraction=0.044, pad=0.02)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(comp["male_template"][valid_f], freqs[valid_f], color="#2563eb", linewidth=2.0, label="male template")
    ax1.plot(comp["female_template"][valid_f], freqs[valid_f], color="#db2777", linewidth=2.0, label="female template")
    ax1.set_ylim(0, 5000)
    ax1.invert_yaxis()
    ax1.set_xlabel("Normalized spectral envelope")
    ax1.set_ylabel("Frequency / Hz")
    ax1.set_title("2) Reference templates", fontweight="bold")
    ax1.grid(True, alpha=0.2)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    ax2.text(
        0.02,
        0.82,
        "Similarity score",
        fontsize=15,
        fontweight="bold",
        transform=ax2.transAxes,
    )
    ax2.text(
        0.02,
        0.62,
        "score_k(t,f) = norm(|X(t,f)|) · template_k(f)",
        fontsize=12,
        transform=ax2.transAxes,
    )
    ax2.text(
        0.02,
        0.40,
        "logit_k(t,f) = temperature · score_k(t,f)\n"
        "              + prior_strength · prior_k(f)",
        fontsize=12,
        transform=ax2.transAxes,
    )
    ax2.text(
        0.02,
        0.18,
        "mask_k(t,f) = softmax(logit_k)",
        fontsize=12,
        transform=ax2.transAxes,
    )

    ax3 = fig.add_subplot(gs[1, 0])
    gap = np.clip(comp["score_gap"], -3.5, 3.5)
    im3 = ax3.imshow(
        gap,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap="coolwarm",
        vmin=-3.0,
        vmax=3.0,
    )
    ax3.set_ylim(0, 5000)
    ax3.set_xlabel("Time / s")
    ax3.set_ylabel("Frequency / Hz")
    ax3.set_title("3) Template similarity gap: male score - female score", fontweight="bold")
    fig.colorbar(im3, ax=ax3, fraction=0.044, pad=0.02)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(freqs[valid_f], comp["male_prior"][valid_f], color="#2563eb", linewidth=2.0, label="male low-frequency prior")
    ax4.plot(freqs[valid_f], comp["female_prior"][valid_f], color="#db2777", linewidth=2.0, label="female high-frequency prior")
    ax4.set_xlim(0, 5000)
    ax4.set_ylim(0, 1.05)
    ax4.set_xlabel("Frequency / Hz")
    ax4.set_ylabel("Prior weight")
    ax4.set_title("4) DSP frequency priors", fontweight="bold")
    ax4.grid(True, alpha=0.22)
    ax4.legend()

    ax5 = fig.add_subplot(gs[1, 2])
    frame_energy = np.mean(np.abs(comp["mix_spec"]), axis=0)
    frame = int(np.argmax(frame_energy))
    ax5.plot(freqs[valid_f], comp["male_mask"][valid_f, frame], color="#2563eb", linewidth=2.0, label="male mask")
    ax5.plot(freqs[valid_f], comp["female_mask"][valid_f, frame], color="#db2777", linewidth=2.0, label="female mask")
    ax5.fill_between(freqs[valid_f], 0, comp["male_mask"][valid_f, frame], color="#2563eb", alpha=0.15)
    ax5.fill_between(freqs[valid_f], comp["male_mask"][valid_f, frame], 1, color="#db2777", alpha=0.12)
    ax5.set_xlim(0, 5000)
    ax5.set_ylim(0, 1.05)
    ax5.set_xlabel("Frequency / Hz")
    ax5.set_ylabel("Mask value")
    ax5.set_title(f"5) Softmax allocation at t={times[frame]:.2f}s", fontweight="bold")
    ax5.grid(True, alpha=0.22)
    ax5.legend()

    ax6 = fig.add_subplot(gs[2, 0])
    im6 = ax6.imshow(
        comp["male_mask"],
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    ax6.set_ylim(0, 5000)
    ax6.set_xlabel("Time / s")
    ax6.set_ylabel("Frequency / Hz")
    ax6.set_title("6) Male mask M_m(t,f)", fontweight="bold")
    fig.colorbar(im6, ax=ax6, fraction=0.044, pad=0.02)

    ax7 = fig.add_subplot(gs[2, 1])
    im7 = ax7.imshow(
        comp["female_mask"],
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap="Reds",
        vmin=0,
        vmax=1,
    )
    ax7.set_ylim(0, 5000)
    ax7.set_xlabel("Time / s")
    ax7.set_ylabel("Frequency / Hz")
    ax7.set_title("7) Female mask M_f(t,f)", fontweight="bold")
    fig.colorbar(im7, ax=ax7, fraction=0.044, pad=0.02)

    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis("off")
    ax8.text(0.02, 0.80, "Mask split", fontsize=15, fontweight="bold", transform=ax8.transAxes)
    ax8.text(
        0.02,
        0.58,
        "At every time-frequency bin:\n"
        "M_m(t,f) + M_f(t,f) = 1",
        fontsize=12,
        transform=ax8.transAxes,
    )
    ax8.text(
        0.02,
        0.34,
        "Estimated spectra:\n"
        "Ŝ_m(t,f) = M_m(t,f) · X(t,f)\n"
        "Ŝ_f(t,f) = M_f(t,f) · X(t,f)",
        fontsize=12,
        transform=ax8.transAxes,
    )
    ax8.text(
        0.02,
        0.12,
        "Then iSTFT reconstructs waveforms.",
        fontsize=12,
        transform=ax8.transAxes,
    )

    fig.savefig(IMAGES / "problem4_reference_guided_similarity_mask_flow.png", bbox_inches="tight")
    plt.close(fig)


def save_weight_allocation_slices(comp: dict[str, np.ndarray]) -> None:
    freqs = comp["freqs"]
    times = comp["times"]
    valid_f = freqs <= 5000
    mix_energy = np.mean(np.abs(comp["mix_spec"]), axis=0)
    candidates = np.argsort(mix_energy)[-4:]
    candidates = sorted(int(i) for i in candidates)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)
    fig.suptitle("Soft mask weight allocation at selected speech frames", fontsize=16, fontweight="bold")
    for ax, frame in zip(axes.ravel(), candidates):
        ax.stackplot(
            freqs[valid_f],
            comp["male_mask"][valid_f, frame],
            comp["female_mask"][valid_f, frame],
            labels=["male weight", "female weight"],
            colors=["#93c5fd", "#f9a8d4"],
            alpha=0.85,
        )
        ax.plot(freqs[valid_f], comp["male_mask"][valid_f, frame], color="#2563eb", linewidth=1.6)
        ax.plot(freqs[valid_f], comp["female_mask"][valid_f, frame], color="#db2777", linewidth=1.6)
        ax.set_xlim(0, 5000)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Frequency / Hz")
        ax.set_ylabel("Allocated weight")
        ax.set_title(f"Frame at t={times[frame]:.2f}s", fontweight="bold")
        ax.grid(True, alpha=0.22)
        ax.legend(loc="upper right", fontsize=8)

    fig.savefig(IMAGES / "problem4_mask_weight_allocation_slices.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    male = read_mono("male_speech_ref.wav")
    female = read_mono("female_speech_ref.wav")
    mix = read_mono("mix_speech.wav")
    save_stft_overview(male, female, mix)
    comp = reference_guided_components(mix, male, female)
    save_reference_guided_flow(comp)
    save_weight_allocation_slices(comp)
    print(IMAGES / "problem4_stft_spectrogram_overview.png")
    print(IMAGES / "problem4_reference_guided_similarity_mask_flow.png")
    print(IMAGES / "problem4_mask_weight_allocation_slices.png")


if __name__ == "__main__":
    main()
