# -*- coding: utf-8 -*-
"""
Problem 4 controlled speech-separation comparison.

Inputs:
  audio/male_speech.wav
  audio/female_speech.wav

Outputs:
  audio/male_speech_ref.wav
  audio/female_speech_ref.wav
  audio/mix_speech.wav
  audio/butter_male.wav
  audio/butter_female.wav
  audio/diff_male.wav
  audio/diff_female.wav
  audio/diff_reconstructed_mix.wav
  images/problem4_speech_method_comparison.png
  images/problem4_speech_filter_response_comparison.png
  images/problem4_speech_training_loss.png
  images/problem4_speech_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal


ROOT_DIR = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT_DIR / "audio"
IMAGE_DIR = ROOT_DIR / "images"

SR = 16_000
N_FFT = 1024
HOP = 256
WIN = 1024
EPS = 1e-8


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 320


# ============================================================
# 1. Basic audio I/O and controlled mixture construction
#    读取男女原始音频，统一采样率/单声道/响度，并构造有真值的混合语音。
# ============================================================

def read_mono(path: Path, sr: int = SR) -> np.ndarray:
    data, source_sr = sf.read(path, always_2d=True)
    mono = data.astype(np.float32).mean(axis=1)
    if source_sr != sr:
        gcd = math.gcd(source_sr, sr)
        mono = signal.resample_poly(mono, sr // gcd, source_sr // gcd).astype(np.float32)
    return mono


def rms_normalize(x: np.ndarray, target_rms: float = 0.12) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(x)) + EPS))
    if rms < 1e-8:
        return x.astype(np.float32)
    return (x / rms * target_rms).astype(np.float32)


def peak_limit(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = float(np.max(np.abs(x)) + EPS)
    if max_abs > peak:
        x = x / max_abs * peak
    return x.astype(np.float32)


def write_wav(path: Path, x: np.ndarray, sr: int = SR) -> None:
    sf.write(path, peak_limit(x), sr)


def prepare_reference_mix() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the controlled experiment input and ground-truth references.

    male_speech.wav and female_speech.wav are normalized separately.
    The shorter signal is zero-padded to the longer one, so the references
    have equal length and can be added sample-by-sample.
    """
    male_path = AUDIO_DIR / "male_speech.wav"
    female_path = AUDIO_DIR / "female_speech.wav"
    if not male_path.exists() or not female_path.exists():
        raise FileNotFoundError("audio/male_speech.wav and audio/female_speech.wav are required.")

    male = rms_normalize(read_mono(male_path))
    female = rms_normalize(read_mono(female_path))
    n = max(len(male), len(female))
    male = np.pad(male, (0, n - len(male))).astype(np.float32)
    female = np.pad(female, (0, n - len(female))).astype(np.float32)
    mix = peak_limit(male + female, peak=0.95)

    write_wav(AUDIO_DIR / "male_speech_ref.wav", male)
    write_wav(AUDIO_DIR / "female_speech_ref.wav", female)
    write_wav(AUDIO_DIR / "mix_speech.wav", mix)
    return male, female, mix


# ============================================================
# 2. Traditional fixed-filter baselines
#    用固定参数的 IIR/FIR 滤波器做男女声分离，对应课程题目里的“设计滤波器”。
# ============================================================

def butterworth_separate(
    mix: np.ndarray,
    male_cutoff: float = 1000.0,
    female_cutoff: float = 1700.0,
    order: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sos_low = signal.butter(order, male_cutoff, btype="lowpass", fs=SR, output="sos")
    sos_high = signal.butter(order, female_cutoff, btype="highpass", fs=SR, output="sos")
    male = signal.sosfiltfilt(sos_low, mix).astype(np.float32)
    female = signal.sosfiltfilt(sos_high, mix).astype(np.float32)
    male = peak_limit(male)
    female = peak_limit(female)
    write_wav(AUDIO_DIR / "butter_male.wav", male)
    write_wav(AUDIO_DIR / "butter_female.wav", female)
    return male, female, sos_low, sos_high


def fixed_filter_separations(
    mix: np.ndarray,
    male_cutoff: float = 1000.0,
    female_cutoff: float = 1700.0,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    """Run several fixed traditional filter pairs for comparison."""
    outputs: list[dict[str, object]] = []

    def add_sos_method(method: str, prefix: str, sos_low: np.ndarray, sos_high: np.ndarray) -> None:
        # SOS form is numerically stable for IIR filters such as Butterworth/Chebyshev/Elliptic.
        male = peak_limit(signal.sosfiltfilt(sos_low, mix).astype(np.float32))
        female = peak_limit(signal.sosfiltfilt(sos_high, mix).astype(np.float32))
        write_wav(AUDIO_DIR / f"{prefix}_male.wav", male)
        write_wav(AUDIO_DIR / f"{prefix}_female.wav", female)
        outputs.append(
            {
                "method": method,
                "prefix": prefix,
                "male": male,
                "female": female,
                "low_kind": "sos",
                "high_kind": "sos",
                "low_filter": sos_low,
                "high_filter": sos_high,
            }
        )

    def add_fir_method(method: str, prefix: str, taps_low: np.ndarray, taps_high: np.ndarray) -> None:
        # FIR-Hamming is a linear-phase finite impulse response baseline.
        male = peak_limit(signal.filtfilt(taps_low, [1.0], mix).astype(np.float32))
        female = peak_limit(signal.filtfilt(taps_high, [1.0], mix).astype(np.float32))
        write_wav(AUDIO_DIR / f"{prefix}_male.wav", male)
        write_wav(AUDIO_DIR / f"{prefix}_female.wav", female)
        outputs.append(
            {
                "method": method,
                "prefix": prefix,
                "male": male,
                "female": female,
                "low_kind": "ba",
                "high_kind": "ba",
                "low_filter": (taps_low, np.array([1.0])),
                "high_filter": (taps_high, np.array([1.0])),
            }
        )

    sos_low = signal.butter(6, male_cutoff, btype="lowpass", fs=SR, output="sos")
    sos_high = signal.butter(6, female_cutoff, btype="highpass", fs=SR, output="sos")
    add_sos_method("Butterworth", "butter", sos_low, sos_high)

    sos_low = signal.cheby1(6, 1.0, male_cutoff, btype="lowpass", fs=SR, output="sos")
    sos_high = signal.cheby1(6, 1.0, female_cutoff, btype="highpass", fs=SR, output="sos")
    add_sos_method("Chebyshev-I", "cheby1", sos_low, sos_high)

    sos_low = signal.ellip(6, 1.0, 60.0, male_cutoff, btype="lowpass", fs=SR, output="sos")
    sos_high = signal.ellip(6, 1.0, 60.0, female_cutoff, btype="highpass", fs=SR, output="sos")
    add_sos_method("Elliptic", "ellip", sos_low, sos_high)

    taps_low = signal.firwin(801, male_cutoff, fs=SR, pass_zero="lowpass", window="hamming")
    taps_high = signal.firwin(801, female_cutoff, fs=SR, pass_zero="highpass", window="hamming")
    add_fir_method("FIR-Hamming", "fir", taps_low, taps_high)

    return outputs, outputs[0]["low_filter"], outputs[0]["high_filter"]


# ============================================================
# 3. STFT/iSTFT helpers shared by the neural methods
#    深度学习方法都在时频域预测 mask，再用 iSTFT 回到波形。
# ============================================================

def stft_torch(x: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    return torch.stft(
        x,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        window=window.to(x.device),
        center=True,
        return_complex=True,
    )


# ============================================================
# 4. Non-trainable VoiceFilter-style reference-guided separator
#    不训练网络，只用参考说话人的频谱模板 + 频率先验生成时频 mask。
# ============================================================

def reference_guided_filter_separate(
    mix: np.ndarray,
    male_ref: np.ndarray,
    female_ref: np.ndarray,
    male_cutoff: float = 900.0,
    female_cutoff: float = 1900.0,
    template_smooth_bins: int = 41,
    temperature: float = 2.6,
    prior_strength: float = 0.55,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """VoiceFilter-style extraction using reference spectral templates and DSP priors.

    The target references are converted to smooth spectral-envelope templates.
    The mixture is separated by a time-frequency soft mask whose logits combine:
    1) similarity to the male/female reference templates;
    2) a differentiable Butterworth-style frequency prior.
    """
    freqs, _, mix_stft = signal.stft(mix, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="zeros")
    _, _, male_stft = signal.stft(male_ref, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="zeros")
    _, _, female_stft = signal.stft(female_ref, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="zeros")

    def spectral_template(spec: np.ndarray) -> np.ndarray:
        # Average active speech frames to estimate a stable spectral envelope.
        mag = np.log1p(np.abs(spec))
        active = np.mean(mag, axis=0) > max(1e-4, 0.08 * float(np.max(np.mean(mag, axis=0))))
        if np.any(active):
            env = np.mean(mag[:, active], axis=1)
        else:
            env = np.mean(mag, axis=1)
        kernel = np.ones(template_smooth_bins, dtype=np.float64) / template_smooth_bins
        env = np.convolve(env, kernel, mode="same")
        env = (env - np.mean(env)) / (np.std(env) + EPS)
        return env.astype(np.float32)

    male_template = spectral_template(male_stft)
    female_template = spectral_template(female_stft)
    mix_log = np.log1p(np.abs(mix_stft))
    mix_norm = (mix_log - np.mean(mix_log, axis=0, keepdims=True)) / (np.std(mix_log, axis=0, keepdims=True) + EPS)

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
    # Softmax forces male/female masks to compete at each time-frequency bin.
    logits = np.stack([male_logit, female_logit], axis=0)
    logits = logits - np.max(logits, axis=0, keepdims=True)
    masks = np.exp(logits)
    masks = masks / (np.sum(masks, axis=0, keepdims=True) + EPS)

    _, male = signal.istft(mix_stft * masks[0], fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP, input_onesided=True)
    _, female = signal.istft(mix_stft * masks[1], fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP, input_onesided=True)
    male = peak_limit(male[: len(mix)].astype(np.float32))
    female = peak_limit(female[: len(mix)].astype(np.float32))
    write_wav(AUDIO_DIR / "ref_male.wav", male)
    write_wav(AUDIO_DIR / "ref_female.wav", female)
    write_wav(AUDIO_DIR / "ref_reconstructed_mix.wav", male + female)

    debug = {
        "freqs": freqs,
        "male_template": male_template,
        "female_template": female_template,
        "male_prior": male_prior.astype(np.float32),
        "female_prior": female_prior.astype(np.float32),
    }
    return male, female, debug


def istft_torch(spec: torch.Tensor, window: torch.Tensor, length: int) -> torch.Tensor:
    return torch.istft(
        spec,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        window=window.to(spec.device),
        center=True,
        length=length,
    )


class ConvBlock(nn.Module):
    """Residual convolution block used by the lightweight mask networks."""
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(4, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))


@dataclass
class DiffConfig:
    channels: int = 24
    prior_weight: float = 1.2
    init_male_fc: float = 1000.0
    init_female_fc: float = 1700.0
    male_fc_min: float = 500.0
    male_fc_max: float = 1500.0
    female_fc_min: float = 1100.0
    female_fc_max: float = 3000.0
    butter_order: float = 4.0


# ============================================================
# 5. DifferentiableFilter: supervised two-output mask network
#    输入混合语音，一次输出男/女两路；内部有可学习低通/高通频率先验。
# ============================================================

class DifferentiableFilterSeparator(nn.Module):
    def __init__(self, cfg: DiffConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DiffConfig()
        c = self.cfg.channels
        self.net = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1),
            nn.GroupNorm(4, c),
            nn.SiLU(),
            ConvBlock(c, 1),
            ConvBlock(c, 2),
            ConvBlock(c, 4),
            ConvBlock(c, 8),
            nn.Conv2d(c, 2, 1),
        )
        self.raw_male_fc = nn.Parameter(self._raw_from_init(self.cfg.init_male_fc, self.cfg.male_fc_min, self.cfg.male_fc_max))
        self.raw_female_fc = nn.Parameter(self._raw_from_init(self.cfg.init_female_fc, self.cfg.female_fc_min, self.cfg.female_fc_max))
        self.register_buffer("window", torch.hann_window(WIN), persistent=False)

    @staticmethod
    def _raw_from_init(value: float, lo: float, hi: float) -> torch.Tensor:
        p = min(0.99, max(0.01, (value - lo) / (hi - lo)))
        return torch.tensor(math.log(p / (1.0 - p)), dtype=torch.float32)

    def learned_cutoffs(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        male_fc = cfg.male_fc_min + torch.sigmoid(self.raw_male_fc) * (cfg.male_fc_max - cfg.male_fc_min)
        female_fc = cfg.female_fc_min + torch.sigmoid(self.raw_female_fc) * (cfg.female_fc_max - cfg.female_fc_min)
        return male_fc, female_fc

    def filter_prior_logits(self, device: torch.device) -> torch.Tensor:
        # Differentiable Butterworth-like priors: male prefers low frequencies,
        # female prefers high frequencies, but the cutoffs are trainable.
        cfg = self.cfg
        freqs = torch.fft.rfftfreq(N_FFT, 1.0 / SR).to(device).clamp_min(1.0)
        male_fc, female_fc = self.learned_cutoffs()
        male = 1.0 / torch.sqrt(1.0 + (freqs / male_fc).pow(2.0 * cfg.butter_order))
        female = 1.0 / torch.sqrt(1.0 + (female_fc / freqs).pow(2.0 * cfg.butter_order))
        priors = torch.stack([male, female], dim=0).clamp(1e-4, 1.0 - 1e-4)
        return cfg.prior_weight * torch.log(priors / (1.0 - priors)).view(1, 2, -1, 1)

    def forward(self, mix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The neural logits and filter-prior logits are added before softmax.
        length = mix.shape[-1]
        mix_spec = stft_torch(mix, self.window)
        mag = torch.log1p(torch.abs(mix_spec)).unsqueeze(1)
        logits = self.net(mag) + self.filter_prior_logits(mix.device)
        masks = torch.softmax(logits, dim=1)
        specs = masks * mix_spec.unsqueeze(1)
        male = istft_torch(specs[:, 0], self.window, length)
        female = istft_torch(specs[:, 1], self.window, length)
        return torch.stack([male, female], dim=1), masks


# ============================================================
# 6. Training datasets
#    RandomSegmentDataset: two-output supervised model.
#    TargetSegmentDataset: target-speaker model with a reference utterance.
# ============================================================

class RandomSegmentDataset(torch.utils.data.Dataset):
    def __init__(self, mix: np.ndarray, male: np.ndarray, female: np.ndarray, segment_samples: int, steps: int) -> None:
        self.mix = torch.from_numpy(mix.astype(np.float32))
        self.targets = torch.from_numpy(np.stack([male, female]).astype(np.float32))
        self.segment_samples = segment_samples
        self.steps = steps
        self.max_start = max(0, len(mix) - segment_samples)

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_start <= 0:
            start = 0
        else:
            start = int(torch.randint(0, self.max_start + 1, (1,)).item())
        end = start + self.segment_samples
        x = self.mix[start:end]
        y = self.targets[:, start:end]
        if x.numel() < self.segment_samples:
            pad = self.segment_samples - x.numel()
            x = F.pad(x, (0, pad))
            y = F.pad(y, (0, pad))
        gain = float(torch.empty(1).uniform_(0.85, 1.15).item())
        return x * gain, y * gain


class TargetSegmentDataset(torch.utils.data.Dataset):
    def __init__(self, mix: np.ndarray, male: np.ndarray, female: np.ndarray, segment_samples: int, steps: int) -> None:
        self.mix = torch.from_numpy(mix.astype(np.float32))
        self.refs = torch.from_numpy(np.stack([male, female]).astype(np.float32))
        self.segment_samples = segment_samples
        self.steps = steps
        self.max_start = max(0, len(mix) - segment_samples)

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Alternate target_id between male and female so one model learns both target extractions.
        target_id = index % 2
        if self.max_start <= 0:
            start = 0
        else:
            start = int(torch.randint(0, self.max_start + 1, (1,)).item())
        end = start + self.segment_samples
        mix = self.mix[start:end]
        target = self.refs[target_id, start:end]
        ref = self.refs[target_id]
        if mix.numel() < self.segment_samples:
            pad = self.segment_samples - mix.numel()
            mix = F.pad(mix, (0, pad))
            target = F.pad(target, (0, pad))
        gain = float(torch.empty(1).uniform_(0.85, 1.15).item())
        return mix * gain, ref, target * gain


# ============================================================
# 7. Trainable reference-guided separator
#    简化版 VoiceFilter 思路：mix + target_ref -> target_speech。
# ============================================================

class TrainableReferenceGuidedSeparator(nn.Module):
    def __init__(self, channels: int = 24, prior_strength: float = 0.8) -> None:
        super().__init__()
        self.prior_strength = prior_strength
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            ConvBlock(channels, 1),
            ConvBlock(channels, 2),
            ConvBlock(channels, 4),
            ConvBlock(channels, 8),
            nn.Conv2d(channels, 1, 1),
        )
        self.register_buffer("window", torch.hann_window(WIN), persistent=False)
        freqs = torch.fft.rfftfreq(N_FFT, 1.0 / SR).clamp_min(1.0)
        band_edges = torch.tensor([80, 150, 300, 600, 1000, 1800, 3000, 5000], dtype=torch.float32)
        centers = torch.sqrt(band_edges[:-1] * band_edges[1:])
        widths = (band_edges[1:] - band_edges[:-1]).clamp_min(1.0)
        filters = []
        for center, width in zip(centers, widths):
            filt = torch.exp(-0.5 * ((freqs - center) / (0.45 * width)) ** 2)
            filters.append(filt / (filt.max() + EPS))
        self.register_buffer("filterbank", torch.stack(filters, dim=0), persistent=False)

    def reference_template(self, ref: torch.Tensor) -> torch.Tensor:
        # Estimate a target speaker template from the full reference utterance.
        # A small Gaussian filterbank turns the spectral envelope into band features.
        ref_spec = stft_torch(ref, self.window)
        ref_mag = torch.log1p(torch.abs(ref_spec))
        active = (ref_mag.mean(dim=1, keepdim=True) > 0.04 * ref_mag.amax(dim=(1, 2), keepdim=True)).float()
        env = (ref_mag * active).sum(dim=2) / (active.sum(dim=2) + EPS)
        env = (env - env.mean(dim=1, keepdim=True)) / (env.std(dim=1, keepdim=True) + EPS)
        band_energy = torch.einsum("bf,kf->bk", torch.relu(env), self.filterbank)
        band_prior = torch.einsum("bk,kf->bf", band_energy, self.filterbank)
        band_prior = (band_prior - band_prior.mean(dim=1, keepdim=True)) / (band_prior.std(dim=1, keepdim=True) + EPS)
        return env, band_prior

    def forward(self, mix: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Features are: mixture spectrum, target template map, and their similarity.
        length = mix.shape[-1]
        mix_spec = stft_torch(mix, self.window)
        mix_mag = torch.log1p(torch.abs(mix_spec))
        mix_norm = (mix_mag - mix_mag.mean(dim=1, keepdim=True)) / (mix_mag.std(dim=1, keepdim=True) + EPS)
        template, band_prior = self.reference_template(ref)
        template_map = template[:, :, None].expand_as(mix_mag)
        similarity = mix_norm * template[:, :, None]
        features = torch.stack([mix_norm, template_map, similarity], dim=1)
        logits = self.net(features).squeeze(1) + self.prior_strength * band_prior[:, :, None]
        mask = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
        target = istft_torch(mix_spec * mask, self.window, length)
        return target, mask


# ============================================================
# 8. Loss functions
#    组合时域 L1、频谱 L1、混合/残差信息和滤波约束。
# ============================================================

def spectral_l1(est: torch.Tensor, target: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    b, s, t = est.shape
    est_mag = torch.log1p(torch.abs(stft_torch(est.reshape(b * s, t), window)))
    tar_mag = torch.log1p(torch.abs(stft_torch(target.reshape(b * s, t), window)))
    return F.l1_loss(est_mag, tar_mag)


def single_spectral_l1(est: torch.Tensor, target: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    est_mag = torch.log1p(torch.abs(stft_torch(est, window)))
    tar_mag = torch.log1p(torch.abs(stft_torch(target, window)))
    return F.l1_loss(est_mag, tar_mag)


def trainable_ref_loss(
    model: TrainableReferenceGuidedSeparator,
    mix: torch.Tensor,
    ref: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    # Target extraction loss: output should match the selected speaker,
    # while the residual mix-output should match the other speaker.
    pred, mask = model(mix, ref)
    wave = F.l1_loss(pred, target)
    spec = single_spectral_l1(pred, target, model.window)
    leakage = F.l1_loss(mix - pred, mix - target)
    mask_smooth = F.l1_loss(mask[:, :, 1:], mask[:, :, :-1])
    total = 0.45 * wave + 1.00 * spec + 0.25 * leakage + 0.02 * mask_smooth
    return total, {
        "total": float(total.detach().cpu()),
        "wave": float(wave.detach().cpu()),
        "spec": float(spec.detach().cpu()),
        "leakage": float(leakage.detach().cpu()),
        "mask_smooth": float(mask_smooth.detach().cpu()),
    }


def diff_loss(model: DifferentiableFilterSeparator, mix: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    # Two-output supervised separation loss.
    pred, _ = model(mix)
    wave = F.l1_loss(pred, target)
    spec = spectral_l1(pred, target, model.window)
    consistency = F.l1_loss(pred.sum(dim=1), mix)

    male_fc, female_fc = model.learned_cutoffs()
    pred_male_mag = torch.abs(stft_torch(pred[:, 0], model.window))
    pred_female_mag = torch.abs(stft_torch(pred[:, 1], model.window))
    freqs = torch.fft.rfftfreq(N_FFT, 1.0 / SR).to(mix.device)
    male_high_w = torch.sigmoid((freqs - female_fc) / 160.0).view(1, -1, 1)
    female_low_w = torch.sigmoid((male_fc - freqs) / 160.0).view(1, -1, 1)
    male_high = (pred_male_mag * male_high_w).sum() / (pred_male_mag.sum() + EPS)
    female_low = (pred_female_mag * female_low_w).sum() / (pred_female_mag.sum() + EPS)
    gap = F.relu(male_fc + 250.0 - female_fc) / 1000.0
    filt = male_high + female_low + gap

    total = 0.40 * wave + 1.00 * spec + 0.20 * consistency + 0.10 * filt
    stats = {
        "total": float(total.detach().cpu()),
        "wave": float(wave.detach().cpu()),
        "spec": float(spec.detach().cpu()),
        "consistency": float(consistency.detach().cpu()),
        "filter": float(filt.detach().cpu()),
        "male_fc": float(male_fc.detach().cpu()),
        "female_fc": float(female_fc.detach().cpu()),
    }
    return total, stats


# ============================================================
# 9. Model training and inference wrappers
#    训练完成后直接对完整 mix_speech.wav 推理并写出 wav 文件。
# ============================================================

def train_diff_model(mix: np.ndarray, male: np.ndarray, female: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, DifferentiableFilterSeparator, list[dict[str, float]]]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    model = DifferentiableFilterSeparator(DiffConfig(channels=args.channels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ds = RandomSegmentDataset(mix, male, female, int(args.segment_seconds * SR), args.steps_per_epoch)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        totals: dict[str, float] = {}
        count = 0
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss, stats = diff_loss(model, x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
        row = {key: value / max(1, count) for key, value in totals.items()}
        row["epoch"] = float(epoch)
        history.append(row)
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | loss={row['total']:.5f} | "
                f"fc_m={row['male_fc']:.1f} Hz | fc_f={row['female_fc']:.1f} Hz"
            )

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(mix.astype(np.float32)).unsqueeze(0).to(device)
        pred, _ = model(x)
    pred_np = pred[0].detach().cpu().numpy()
    write_wav(AUDIO_DIR / "diff_male.wav", pred_np[0])
    write_wav(AUDIO_DIR / "diff_female.wav", pred_np[1])
    write_wav(AUDIO_DIR / "diff_reconstructed_mix.wav", pred_np.sum(axis=0))
    return pred_np[0], pred_np[1], model.cpu(), history


def train_reference_guided_model(
    mix: np.ndarray,
    male: np.ndarray,
    female: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, TrainableReferenceGuidedSeparator, list[dict[str, float]]]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed + 17)
    model = TrainableReferenceGuidedSeparator(channels=args.channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ds = TargetSegmentDataset(mix, male, female, int(args.segment_seconds * SR), args.steps_per_epoch * 2)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    history: list[dict[str, float]] = []

    for epoch in range(1, args.ref_epochs + 1):
        totals: dict[str, float] = {}
        count = 0
        model.train()
        for x, ref, target in loader:
            x = x.to(device)
            ref = ref.to(device)
            target = target.to(device)
            opt.zero_grad(set_to_none=True)
            loss, stats = trainable_ref_loss(model, x, ref, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
        row = {key: value / max(1, count) for key, value in totals.items()}
        row["epoch"] = float(epoch)
        history.append(row)
        if epoch == 1 or epoch % 20 == 0 or epoch == args.ref_epochs:
            print(f"Ref epoch {epoch:03d} | loss={row['total']:.5f} | spec={row['spec']:.5f}")

    model.eval()
    with torch.no_grad():
        mix_t = torch.from_numpy(mix.astype(np.float32)).unsqueeze(0).to(device)
        male_ref_t = torch.from_numpy(male.astype(np.float32)).unsqueeze(0).to(device)
        female_ref_t = torch.from_numpy(female.astype(np.float32)).unsqueeze(0).to(device)
        pred_male, _ = model(mix_t, male_ref_t)
        pred_female, _ = model(mix_t, female_ref_t)
    male_np = pred_male[0].detach().cpu().numpy()
    female_np = pred_female[0].detach().cpu().numpy()
    write_wav(AUDIO_DIR / "tref_male.wav", male_np)
    write_wav(AUDIO_DIR / "tref_female.wav", female_np)
    write_wav(AUDIO_DIR / "tref_reconstructed_mix.wav", male_np + female_np)
    return male_np, female_np, model.cpu(), history


# ============================================================
# 10. Objective metrics
#     这些指标用真实 male/female reference 评价不同分离方法。
# ============================================================

def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    scale = np.dot(estimate, reference) / (np.dot(reference, reference) + EPS)
    target = scale * reference
    noise = estimate - target
    return float(10.0 * np.log10((np.sum(target * target) + EPS) / (np.sum(noise * noise) + EPS)))


def corrcoef(reference: np.ndarray, estimate: np.ndarray) -> float:
    if np.std(reference) < 1e-8 or np.std(estimate) < 1e-8:
        return 0.0
    return float(np.corrcoef(reference, estimate)[0, 1])


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    err = reference - estimate
    return float(10.0 * np.log10((np.sum(reference * reference) + EPS) / (np.sum(err * err) + EPS)))


def spectral_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    _, _, ref_stft = signal.stft(reference, fs=SR, nperseg=1024, noverlap=768)
    _, _, est_stft = signal.stft(estimate, fs=SR, nperseg=1024, noverlap=768)
    return float(np.mean(np.abs(np.log1p(np.abs(ref_stft)) - np.log1p(np.abs(est_stft)))))


def metric_rows(
    male_ref: np.ndarray,
    female_ref: np.ndarray,
    mix: np.ndarray,
    method_outputs: list[tuple[str, np.ndarray, np.ndarray]],
    diff_male: np.ndarray,
    diff_female: np.ndarray,
) -> list[dict[str, str | float]]:
    rows = []
    all_methods = method_outputs + [("DifferentiableFilter", diff_male, diff_female)]
    for method, male_est, female_est in all_methods:
        rec = male_est + female_est
        rec_err = float(np.linalg.norm(mix - rec) / (np.linalg.norm(mix) + EPS))
        for stem, ref, est in [("male", male_ref, male_est), ("female", female_ref, female_est)]:
            rows.append(
                {
                    "method": method,
                    "stem": stem,
                    "si_sdr_db": si_sdr(ref, est),
                    "snr_db": snr_db(ref, est),
                    "corr": corrcoef(ref, est),
                    "mse": float(np.mean(np.square(ref - est))),
                    "spectral_l1": spectral_error(ref, est),
                    "reconstruction_error": rec_err,
                }
            )
    return rows


# ============================================================
# 11. Result saving and plotting
#     写 CSV、画波形/频谱/指标图、训练曲线和滤波器响应。
# ============================================================

def save_metrics(rows: list[dict[str, str | float]]) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGE_DIR / "problem4_speech_metrics.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_wave(ax: plt.Axes, x: np.ndarray, title: str, color: str) -> None:
    max_points = 20_000
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points).astype(int)
        t = idx / SR
        y = x[idx]
    else:
        t = np.arange(len(x)) / SR
        y = x
    ax.plot(t, y, color=color, linewidth=0.5)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)


def plot_spec(ax: plt.Axes, x: np.ndarray, title: str, color: str) -> None:
    f, pxx = signal.welch(x, fs=SR, nperseg=1024, noverlap=768)
    db = 10.0 * np.log10(pxx + EPS)
    ax.plot(f, db, color=color, linewidth=0.9)
    ax.set_xlim(0, 5000)
    ax.set_ylim(np.max(db) - 90, np.max(db) + 5)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Frequency / Hz")
    ax.set_ylabel("PSD / dB")
    ax.grid(True, alpha=0.25)


def save_comparison_figure(
    male: np.ndarray,
    female: np.ndarray,
    mix: np.ndarray,
    butter_male: np.ndarray,
    butter_female: np.ndarray,
    diff_male: np.ndarray,
    diff_female: np.ndarray,
    rows: list[dict[str, str | float]],
) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 3, figsize=(18, 18), constrained_layout=True)
    fig.suptitle("Controlled speech separation: Butterworth vs differentiable filter model", fontsize=16, fontweight="bold")

    signals = [
        ("Reference male", male, "#2563eb"),
        ("Reference female", female, "#dc2626"),
        ("Mixed speech", mix, "#334155"),
        ("Butterworth male", butter_male, "#1d4ed8"),
        ("Butterworth female", butter_female, "#b91c1c"),
        ("Differentiable male", diff_male, "#0f766e"),
        ("Differentiable female", diff_female, "#7c3aed"),
    ]
    for idx, (title, data, color) in enumerate(signals[:5]):
        plot_wave(axes[idx, 0], data, title + " waveform", color)
        plot_spec(axes[idx, 1], data, title + " spectrum", color)

    plot_wave(axes[0, 2], diff_male, "Differentiable male waveform", "#0f766e")
    plot_spec(axes[1, 2], diff_male, "Differentiable male spectrum", "#0f766e")
    plot_wave(axes[2, 2], diff_female, "Differentiable female waveform", "#7c3aed")
    plot_spec(axes[3, 2], diff_female, "Differentiable female spectrum", "#7c3aed")

    labels = [f"{r['method']}-{r['stem']}" for r in rows]
    si = [float(r["si_sdr_db"]) for r in rows]
    corr = [float(r["corr"]) for r in rows]
    x = np.arange(len(labels))
    ax = axes[4, 2]
    ax.bar(x - 0.18, si, width=0.36, label="SI-SDR (dB)", color="#0891b2")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, corr, width=0.36, label="Corr", color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Objective metrics", fontweight="bold")
    ax.set_ylabel("SI-SDR / dB")
    ax2.set_ylabel("Correlation")
    ax.grid(True, axis="y", alpha=0.25)

    fig.savefig(IMAGE_DIR / "problem4_speech_method_comparison.png", bbox_inches="tight")
    plt.close(fig)


def save_training_loss(history: list[dict[str, float]]) -> None:
    epochs = [row["epoch"] for row in history]
    loss = [row["total"] for row in history]
    male_fc = [row["male_fc"] for row in history]
    female_fc = [row["female_fc"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].plot(epochs, loss, linewidth=2)
    axes[0].set_title("Differentiable filter training loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, male_fc, label="male LPF cutoff", linewidth=2)
    axes[1].plot(epochs, female_fc, label="female HPF cutoff", linewidth=2)
    axes[1].set_title("Learned cutoff frequencies", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Frequency / Hz")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.savefig(IMAGE_DIR / "problem4_speech_training_loss.png", bbox_inches="tight")
    plt.close(fig)


def save_reference_training_loss(history: list[dict[str, float]]) -> None:
    epochs = [row["epoch"] for row in history]
    loss = [row["total"] for row in history]
    spec = [row["spec"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(epochs, loss, label="total", linewidth=2)
    ax.plot(epochs, spec, label="spectral", linewidth=2)
    ax.set_title("Trainable reference-guided extraction loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(IMAGE_DIR / "problem4_reference_training_loss.png", bbox_inches="tight")
    plt.close(fig)


def filter_response(kind: str, filt: object) -> tuple[np.ndarray, np.ndarray]:
    if kind == "sos":
        return signal.sosfreqz(filt, worN=2048, fs=SR)
    b, a = filt
    return signal.freqz(b, a, worN=2048, fs=SR)


def save_filter_response_figure(model: DifferentiableFilterSeparator, fixed_outputs: list[dict[str, object]]) -> None:
    freqs = torch.fft.rfftfreq(N_FFT, 1.0 / SR).clamp_min(1.0)
    with torch.no_grad():
        male_fc, female_fc = model.learned_cutoffs()
        order = model.cfg.butter_order
        diff_male = 1.0 / torch.sqrt(1.0 + (freqs / male_fc).pow(2.0 * order))
        diff_female = 1.0 / torch.sqrt(1.0 + (female_fc / freqs).pow(2.0 * order))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)
    for item in fixed_outputs:
        method = str(item["method"])
        w_low, h_low = filter_response(str(item["low_kind"]), item["low_filter"])
        w_high, h_high = filter_response(str(item["high_kind"]), item["high_filter"])
        axes[0].plot(w_low, np.abs(h_low), linewidth=1.5, label=f"{method} male")
        axes[0].plot(w_high, np.abs(h_high), linewidth=1.5, linestyle="--", label=f"{method} female")
        axes[1].plot(w_low, 20 * np.log10(np.abs(h_low) + EPS), linewidth=1.5)
        axes[1].plot(w_high, 20 * np.log10(np.abs(h_high) + EPS), linewidth=1.5, linestyle="--")

    axes[0].plot(freqs, diff_male, color="black", linewidth=2.4, label=f"Learned male fc={male_fc.item():.1f} Hz")
    axes[0].plot(freqs, diff_female, color="black", linestyle="--", linewidth=2.4, label=f"Learned female fc={female_fc.item():.1f} Hz")
    axes[0].set_xlim(0, 5000)
    axes[0].set_ylim(-0.03, 1.08)
    axes[0].set_title("Filter responses: linear gain", fontweight="bold")
    axes[0].set_xlabel("Frequency / Hz")
    axes[0].set_ylabel("Gain")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(freqs, 20 * torch.log10(diff_male + EPS), color="black", linewidth=2.4)
    axes[1].plot(freqs, 20 * torch.log10(diff_female + EPS), color="black", linestyle="--", linewidth=2.4)
    axes[1].set_xlim(0, 5000)
    axes[1].set_ylim(-90, 5)
    axes[1].set_title("Filter responses: dB", fontweight="bold")
    axes[1].set_xlabel("Frequency / Hz")
    axes[1].set_ylabel("Magnitude / dB")
    axes[1].grid(True, alpha=0.3)

    fig.savefig(IMAGE_DIR / "problem4_speech_filter_response_comparison.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 12. Main experiment pipeline
#     按顺序执行：构造混合 -> 固定滤波 -> reference 方法 -> 训练模型 -> 指标/图表。
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed Butterworth and differentiable filter speech separation.")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--steps-per-epoch", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ref-epochs", type=int, default=160)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--male-cutoff", type=float, default=1000.0)
    parser.add_argument("--female-cutoff", type=float, default=1700.0)
    args = parser.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    male, female, mix = prepare_reference_mix()
    fixed_outputs, _, _ = fixed_filter_separations(mix, args.male_cutoff, args.female_cutoff)
    butter = next(item for item in fixed_outputs if item["method"] == "Butterworth")
    butter_male = butter["male"]
    butter_female = butter["female"]
    ref_male, ref_female, ref_debug = reference_guided_filter_separate(mix, male, female)
    tref_male, tref_female, tref_model, tref_history = train_reference_guided_model(mix, male, female, args)
    diff_male, diff_female, model, history = train_diff_model(mix, male, female, args)
    method_outputs = [(str(item["method"]), item["male"], item["female"]) for item in fixed_outputs]
    method_outputs.append(("ReferenceGuidedFilter", ref_male, ref_female))
    method_outputs.append(("TrainableRefGuided", tref_male, tref_female))
    rows = metric_rows(male, female, mix, method_outputs, diff_male, diff_female)
    save_metrics(rows)
    save_training_loss(history)
    save_reference_training_loss(tref_history)
    save_filter_response_figure(model, fixed_outputs)
    save_comparison_figure(male, female, mix, butter_male, butter_female, diff_male, diff_female, rows)

    print("\nMetrics:")
    for row in rows:
        print(
            f"{row['method']:>22s} {row['stem']:>6s} | "
            f"SI-SDR={float(row['si_sdr_db']):7.2f} dB | "
            f"Corr={float(row['corr']):.4f} | "
            f"SNR={float(row['snr_db']):7.2f} dB"
        )
    male_fc, female_fc = model.learned_cutoffs()
    print(f"\nLearned male cutoff:   {male_fc.item():.2f} Hz")
    print(f"Learned female cutoff: {female_fc.item():.2f} Hz")


if __name__ == "__main__":
    main()
