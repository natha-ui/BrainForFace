"""Image -> stimulation encoders for the face-cell experiment.

Array split (8x8 MEA, channel = row + 8*col):
  * INPUT  = columns 0-3. A face becomes an 8-row x 4-col image, one pixel per
             electrode. Dead electrodes 0, 4, 7 simply drop their pixel.
  * OUTPUT = columns 4-7, live channels only (30 neurons). These are the
             candidate "face cells". They are never stimulated directly, so any
             response has to travel through the small-world network.

Each encoder returns a list of pulses (channel, frame_offset, amplitude_uA).
Pulses are charge-balanced biphasic 200 us / 200 us, cathodic first, which is
a valid cl.StimDesign(200, -a, 200, a) on a CL1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROWS, IN_COLS = 8, 4
NON_SPIKING = frozenset((0, 4, 7, 56, 63))
FPS = 25_000


def ch(r: int, c: int) -> int:
    return r + 8 * c


INPUT_MAP = np.array([[ch(r, c) for c in range(IN_COLS)] for r in range(ROWS)])  # (8,4)
OUTPUT_CHANNELS = [x for x in range(64) if x // 8 >= 4 and x not in NON_SPIKING]
HIRES = (32, 64)  # (w, h) working resolution before the final 4x8 downsample

Pulse = tuple[int, int, float]  # (channel, frame_offset, amplitude_uA)


# --------------------------------------------------------------------------- images
def load_face(path: str | Path) -> np.ndarray:
    """Grayscale, centre-crop to a 1:2 (w:h) strip, resize to HIRES. Returns
    float array in [0,1], shape (64, 32). Works best on roughly centred headshots."""
    im = Image.open(path).convert("L")
    w, h = im.size
    if h >= 2 * w:
        crop_w, crop_h = w, 2 * w
    else:
        crop_w, crop_h = h // 2, h
    left, top = (w - crop_w) // 2, (h - crop_h) // 2
    im = im.crop((left, top, left + crop_w, top + crop_h)).resize(HIRES, Image.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255.0


def downsample(hires: np.ndarray) -> np.ndarray:
    """(64,32) -> (8,4) by block averaging, then per-image contrast normalisation
    so overall brightness does not become an identity cue."""
    small = hires.reshape(ROWS, hires.shape[0] // ROWS, IN_COLS, hires.shape[1] // IN_COLS).mean(axis=(1, 3))
    lo, hi = np.percentile(small, 5), np.percentile(small, 95)
    return np.clip((small - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def augment(hires: np.ndarray, rng: np.random.Generator, max_shift: int = 3,
            gain: float = 0.15, noise: float = 0.05) -> np.ndarray:
    """A new 'view' of the same face: small translation, brightness/contrast
    jitter, pixel noise. Returns an (8,4) encoded image."""
    dy, dx = rng.integers(-max_shift, max_shift + 1, size=2)
    img = np.roll(np.roll(hires, dy, axis=0), dx, axis=1)
    img = img * rng.uniform(1 - gain, 1 + gain) + rng.uniform(-gain, gain) / 2
    small = downsample(np.clip(img, 0, 1))
    return np.clip(small + rng.normal(0, noise, small.shape), 0, 1)


def norm_code(img: np.ndarray, mean_face: np.ndarray, target_sum: float | None = None) -> np.ndarray:
    """Norm-based face code (Leopold et al. 2006; Chang & Tsao 2017): keep only how
    this face deviates from the average face. Positive (brighter-than-average)
    deviations are stimulated, scaled per image so every face delivers a similar
    total drive and loudness is not an identity cue."""
    dev = np.maximum(img - mean_face, 0.0)
    if target_sum is None:
        return dev / max(dev.max(), 1e-6)
    # Equal total charge per face: faces with few brighter-than-average pixels
    # (dark hair/skin) are not under-driven. Iterative so clipping at 1 is respected.
    code = np.zeros_like(dev)
    for _ in range(10):
        free = code < 1.0
        rem = target_sum - code.sum()
        if rem <= 1e-6 or dev[free].sum() <= 0:
            break
        code[free] += dev[free] * rem / dev[free].sum()
        np.clip(code, 0.0, 1.0, out=code)
    return code


# --------------------------------------------------------------------------- encoders
@dataclass
class EncoderConfig:
    max_uA: float = 1.5          # CL1 limit is 3 uA; 1.5 gives ~37 mV of drive per pulse
    pixel_floor: float = 0.15    # pixels below this are not stimulated
    n_pulses: int = 10           # spatial: repeats of the amplitude map (swept: best z, ~7% disrupted)
    pulse_gap_ms: float = 5.0    # 200 Hz: the CL1 BurstDesign maximum
    max_latency_ms: float = 20.0 # latency: darkest stimulated pixel fires this late
    max_count: int = 4           # count: brightest pixel gets this many pulses


def _ms(ms: float) -> int:
    return int(round(ms * FPS / 1000))


def encode_spatial(img: np.ndarray, cfg: EncoderConfig) -> list[Pulse]:
    """Pixel intensity -> cathodic amplitude, all electrodes at once."""
    out = []
    for r in range(ROWS):
        for c in range(IN_COLS):
            chn, p = int(INPUT_MAP[r, c]), float(img[r, c])
            if chn in NON_SPIKING or p < cfg.pixel_floor:
                continue
            for k in range(cfg.n_pulses):
                out.append((chn, _ms(k * cfg.pulse_gap_ms), p * cfg.max_uA))
    return out


def encode_latency(img: np.ndarray, cfg: EncoderConfig) -> list[Pulse]:
    """Brighter pixel -> earlier pulse (fixed amplitude). Rank-order code."""
    out = []
    for r in range(ROWS):
        for c in range(IN_COLS):
            chn, p = int(INPUT_MAP[r, c]), float(img[r, c])
            if chn in NON_SPIKING or p < cfg.pixel_floor:
                continue
            out.append((chn, _ms((1 - p) * cfg.max_latency_ms), cfg.max_uA))
    return out


def encode_count(img: np.ndarray, cfg: EncoderConfig) -> list[Pulse]:
    """Brighter pixel -> more pulses (fixed amplitude). Rate code; most charge."""
    out = []
    for r in range(ROWS):
        for c in range(IN_COLS):
            chn, p = int(INPUT_MAP[r, c]), float(img[r, c])
            if chn in NON_SPIKING or p < cfg.pixel_floor:
                continue
            for k in range(int(round(p * cfg.max_count))):
                out.append((chn, _ms(k * cfg.pulse_gap_ms / 2), cfg.max_uA))
    return out


ENCODERS = {"spatial": encode_spatial, "latency": encode_latency, "count": encode_count}
