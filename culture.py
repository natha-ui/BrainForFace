"""Thin trial-based wrapper around the repo's LIFDataSource.

We drive the data source directly (open / on_stims / read) instead of through
cl.open(): it is faster, deterministic, and lets us call apply_reward() and read
the weights, which the SDK subprocess hides.

Unsupervised STDP = the source's trace-based STDP eligibility, consolidated with
a constant reward of +1 after every exposure trial (i.e. plain Hebbian STDP).
Two stabilisers keep it from blowing up, since the model has no inhibition:
  * structural mask: weights only change on existing small-world edges;
  * synaptic scaling: each neuron's total incoming weight is renormalised to its
    starting value, so potentiating one input depresses the others
    (competition -> selectivity, Oja-style).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # vendored organoid_simulator
from cl.sim import DataSourceStim
from organoid_simulator import LIFDataSource

from encode import FPS, NON_SPIKING, Pulse

LIVE = np.array([c for c in range(64) if c not in NON_SPIKING])
OUT = np.array([c for c in LIVE if c // 8 >= 4])   # unstimulated half: where a seizure shows up


@dataclass
class SafetyConfig:
    burst_bin_ms: float = 10.0
    burst_frac: float = 0.6        # >60% of OUTPUT channels firing in one 10 ms bin = population burst
    max_rate_hz: float = 60.0      # mean per-channel rate in the window above this = over-driven
    backoff: float = 0.9           # stim scale multiplier after a disruptive trial
    recover: float = 1.02          # ...and slow recovery towards 1.0 after quiet trials
    min_scale: float = 0.4


@dataclass
class Culture:
    seed: int = 42
    coupling_gain_mV: float = 7.0      # swept: 8+ mV -> frequent population bursts
    background_drive_std_mV: float = 0.5
    stdp: bool = True
    lr: float = 0.002
    homeostasis: bool = True
    inhibition_mV: float = 0.0         # lateral inhibition: each output spike hyperpolarises every other output cell
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    def __post_init__(self):
        self.src = LIFDataSource(
            random_seed=self.seed, coupling_gain_mV=self.coupling_gain_mV,
            background_drive_std_mV=self.background_drive_std_mV,
            plasticity=self.stdp, stdp=self.stdp, plasticity_lr=self.lr,
            separate_noise_streams=True,
        )
        self.src.open()
        self.t = 0
        self.mask = self.src._W > 0
        self.in_target = self.src._W.sum(axis=1)
        self.W0 = self.src._W.copy()
        # Fixed inhibitory overlay (abstracted interneurons). The LIF read loop just
        # sums W*fired, so negative entries act as inhibition. STDP/apply_reward only
        # ever see the excitatory part (apply_reward would clip negatives to 0).
        self.W_exc = self.src._W.copy()
        self.W_inh = np.zeros_like(self.W_exc)
        if self.inhibition_mV > 0:
            self.W_inh[np.ix_(OUT, OUT)] = -self.inhibition_mV
            np.fill_diagonal(self.W_inh, 0.0)
        self.src._W = self.W_exc + self.W_inh
        self.src._use_coupling = True
        self.stim_scale = 1.0
        self.n_disrupted = 0
        self.log: list[dict] = []

    # ------------------------------------------------------------------ core
    def run(self, n_frames: int, pulses: list[Pulse] = ()) -> tuple[np.ndarray, np.ndarray]:
        if pulses:
            self.src.on_stims([
                DataSourceStim(timestamp=self.t + off, channel=c,
                               phase_durations_us=(200, 200),
                               phase_currents_uA=(-a * self.stim_scale, a * self.stim_scale))
                for c, off, a in pulses
            ])
        b = self.src.read(self.t, n_frames)
        t0 = self.t
        self.t += n_frames
        ts = np.fromiter((s.timestamp - t0 for s in b.spikes), dtype=np.int64, count=len(b.spikes))
        chs = np.fromiter((s.channel for s in b.spikes), dtype=np.int64, count=len(b.spikes))
        return ts, chs

    def trial(self, pulses: list[Pulse], window_ms: float = 60, rest_ms: float = 100,
              learn: bool = False, tag: str = "") -> np.ndarray:
        """Rest, then stimulate and count spikes per channel in the window."""
        n_win = int(window_ms * FPS / 1000)
        rts, rch = self.run(int(rest_ms * FPS / 1000))
        if self.stdp:
            self.src._elig[...] = 0.0          # only credit stimulus-locked co-activity
        ts, chs = self.run(n_win, pulses)
        counts = np.bincount(chs, minlength=64).astype(float)

        # ---- safety: population bursts / over-driving
        bin_f = int(self.safety.burst_bin_ms * FPS / 1000)
        burst = False
        if len(ts):
            m = np.isin(chs, OUT)
            bins = ts[m] // bin_f
            for b in np.unique(bins):
                if len(np.unique(chs[m][bins == b])) > self.safety.burst_frac * len(OUT):
                    burst = True
                    break
        rate = counts[OUT].mean() / (window_ms / 1000)
        disrupted = burst or rate > self.safety.max_rate_hz
        if disrupted:
            self.n_disrupted += 1
            self.stim_scale = max(self.safety.min_scale, self.stim_scale * self.safety.backoff)
        elif pulses:
            self.stim_scale = min(1.0, self.stim_scale * self.safety.recover)

        if learn and self.stdp:
            if disrupted:
                self.src._elig[...] = 0.0      # never learn from a seizure
            else:
                self.src._W = self.W_exc
                self.src.apply_reward(1.0)
                self._stabilise()
                self.W_exc = self.src._W.copy()
                self.src._W = self.W_exc + self.W_inh
        self.log.append(dict(tag=tag, rate_hz=float(rate), rest_rate_hz=float(
            len(rch) / len(LIVE) / (rest_ms / 1000)), burst=burst, disrupted=disrupted,
            scale=self.stim_scale))
        return counts

    def _stabilise(self):
        W = self.src._W
        W *= self.mask
        if self.homeostasis:
            s = W.sum(axis=1)
            ok = s > 0
            W[ok] *= (self.in_target[ok] / s[ok])[:, None]
            np.clip(W, 0.0, self.src._weight_max, out=W)

    @property
    def weights(self) -> np.ndarray:
        return self.W_exc.copy()
