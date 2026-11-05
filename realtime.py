"""Real-time face-cell demo through the Cortical Labs SDK (cl.open / neurons.loop).

    python face_cells/realtime.py                        # held-out photo of each person, live
    python face_cells/realtime.py --image photo.jpg      # "who is this?" in real time
    CL_SDK_ACCELERATED_TIME=1 python face_cells/realtime.py --no-browser   # fast, no viewer

Same code path as natha-ui/biocompute-outlier-labs `experiments/example1.py`: the SDK
runs the LIF network as a simulator subprocess at wall-clock speed and opens its
live web viewer (raw traces, spikes and stims over the 8x8 array). On a CL1 the same
loop would drive real tissue (there, skip the set_simulator_data_source call).

The culture is the one trained offline by run.py (STDP needs apply_reward, which the
SDK subprocess does not expose). Its learned excitatory weights plus the fixed lateral
inhibition are handed to the simulator as its `connectivity` matrix, with plasticity off.

Every stimulus is a real CL1-valid command: per input electrode one
StimDesign(200 us, -a uA, 200 us, +a uA) delivered as a BurstDesign(10 pulses, 200 Hz).
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # vendored organoid_simulator; the SDK forwards sys.path to its subprocess

import cl  # noqa: E402

from analysis import SD_FLOOR  # noqa: E402
from encode import ENCODERS, OUTPUT_CHANNELS, EncoderConfig, downsample, load_face, norm_code  # noqa: E402

TPS = 50                  # 20 ms ticks
WINDOW_TICKS = 3          # count spikes for 60 ms after the burst starts
REST_TICKS = 5            # 100 ms rest between trials


def stim_commands(img, d, cfg):
    """Group pulses into one (ChannelSet, StimDesign, BurstDesign) per amplitude."""
    pulses = ENCODERS[str(d["encoding"])](img, cfg)
    amps = {}
    for ch, off, a in pulses:
        if off == 0:                                   # first pulse defines the channel's amplitude
            amps[ch] = round(min(a, 3.0), 1)           # 0.1 uA steps, CL1 limit 3 uA
    by_amp = defaultdict(list)
    for ch, a in amps.items():
        if a >= 0.1:
            by_amp[a].append(ch)
    hz = min(200.0, 1000.0 / cfg.pulse_gap_ms)
    burst = cl.BurstDesign(cfg.n_pulses, hz)
    return [(cl.ChannelSet(*chs), cl.StimDesign(200, -a, 200, a), burst) for a, chs in by_amp.items()]


def encode_photo(path, d):
    img = downsample(load_face(path))
    if str(d["code"]) == "norm":
        ch = float(d["charge"]) if "charge" in d else 0.0
        img = norm_code(img, d["mean_face"], ch if ch > 0 else None)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--culture", type=Path, default=HERE.parent / "results" / "trained_culture.npz")
    ap.add_argument("--faces", type=Path, default=HERE.parent / "data" / "faces")
    ap.add_argument("--image", type=Path, action="append", help="photo(s) to identify; default = one held-out photo per person")
    ap.add_argument("--reps", type=int, default=8, help="presentations per photo")
    ap.add_argument("--n-blank", type=int, default=30)
    ap.add_argument("--k", type=float, default=5.0)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-hold", action="store_true")
    a = ap.parse_args()

    d = np.load(a.culture)
    people = [str(p) for p in d["people"]]
    face_cells = [int(c) for c in d["face_cells"]]
    coupling = float(d["coupling"])
    inh = float(d["inhibition"]) if "inhibition" in d else 4.0
    W = d["W"].copy()
    out = [c for c in range(64) if c // 8 >= 4 and c not in (56, 63)]
    W[np.ix_(out, out)] -= inh                          # same fixed lateral inhibition as training
    np.fill_diagonal(W, 0.0)
    cfg = EncoderConfig()

    photos = a.image or []
    if not photos:  # every 3rd photo per person was held out from training
        for p in people:
            held = sorted(a.faces.glob(f"{p}_*.png"))[2::3]
            photos.append(held[len(held) // 2])
    stimuli = [(ph.name, stim_commands(encode_photo(ph, d), d, cfg)) for ph in photos]

    # trial schedule: blanks first, then photos counterbalanced (rotating order)
    rng = np.random.default_rng(0)
    schedule = [("blank", None)] * a.n_blank
    for r in range(a.reps):
        order = np.roll(np.arange(len(stimuli)), r)
        schedule += [stimuli[i] for i in (order if r % 2 == 0 else order[::-1])]

    if cl.is_simulator():
        cl.sim.set_simulator_data_source(
            "organoid_simulator.lif_data_source:make_lif_source",
            config={"random_seed": int(d["seed"]), "coupling_gain_mV": coupling,
                    "background_drive_std_mV": 0.5,
                    "connectivity": (W / coupling).tolist()},   # W = gain * connectivity
        )

    counts = defaultdict(list)
    trial_len = WINDOW_TICKS + REST_TICKS
    with cl.open() as neurons:
        server = getattr(neurons, "_websocket_server", None)
        url = server.web_url if server is not None else None
        if url:
            print(f"live viewer: {url}")
            if not a.no_browser:
                webbrowser.open(url)
        print(f"{len(schedule)} trials ({len(schedule) * trial_len / TPS:.0f} s): {a.n_blank} blanks, then "
              + ", ".join(n for n, _ in stimuli) + f" x{a.reps}\n")
        acc = np.zeros(64)
        thr = None
        n_ticks = len(schedule) * trial_len + 1
        for tick in neurons.loop(ticks_per_second=TPS, stop_after_ticks=n_ticks):
            t, phase = divmod(tick.iteration, trial_len)
            if 1 <= phase <= WINDOW_TICKS:             # spikes arriving after the burst
                for s in tick.analysis.spikes:
                    acc[s.channel] += 1
            if phase == WINDOW_TICKS and t < len(schedule):
                name, _ = schedule[t]
                counts[name].append(acc[OUTPUT_CHANNELS].copy())
                if name != "blank":
                    if thr is None:
                        B = np.array(counts["blank"])
                        mu, sd = B.mean(0), np.maximum(B.std(0), SD_FLOOR)
                        thr = True
                    z = (acc[OUTPUT_CHANNELS] - mu) / sd
                    best = int(np.argmax(z))
                    tags = "  ".join(f"{p}:{z[OUTPUT_CHANNELS.index(c)]:4.1f}" for p, c in zip(people, face_cells))
                    print(f"  t={tick.iteration / TPS:5.1f}s  {name:>16s}  top ch{OUTPUT_CHANNELS[best]:<2d} z={z[best]:4.1f} | face cells  {tags}")
                acc[:] = 0
            if phase == 0 and t < len(schedule):         # start of a trial: deliver the face
                acc[:] = 0
                _, cmds = schedule[t]
                for chs, design, burst in cmds or []:
                    neurons.stim(chs, design, burst)

    # ---- verdicts
    B = np.array(counts["blank"]); mu, sd = B.mean(0), np.maximum(B.std(0), SD_FLOOR)
    print("\nVerdict (mean over presentations, z vs blank baseline; FIRED = above k SD)")
    for name, _ in stimuli:
        z = (np.mean(counts[name], 0) - mu) / sd
        fired = [p for p, c in zip(people, face_cells) if z[OUTPUT_CHANNELS.index(c)] > a.k]
        verdict = (f"identified as {fired[0]}" if len(fired) == 1 else
                   "unknown (no face cell fired)" if not fired else "ambiguous: " + ", ".join(fired))
        zs = "  ".join(f"{p}'s ch{c} z={z[OUTPUT_CHANNELS.index(c)]:.1f}" for p, c in zip(people, face_cells))
        print(f"  {name:>16s} -> {verdict:28s} [{zs}]")

    if url and not a.no_hold:
        input("\nPress Enter to close the viewer...")


if __name__ == "__main__":
    main()
