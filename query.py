"""Show a new photo to a trained culture and report which output neuron fires.

    uv run python experiments/face_cells/query.py --culture face_cell_results/trained_culture.npz photo.jpg

Rebuilds the culture from the saved seed and learned excitatory weights (plasticity
frozen), measures a blank baseline, presents the photo `--reps` times and
reports (a) every output cell above baseline + k SD and (b) whether the face cell
learned for each known person fired, so it gives a predicted identity or "unknown".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analysis import SD_FLOOR  # noqa: E402
from culture import Culture  # noqa: E402
from encode import ENCODERS, OUTPUT_CHANNELS, EncoderConfig, downsample, load_face, norm_code  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--culture", type=Path, default=Path("face_cell_results/trained_culture.npz"))
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--k", type=float, default=5.0)
    a = ap.parse_args()

    d = np.load(a.culture, allow_pickle=False)
    c = Culture(seed=int(d["seed"]), coupling_gain_mV=float(d["coupling"]), stdp=False,
                inhibition_mV=float(d["inhibition"]) if "inhibition" in d else 8.0)
    c.W_exc = d["W"].copy()
    c.src._W = c.W_exc + c.W_inh
    c.run(25_000)

    img = downsample(load_face(a.image))
    if str(d["code"]) == "norm":
        ch = float(d["charge"]) if "charge" in d else 0.0
        img = norm_code(img, d["mean_face"], ch if ch > 0 else None)
    pulses = ENCODERS[str(d["encoding"])](img, EncoderConfig())

    B = np.array([c.trial([])[OUTPUT_CHANNELS] for _ in range(30)])
    R = np.array([c.trial(pulses)[OUTPUT_CHANNELS] for _ in range(a.reps)])
    z = (R.mean(0) - B.mean(0)) / np.maximum(B.std(0), SD_FLOOR)

    print(f"\n{a.image.name}: {len(pulses)} pulses, disrupted {c.n_disrupted}/{len(c.log)} trials")
    order = np.argsort(-z)
    print("top output cells:", ", ".join(f"ch{OUTPUT_CHANNELS[i]} z={z[i]:.1f}" for i in order[:5]))
    hits = []
    for name, cell in zip(d["people"], d["face_cells"]):
        zi = z[OUTPUT_CHANNELS.index(int(cell))]
        fired = zi > a.k
        hits.append((name, zi)) if fired else None
        print(f"  {name:>12s}'s face cell ch{int(cell):<3d} z={zi:5.1f} {'FIRED' if fired else ''}")
    if len(hits) == 1:
        print(f"-> identified as {hits[0][0]}")
    elif not hits:
        print("-> no known face cell fired (unknown person)")
    else:
        print(f"-> ambiguous: {', '.join(h[0] for h in hits)}")


if __name__ == "__main__":
    main()
