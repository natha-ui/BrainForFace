"""Minimal, slightly artistic summary: face -> 8x4 code -> MEA, with each person's face cell.

    python experiments/face_cells/visualise.py --results results/ --faces faces/ --out results/face_cells_poster.png

One column per person:
  * the face (circular crop),
  * the 8x4 norm-based code that is delivered to the input half of the array,
  * the 8x8 MEA: input electrodes glow by stimulation amplitude; output electrodes are
    dots sized by their (held-out, post-learning) response to that person. A ring marks
    the candidate cell for this person; a solid ring + label means it passed the
    face-cell test (responds above baseline + 5 SD to this person only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from encode import ENCODERS, NON_SPIKING, OUTPUT_CHANNELS, EncoderConfig, downsample, load_face, norm_code  # noqa: E402
from run import load_people  # noqa: E402

BG, INK, DIM = "#0d1117", "#e6edf3", "#3a4350"
PALETTE = ["#ff7b72", "#79c0ff", "#d2a8ff", "#7ee787", "#ffa657", "#f2cc60"]


def circ_face(path, size=220):
    im = Image.open(path).convert("L")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, int((h - s) * 0.3), (w - s) // 2 + s, int((h - s) * 0.3) + s)).resize((size, size))
    a = np.asarray(im, dtype=float) / 255
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
    alpha = np.clip((1 - r) * 25, 0, 1)
    rgba = np.dstack([a, a, a, alpha])
    return rgba


def typical(paths):
    """The most typical photo of a person: closest to their own average face."""
    X = np.array([load_face(p) for p in paths])
    return paths[int(((X - X.mean(0)) ** 2).sum(axis=(1, 2)).argmin())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--faces", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--run", default="STDP")
    a = ap.parse_args()

    res = json.loads((a.results / "results.json").read_text())
    d = np.load(a.results / "trained_culture.npz")
    names = res["people"]
    ev = res["runs"][a.run]["post"]
    T = np.array(ev["tuning_test"]); thr = np.array(ev["threshold"])
    people = load_people(a.faces)
    charge = float(d["charge"]) if "charge" in d else 0.0
    enc, cfg = ENCODERS[str(d["encoding"])], EncoderConfig()

    n = len(names)
    fig = plt.figure(figsize=(3.2 * n + 0.6, 10.2), facecolor=BG)
    fig.text(0.5, 0.965, "one neuron, one face", ha="center", color=INK, fontsize=22, family="serif", style="italic")
    fig.text(0.5, 0.935, "unsupervised STDP in a simulated CL1 culture  ·  59 LIF neurons, small-world  ·  held-out photos",
             ha="center", color="#8b949e", fontsize=9)

    for i, name in enumerate(names):
        col = PALETTE[i % len(PALETTE)]
        x0 = (i + 0.08) / n
        w = 0.84 / n
        info = ev["per_identity"][i]
        portrait = typical(people[name])

        # face
        ax = fig.add_axes([x0 + w * 0.15, 0.68, w * 0.7, 0.22]); ax.axis("off")
        ax.imshow(circ_face(portrait)); ax.add_patch(Circle((110, 110), 112, fill=False, ec=col, lw=1.5))
        ax.set_title(name, color=col, fontsize=13, family="serif", pad=6)

        # 8x4 code
        img = downsample(load_face(portrait))
        if str(d["code"]) == "norm":
            img = norm_code(img, d["mean_face"], charge if charge > 0 else None)
        ax = fig.add_axes([x0 + w * 0.33, 0.47, w * 0.34, 0.18]); ax.axis("off")
        ax.set_xlim(-.6, 3.6); ax.set_ylim(7.6, -.6)
        for r in range(8):
            for c in range(4):
                v = img[r, c]
                ax.add_patch(FancyBboxPatch((c - .42, r - .42), .84, .84, boxstyle="round,pad=0,rounding_size=.18",
                                            fc=(1, 1, 1, 0.06 + 0.9 * v), ec="none"))
        ax.set_title("deviation from average face", color="#8b949e", fontsize=7.5, pad=3)

        # MEA
        ax = fig.add_axes([x0, 0.08, w, 0.36]); ax.axis("off"); ax.set_aspect("equal")
        ax.set_xlim(-.8, 7.8); ax.set_ylim(7.8, -.8)
        stim = {chn: amp for chn, _, amp in enc(img, cfg)}
        Tmax = max(T.max(), 1e-6)
        for chn in range(64):
            r, c = chn % 8, chn // 8
            if chn in NON_SPIKING:
                ax.plot(c, r, marker="x", color=DIM, ms=4, mew=.8)
            elif c < 4:
                v = stim.get(chn, 0) / cfg.max_uA
                ax.add_patch(FancyBboxPatch((c - .3, r - .3), .6, .6, boxstyle="round,pad=0,rounding_size=.15",
                                            fc=matplotlib.colors.to_rgba(col, 0.1 + 0.85 * v), ec="none"))
            else:
                k = OUTPUT_CHANNELS.index(chn)
                resp = T[i, k] / Tmax
                above = T[i, k] > thr[k]
                ax.add_patch(Circle((c, r), 0.08 + 0.36 * np.sqrt(resp), fc=INK if above else "#6e7681",
                                    alpha=0.9 if above else 0.5, ec="none"))
        cc = info["cell"]; r, c = cc % 8, cc // 8
        ok = info["is_face_cell"]
        for rad, al in ((.62, .15), (.52, .35)) if ok else ((.52, .0),):
            ax.add_patch(Circle((c, r), rad, fc="none", ec=col, lw=6 if rad > .6 else 2, alpha=al))
        ax.add_patch(Circle((c, r), .48, fc="none", ec=col, lw=1.6, ls="-" if ok else (0, (2, 2))))
        ax.axvline(3.5, color=DIM, lw=.8, ls=(0, (1, 3)))
        ax.text(1.5, 8.35, "input", ha="center", color="#8b949e", fontsize=7.5)
        ax.text(5.5, 8.35, "output", ha="center", color="#8b949e", fontsize=7.5)
        label = f"ch{cc}  ·  face cell" if ok else f"ch{cc}  ·  candidate"
        ax.text(3.5, 9.3, label, ha="center", color=col if ok else "#8b949e", fontsize=10, family="serif")
        ax.text(3.5, 10.0, f"z {info['z']:.1f}   AUC {info['AUC']:.2f}   SI {info['SI']:+.2f}", ha="center",
                color="#8b949e", fontsize=7.5)

    fig.text(0.5, 0.015, "dot size = spikes to held-out photos of that person (after learning)  ·  white = above baseline + 5 SD"
             "  ·  solid ring = fires for this person only.  Simulation, not CL1 tissue.",
             ha="center", color="#6e7681", fontsize=7.5)
    out = a.out or a.results / "face_cells_poster.png"
    fig.savefig(out, dpi=180, facecolor=BG)
    print("wrote", out)


if __name__ == "__main__":
    main()
