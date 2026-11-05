"""Summary figures for run.py."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from encode import INPUT_MAP, NON_SPIKING, OUTPUT_CHANNELS, downsample, load_face  # noqa: E402


def make_figures(out, names, people, xf, results, enc, cfg):
    n = len(names)

    # 1. encoding: photo crop -> 8x4 image -> stimulated input electrodes on the MEA
    fig, ax = plt.subplots(3, n, figsize=(2.2 * n, 7), squeeze=False)
    for i, name in enumerate(names):
        hi = load_face(people[name][0])
        ax[0, i].imshow(hi, cmap="gray", vmin=0, vmax=1); ax[0, i].set_title(name)
        code = xf(downsample(hi))
        ax[1, i].imshow(code, cmap="gray", vmin=0, vmax=1)
        mea = np.full((8, 8), np.nan)
        for chn, _, a in enc(code, cfg):
            mea[chn % 8, chn // 8] = a
        for chn in NON_SPIKING:
            mea[chn % 8, chn // 8] = np.nan
        ax[2, i].imshow(mea, cmap="viridis", vmin=0, vmax=cfg.max_uA)
        ax[2, i].axvline(3.5, color="r", lw=1)
        for a in ax[:, i]:
            a.set_xticks([]); a.set_yticks([])
    ax[0, 0].set_ylabel("photo crop"); ax[1, 0].set_ylabel("8x4 code"); ax[2, 0].set_ylabel("MEA stim (uA)\nin | out")
    fig.tight_layout(); fig.savefig(out / "fig1_encoding.png", dpi=130); plt.close(fig)

    # 2. tuning heatmaps: test-view response minus threshold, per output cell
    runs = [(r, p) for r in results for p in ("pre", "post")]
    fig, ax = plt.subplots(len(runs), 1, figsize=(10, 1.1 * n * len(runs) + 1), squeeze=False)
    for a, (r, p) in zip(ax[:, 0], runs):
        ev = results[r][p]["eval"]
        T = np.array(ev["tuning_test"]) - np.array(ev["threshold"])[None]
        im = a.imshow(T, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        for d in ev["per_identity"]:
            if d["is_face_cell"]:
                a.add_patch(plt.Rectangle((d["cell_index"] - .5, d["identity"] - .5), 1, 1, fill=False, ec="k", lw=2))
        a.set_yticks(range(n), names); a.set_xticks(range(len(OUTPUT_CHANNELS)), [str(c) for c in OUTPUT_CHANNELS], fontsize=6)
        a.set_title(f"{r} - {p.upper()}: response - (baseline + k SD); boxed = face cell on held-out views", fontsize=9)
    fig.colorbar(im, ax=ax[:, 0], shrink=.6, label="spikes/window above threshold")
    fig.savefig(out / "fig2_tuning.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # 3. best candidate cell per identity: responses to each identity, pre vs post (STDP run)
    res = results["STDP"]
    fig, ax = plt.subplots(1, n, figsize=(2.6 * n, 3), squeeze=False)
    for i in range(n):
        d_post = res["post"]["eval"]["per_identity"][i]
        c = d_post["cell_index"]
        for off, p, col in ((-.2, "pre", "0.6"), (.2, "post", "C3")):
            T = np.array(res[p]["eval"]["tuning_test"])[:, c]
            thr = res[p]["eval"]["threshold"][c]
            ax[0, i].bar(np.arange(n) + off, T, width=.4, color=col, label=p)
            ax[0, i].hlines(thr, -.5, n - .5, colors=col, linestyles="--", lw=1)
        ax[0, i].set_xticks(range(n), names, rotation=45, fontsize=7)
        ax[0, i].set_title(f"{names[i]}'s cell: ch{d_post['cell']}" + (" *" if d_post["is_face_cell"] else ""), fontsize=9)
    ax[0, 0].set_ylabel("spikes / window (held-out views)"); ax[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig3_face_cells.png", dpi=130); plt.close(fig)

    # 4. learning + safety: weight change and per-trial population rate
    c = res["culture"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    dW = c.weights - c.W0
    v = np.abs(dW).max() or 1
    im = ax[0].imshow(dW, cmap="RdBu_r", vmin=-v, vmax=v); ax[0].set_title("STDP weight change W[post, pre] (mV)")
    ax[0].set_xlabel("pre channel"); ax[0].set_ylabel("post channel"); fig.colorbar(im, ax=ax[0], shrink=.8)
    rates = np.array([l["rate_hz"] for l in c.log]); dis = np.array([l["disrupted"] for l in c.log])
    ax[1].plot(rates, lw=.6, color="k"); ax[1].scatter(np.nonzero(dis)[0], rates[dis], color="r", s=8, label="disrupted (burst / over-driven)")
    ax[1].set_xlabel("trial"); ax[1].set_ylabel("mean rate in window (Hz)"); ax[1].legend(fontsize=7)
    ax[1].set_title(f"network safety: {c.n_disrupted}/{len(c.log)} trials disrupted")
    fig.tight_layout(); fig.savefig(out / "fig4_learning_safety.png", dpi=130); plt.close(fig)
