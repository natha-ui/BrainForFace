"""Can a single simulated neuron learn to encode one person's face?

    uv run python experiments/face_cells/run.py --faces path/to/faces

Faces folder: one image per file, named <person>_<anything>.jpg/png (or one
sub-folder per person). With 2+ photos per person the last photo is held out
and never used for training, so it tests invariance to a new picture (as in
Quiroga 2005). With a single photo the held-out views are new augmentations of
the same photo, which is a weaker test.

Protocol (all in one simulated culture, plus a no-plasticity control culture):
  baseline blanks -> PRE probe -> unsupervised STDP exposure -> baseline blanks -> POST probe
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analysis import evaluate  # noqa: E402
from culture import Culture  # noqa: E402
from encode import (ENCODERS, OUTPUT_CHANNELS, EncoderConfig, augment,  # noqa: E402
                    downsample, load_face, norm_code)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_people(folder: Path) -> dict[str, list[Path]]:
    people = defaultdict(list)
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        name = p.parent.name if p.parent != folder else p.stem.split("_")[0]
        people[name].append(p)
    return dict(people)


def build_views(people, n_views, rng):
    """Returns names, train (exposure + selection) views, held-out test views."""
    names = sorted(people)
    train, test = [], []  # lists of (label, img8x4)
    for i, n in enumerate(names):
        photos = [load_face(p) for p in people[n]]
        if len(photos) >= 3:      # every 3rd photo held out: new poses never seen in training
            tr_photos = [p for j, p in enumerate(photos) if j % 3 != 2]
            te_photos = [p for j, p in enumerate(photos) if j % 3 == 2]
        elif len(photos) == 2:
            tr_photos, te_photos = photos[:1], photos[1:]
        else:
            tr_photos = te_photos = photos
        for v in range(n_views):
            train.append((i, augment(tr_photos[v % len(tr_photos)], rng)))
            test.append((i, augment(te_photos[v % len(te_photos)], rng)))
    return names, train, test


def balanced_order(labels, rng):
    """Counterbalanced presentation order: shuffle within each identity, then
    interleave identities in cycles whose order rotates (Latin square), so every
    identity is equally often first and equally often follows each other one."""
    labels = np.asarray(labels)
    ids = np.unique(labels)
    pools = [list(rng.permutation(np.nonzero(labels == i)[0])) for i in ids]
    rows = [list(np.roll(ids, -r)) for r in range(len(ids))]
    rows += [r[::-1] for r in rows] if len(ids) > 2 else []   # both directions -> balanced carry-over
    order, cyc = [], 0
    while any(pools):
        for i in rows[cyc % len(rows)]:
            pool = pools[int(np.nonzero(ids == i)[0][0])]
            if pool:
                order.append(int(pool.pop()))
        cyc += 1
    return np.array(order)


def probe(culture, views, enc, cfg, reps, rng, tag):
    order = [(lab, img) for lab, img in views for _ in range(reps)]
    idx = balanced_order([l for l, _ in order], rng)
    R = np.zeros((len(order), len(OUTPUT_CHANNELS)))
    labs = np.zeros(len(order), dtype=int)
    for j, k in enumerate(idx):
        lab, img = order[k]
        R[j] = culture.trial(enc(img, cfg), tag=tag)[OUTPUT_CHANNELS]
        labs[j] = lab
    return R, labs


def blanks(culture, n, tag):
    return np.array([culture.trial([], tag=tag)[OUTPUT_CHANNELS] for _ in range(n)])


def run_culture(plastic, inhibition, args, names, train, test, enc, cfg):
    rng = np.random.default_rng(args.seed + 1)
    c = Culture(seed=args.seed, coupling_gain_mV=args.coupling, stdp=plastic, lr=args.lr,
                inhibition_mV=inhibition)
    c.run(25_000)  # 1 s settle
    out = {}
    for phase in ("pre", "post"):
        if phase == "post":
            for ep in range(args.epochs):
                for k in balanced_order([l for l, _ in train], rng):
                    c.trial(enc(train[k][1], cfg), learn=plastic, tag=f"train{ep}")
        B = blanks(c, args.n_blank, f"blank_{phase}")
        Rs, ls = probe(c, train, enc, cfg, args.reps, rng, f"{phase}_sel")
        Rt, lt = probe(c, test, enc, cfg, args.reps, rng, f"{phase}_test")
        ev = evaluate(Rs, ls, Rt, lt, B, len(names), OUTPUT_CHANNELS, k=args.k, n_perm=args.n_perm)
        out[phase] = dict(eval=ev, R_test=Rt, lab_test=lt, B=B)
    out["culture"] = c
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--faces", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("face_cell_results"))
    ap.add_argument("--encoding", choices=list(ENCODERS), default="spatial")
    ap.add_argument("--code", choices=["norm", "raw"], default="norm",
                    help="norm = deviation from the mean training face (norm-based coding); raw = pixels")
    ap.add_argument("--charge", type=float, default=0.0,
                    help="equalised total code per face (sum of 0-1 pixel drive); 0 = off (max-normalised)")
    ap.add_argument("--epochs", type=int, default=15, help="exposure passes over the training views")
    ap.add_argument("--views", type=int, default=12, help="augmented views per person per set")
    ap.add_argument("--reps", type=int, default=2, help="repeats of each probe view")
    ap.add_argument("--n-blank", type=int, default=40)
    ap.add_argument("--k", type=float, default=5.0, help="responsiveness threshold in baseline SDs")
    ap.add_argument("--coupling", type=float, default=7.0)
    ap.add_argument("--inhibition", type=float, default=4.0, help="lateral inhibition between output cells (mV per spike)")
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--skip-noinh", action="store_true", help="skip the STDP-without-inhibition control")
    args = ap.parse_args()

    t0 = time.time()
    people = load_people(args.faces)
    if len(people) < 2:
        sys.exit(f"need at least 2 people in {args.faces}, found {list(people)}")
    rng = np.random.default_rng(args.seed)
    names, train, test = build_views(people, args.views, rng)
    enc, cfg = ENCODERS[args.encoding], EncoderConfig()
    mean_face = np.mean([im for _, im in train], axis=0)  # training photos only
    ts = args.charge if args.charge > 0 else None
    xf = (lambda im: norm_code(im, mean_face, ts)) if args.code == "norm" else (lambda im: im)
    train = [(l, xf(im)) for l, im in train]
    test = [(l, xf(im)) for l, im in test]
    X = np.array([im.ravel() for _, im in train]); labs = np.array([l for l, _ in train])
    C = np.corrcoef(X)
    same = C[(labs[:, None] == labs[None]) & ~np.eye(len(labs), dtype=bool)].mean()
    diff = C[labs[:, None] != labs[None]].mean()
    print(f"input correlation ({args.code}): same person {same:.2f}, different people {diff:.2f}")
    for i in range(int(labs.max()) + 1):
        print(f"  person {i}: mean pulses/trial {np.mean([len(enc(im, cfg)) for l, im in train if l == i]):.0f}")
    print(f"{len(names)} people: " + ", ".join(f"{n} ({len(people[n])} photos)" for n in names))

    results = {"STDP": run_culture(True, args.inhibition, args, names, train, test, enc, cfg)}
    if not args.no_control:
        results["control: no plasticity"] = run_culture(False, args.inhibition, args, names, train, test, enc, cfg)
        if args.inhibition > 0 and not args.skip_noinh:
            results["control: STDP, no inhibition"] = run_culture(True, 0.0, args, names, train, test, enc, cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = dict(config={k: str(v) for k, v in vars(args).items()}, people=names, runs={})
    for run, res in results.items():
        c = res["culture"]
        summary["runs"][run] = dict(
            pre=res["pre"]["eval"], post=res["post"]["eval"],
            disrupted_trials=c.n_disrupted, total_trials=len(c.log), final_stim_scale=c.stim_scale,
        )
    stdp = results["STDP"]["culture"]
    np.savez(args.out / "trained_culture.npz", W=stdp.weights, W0=stdp.W0, seed=args.seed,
             coupling=args.coupling, encoding=args.encoding, code=args.code, mean_face=mean_face, charge=args.charge, inhibition=args.inhibition, people=np.array(names),
             face_cells=np.array([d["cell"] for d in summary["runs"]["STDP"]["post"]["per_identity"]]))
    (args.out / "results.json").write_text(json.dumps(summary, indent=1))

    from figures import make_figures
    make_figures(args.out, names, people, xf, results, enc, cfg)

    # ---- console report
    print(f"\nencoding={args.encoding}  code={args.code}  inhibition={args.inhibition} mV  epochs={args.epochs}  k={args.k} SD  ({time.time() - t0:.0f} s)")
    for run, s in summary["runs"].items():
        print(f"\n[{run}]  disrupted {s['disrupted_trials']}/{s['total_trials']} trials, stim scale {s['final_stim_scale']:.2f}")
        for phase in ("pre", "post"):
            ev = s[phase]
            print(f"  {phase.upper():4s} single-identity cells: {ev['n_cells_single_identity']:2d}/30, "
                  f"multi: {ev['n_cells_multi_identity']:2d}, mean sparseness {ev['mean_sparseness']:.2f}")
            for d in ev["per_identity"]:
                print(f"     {names[d['identity']]:>12s} -> ch{d['cell']:<3d} {'FACE CELL' if d['is_face_cell'] else '         '} "
                      f"pref {d['resp_pref']:.2f} vs other-max {d['resp_other_max']:.2f} (thr {d['threshold']:.2f}) "
                      f"z {d['z']:4.1f} SI {d['SI']:+.2f} AUC {d['AUC']:.2f} perm-p {d['perm_p_population']:.3f}")
    print(f"\nwrote {args.out}/")


if __name__ == "__main__":
    main()
