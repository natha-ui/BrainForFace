"""One-click pipeline: (crop) -> experiment -> figures + poster -> example queries.

    python run_all.py                    # cropped faces in data/faces -> results/  (~4-5 min)
    python run_all.py --raw data/raw     # crop raw photos first (data/raw/<person>/*.jpg)
    python run_all.py --quick            # ~1 min smoke test (fewer epochs/views)
    python run_all.py --query photo.jpg  # only ask the trained culture who this is
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FC = ROOT / "face_cells"


def step(title, *args):
    print(f"\n=== {title}", flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, help="raw photos, one sub-folder per person; cropped into --faces")
    ap.add_argument("--faces", type=Path, default=ROOT / "data" / "faces")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--query", type=Path, help="skip the experiment; query this photo against --out")
    ap.add_argument("extra", nargs="*", help="extra flags passed to run.py, after --  (e.g. -- --inhibition 6)")
    a = ap.parse_args()

    if a.query:
        step("query", FC / "query.py", "--culture", a.out / "trained_culture.npz", a.query)
        return
    if a.raw:
        step("crop faces", FC / "crop_faces.py", a.raw, a.faces)
    quick = ["--epochs", "3", "--views", "4", "--n-perm", "100", "--skip-noinh"] if a.quick else []
    step("experiment (baseline -> pre-probe -> STDP exposure -> post-probe, with controls)",
         FC / "run.py", "--faces", a.faces, "--out", a.out, *quick, *a.extra)
    step("poster", FC / "visualise.py", "--results", a.out, "--faces", a.faces)
    # demo: query one held-out photo (every 3rd photo is held out) per person
    seen = set()
    for f in sorted(a.faces.glob("*_*.png")):
        person, idx = f.stem.rsplit("_", 1)
        if person not in seen and idx.isdigit() and int(idx) % 3 == 2:
            seen.add(person)
            step(f"query held-out photo {f.name}", FC / "query.py", "--culture", a.out / "trained_culture.npz", f)
    print(f"\nDone. Open {a.out / 'face_cells_poster.png'} and {a.out / 'console.txt' if (a.out / 'console.txt').exists() else a.out}")


if __name__ == "__main__":
    main()
