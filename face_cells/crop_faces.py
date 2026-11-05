"""Crop faces from raw photos.

    python face_cells/crop_faces.py data/raw data/faces

data/raw/<person>/*.jpg  ->  data/faces/<person>_NN.png (160x200 grayscale head crops)

Uses OpenCV's Haar frontal-face detector (opencv-python-headless < 5; v5 removed
the Haar cascades). If no face is found (head tilted up/down), falls back to a
fixed head-and-shoulders region in the upper middle of the frame.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

EXT = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp"}


def main(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    people = sorted(p for p in src.iterdir() if p.is_dir())
    if not people:
        sys.exit(f"put one sub-folder per person in {src} (e.g. {src}/alice/*.jpg)")
    for person in people:
        files = sorted(f for f in person.iterdir() if f.suffix.lower() in EXT)
        n_det = 0
        for k, f in enumerate(files):
            im = np.asarray(ImageOps.exif_transpose(Image.open(f)).convert("L"))
            s = 600 / im.shape[0]
            det = casc.detectMultiScale(cv2.resize(im, None, fx=s, fy=s), 1.1, 5, minSize=(80, 80))
            if len(det):
                x, y, w, h = max(det, key=lambda d: d[2] * d[3]) / s
                cx, cy, side = x + w / 2, y + h / 2, h * 1.5
                n_det += 1
            else:
                H, W = im.shape
                cx, cy, side = W / 2, H * 0.33, H * 0.42
            box = (int(cx - side * .4), int(cy - side * .5), int(cx + side * .4), int(cy + side * .5))
            Image.fromarray(im).crop(box).resize((160, 200)).save(dst / f"{person.name}_{k:02d}.png")
        print(f"{person.name}: {len(files)} photos, face detected in {n_det}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
