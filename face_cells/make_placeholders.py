"""Procedural placeholder 'identities' for smoke-testing only (not real faces)."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

out = Path(sys.argv[1] if len(sys.argv) > 1 else "placeholder_faces"); out.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
for pid in range(4):
    p = dict(eye_y=rng.uniform(.36,.46), eye_dx=rng.uniform(.12,.22), eye_r=rng.uniform(.03,.07),
             mouth_y=rng.uniform(.68,.78), mouth_w=rng.uniform(.12,.3), hair=rng.uniform(.1,.35),
             skin=int(rng.uniform(150,230)), brow=rng.random()>.5, beard=rng.random()>.5)
    for v in range(2):
        W, H = 200, 260; im = Image.new("L", (W, H), int(rng.uniform(30,90))); d = ImageDraw.Draw(im)
        j = lambda: rng.normal(0, 3)
        d.ellipse([30+j(), 20+j(), 170+j(), 250+j()], fill=p["skin"])
        d.rectangle([30, 10, 170, 20 + p["hair"]*H], fill=40)
        for s in (-1, 1):
            cx, cy, r = W/2 + s*p["eye_dx"]*W + j(), p["eye_y"]*H + j(), p["eye_r"]*W
            d.ellipse([cx-r, cy-r*.7, cx+r, cy+r*.7], fill=25)
            if p["brow"]: d.rectangle([cx-r*1.3, cy-r*1.8, cx+r*1.3, cy-r*1.2], fill=30)
        my = p["mouth_y"]*H + j(); mw = p["mouth_w"]*W
        d.rectangle([W/2-mw, my-4, W/2+mw, my+4], fill=60)
        if p["beard"]: d.pieslice([50, 150, 150, 260], 0, 180, fill=50)
        im.filter(ImageFilter.GaussianBlur(2)).save(out / f"person{pid}_{v}.png")
print("wrote", len(list(out.glob('*.png'))), "images to", out)
