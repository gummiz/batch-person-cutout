#!/usr/bin/env python3
"""Contact sheet (JPG) per subfolder of cutouts, with file names under each thumbnail."""
import argparse
import pathlib

from PIL import Image, ImageDraw


def sheet(files, out, size, cols):
    rows = (len(files) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * size, rows * (size + 24)), (48, 48, 48))
    draw = ImageDraw.Draw(canvas)
    for k, f in enumerate(files):
        im = Image.open(f).convert("RGBA")
        im.thumbnail((size, size))
        x, y = k % cols * size, k // cols * (size + 24)
        canvas.paste(im, (x + (size - im.width) // 2, y), im)
        draw.text((x + 4, y + size + 4), f.stem[:40], fill=(255, 255, 0))
    canvas.save(out, quality=85)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=pathlib.Path, help="cutout run folder")
    ap.add_argument("--out", type=pathlib.Path, help="output folder (default: <run_dir>/_sheets)")
    ap.add_argument("--size", type=int, default=360, help="thumbnail size in px (360)")
    ap.add_argument("--cols", type=int, default=6, help="columns (6)")
    o = ap.parse_args()
    run = o.run_dir.resolve()
    out = (o.out or run / "_sheets").resolve()
    out.mkdir(parents=True, exist_ok=True)
    for d in sorted(p for p in run.iterdir() if p.is_dir() and p != out and p.name != "_sheets"):
        files = sorted(d.glob("*.png"))
        if files:
            dest = out / f"contact-{d.name}.jpg"
            sheet(files, dest, o.size, o.cols)
            print(f"{dest} ({len(files)})")


if __name__ == "__main__":
    main()
