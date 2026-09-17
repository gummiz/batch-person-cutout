#!/usr/bin/env python3
"""Batch person cutouts, fully local.

Pipeline per person: instance mask (who) -> padded crop -> BiRefNet matte (edges)
-> limit to dilated instance mask -> subtract other people -> trim -> RGBA PNG.
See references/pipeline.md for the reasoning behind each step.
"""
import argparse
import csv
import pathlib
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageFilter, ImageOps

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from instances import person_masks  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".webp"}
_session = None
_opts = None


def providers_for(name):
    if name == "auto":
        # CoreML was not faster for BiRefNet in testing; keep CPU unless asked
        name = "cpu"
    if name == "coreml":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def init_worker(opts):
    global _session, _opts
    from rembg import new_session
    _opts = opts
    _session = new_session(opts.model, providers=providers_for(opts.provider))


def next_version(base):
    base.mkdir(parents=True, exist_ok=True)
    nums = [int(m.group(1)) for p in base.iterdir() if (m := re.fullmatch(r"v(\d+)", p.name))]
    run = base / f"v{max(nums, default=0) + 1}"
    run.mkdir()
    return run


def to_img(mask):
    return Image.fromarray(mask.astype(np.uint8) * 255)


def cut_one(img, inst, pick, o):
    """Return (rgba, stats) for instance `pick` of `inst`."""
    from rembg import remove
    ys, xs = np.nonzero(pick)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    W, H = img.size
    mx, my = int((x1 - x0) * o.margin_x), int((y1 - y0) * o.margin_y)
    box = (max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my))
    crop = img.crop(box)
    alpha = np.array(remove(crop, session=_session, only_mask=True), dtype=np.float32) / 255

    # Limit to the dilated instance mask. Dilate at 1/8 scale: MaxFilter cost grows with radius^2.
    lim = to_img(pick).crop(box)
    small = lim.resize((max(1, lim.width // 8), max(1, lim.height // 8)))
    r = max(3, int(small.height * .07)) | 1
    small = small.filter(ImageFilter.MaxFilter(r)).filter(ImageFilter.GaussianBlur(r / 2))
    alpha *= np.array(small.resize(lim.size, Image.BILINEAR), dtype=np.float32) / 255

    # Subtract other people (a neighbour's arm in front of the body)
    for m in inst:
        if m is pick:
            continue
        om = to_img(m).crop(box).filter(ImageFilter.GaussianBlur(4))
        alpha *= 1 - np.array(om, dtype=np.float32) / 255

    if o.close:
        k = o.close | 1
        a = Image.fromarray((alpha * 255).astype(np.uint8))
        alpha = np.array(a.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MinFilter(k)), dtype=np.float32) / 255

    solid = alpha > .5
    pick_crop = np.array(lim) > 127
    stats = {
        "visible": solid.mean(),
        "overlap": (solid & pick_crop).sum() / max(1, solid.sum()),
    }
    a8 = Image.fromarray((alpha * 255).astype(np.uint8))
    rgba = crop.convert("RGBA")
    rgba.putalpha(a8)
    bbox = a8.getbbox()
    if bbox is None:
        return None, stats
    rgba = rgba.crop(bbox)
    if o.fade_bottom:
        h = rgba.height
        n = max(1, int(h * o.fade_bottom))
        ramp = np.ones(h, np.float32)
        ramp[h - n:] = np.linspace(1, 0, n)
        a = np.array(rgba.getchannel("A"), np.float32) * ramp[:, None]
        rgba.putalpha(Image.fromarray(a.astype(np.uint8)))
    return rgba, stats


def choose(inst, pos):
    areas = [m.sum() for m in inst]
    big = [m for m, a in zip(inst, areas) if a > .3 * max(areas)]
    cx = lambda m: np.nonzero(m)[1].mean()  # noqa: E731
    if pos == "C":
        return max(big, key=lambda m: m.sum())
    return (min if pos == "L" else max)(big, key=cx)


def height(m):
    ys = np.nonzero(m.any(axis=1))[0]
    return ys[-1] - ys[0] + 1


def process(job):
    """job = (src, [(name, pos), ...]) or (src, None) for auto mode. Returns report rows."""
    src, picks = job
    o = _opts
    rows = []
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")  # in memory only
    inst = person_masks(src, img, o.backend, o.mask_cache)
    if not inst:
        return [dict(file=src.name, reason="no_person", size="", seconds=0)]
    if picks is None:
        todo = []
        for n, m in enumerate(sorted(inst, key=lambda m: np.nonzero(m)[1].mean()), 1):
            h = height(m)
            if h >= o.min_height and (not o.max_height or h <= o.max_height):
                todo.append((m, pathlib.Path("unsorted") / f"{src.stem}-p{n}.png"))
    else:
        todo = [(choose(inst, pos), pathlib.Path(name) / f"{name}-{src.stem}.png") for name, pos in picks]
    for m, rel in todo:
        t = time.time()
        rgba, st = cut_one(img, inst, m, o)
        reason = "ok"
        if rgba is None:
            reason = "empty"
        elif st["visible"] < .02:
            reason = "tiny"
        elif st["overlap"] < .5:
            reason = "wrong_person"
        out = o.run / (rel if reason == "ok" else pathlib.Path("_rejected") / f"{reason}-{rel.name}")
        if rgba is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            rgba.save(out, optimize=True)
        dt = round(time.time() - t, 1)
        size = f"{rgba.width}x{rgba.height}" if rgba else ""
        print(f"{reason:12} {out.relative_to(o.run)} {size} {dt}s", flush=True)
        rows.append(dict(file=str(out.relative_to(o.run)), reason=reason, size=size, seconds=dt))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=pathlib.Path, help="folder with photos (never modified)")
    ap.add_argument("output_dir", type=pathlib.Path, nargs="?",
                    help="base output folder (default: <input>/../cutouts); each run goes to a new vN subfolder")
    ap.add_argument("--select", type=pathlib.Path,
                    help="selection file, lines '<photo-stem> <name> <C|L|R>' (C=largest, L=leftmost, R=rightmost)")
    ap.add_argument("--min-height", type=int, default=800, help="auto mode: minimum person height in px (800)")
    ap.add_argument("--max-height", type=int, default=0, help="auto mode: maximum person height in px (off)")
    ap.add_argument("--margin-x", type=float, default=.15, help="horizontal crop margin, fraction of bbox width (.15)")
    ap.add_argument("--margin-y", type=float, default=.08, help="vertical crop margin, fraction of bbox height (.08)")
    ap.add_argument("--model", default="birefnet-general",
                    help="rembg model, e.g. birefnet-general, birefnet-portrait (birefnet-general)")
    ap.add_argument("--backend", default="auto", choices=["auto", "vision", "torchvision"],
                    help="person instance backend (auto: vision on macOS, else torchvision)")
    ap.add_argument("--provider", default="auto", choices=["auto", "cpu", "coreml"],
                    help="onnxruntime provider for the matting model (auto = cpu)")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes, one model per process (1)")
    ap.add_argument("--close", type=int, default=0, metavar="PX", help="morphological closing of alpha (off)")
    ap.add_argument("--fade-bottom", type=float, default=0, metavar="FRAC",
                    help="fade out the bottom fraction of each cutout (off)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N photos")
    ap.add_argument("--only", nargs="+", default=[], metavar="STEM", help="process only these photo stems")
    o = ap.parse_args()

    src_dir = o.input_dir.resolve()
    photos = {p.stem: p for p in sorted(src_dir.iterdir()) if p.suffix.lower() in EXTS}
    if o.select:
        sel = {}
        for line in o.select.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                stem, name, pos = line.split()
                if stem not in photos:
                    sys.exit(f"--select: photo not found: {stem}")
                sel.setdefault(stem, []).append((name, pos.upper()))
        jobs = [(photos[s], p) for s, p in sel.items()]
    else:
        jobs = [(p, None) for p in photos.values()]
    if o.only:
        jobs = [j for j in jobs if j[0].stem in o.only]
    if o.limit:
        jobs = jobs[:o.limit]
    if not jobs:
        sys.exit("nothing to do")

    base = (o.output_dir or src_dir.parent / "cutouts").resolve()
    if base == src_dir:
        sys.exit("output_dir must not be the input folder")
    o.run = next_version(base)
    o.mask_cache = base / "_masks"
    print(f"{len(jobs)} photos -> {o.run}", flush=True)

    t0 = time.time()
    rows = []
    if o.workers <= 1:
        init_worker(o)
        for j in jobs:
            rows += process(j)
    else:
        with ProcessPoolExecutor(o.workers, initializer=init_worker, initargs=(o,)) as ex:
            for r in ex.map(process, jobs):
                rows += r
    with open(o.run / "report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["file", "reason", "size", "seconds"])
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for r in rows:
        counts[r["reason"]] = counts.get(r["reason"], 0) + 1
    print(f"done in {time.time() - t0:.0f}s: {counts} -> {o.run}")


if __name__ == "__main__":
    main()
