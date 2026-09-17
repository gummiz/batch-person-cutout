#!/usr/bin/env python3
"""Group cutouts by face, fully local.

YuNet finds the largest face in each cutout, SFace embeds it, and average-linkage
clustering on cosine similarity groups the embeddings. Writes clusters.json,
clusters.html (open locally, type names, download names.json) and names.template.json.
"""
import argparse
import html
import json
import os
import pathlib
import sys

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # silence backend warnings; must precede the cv2 import

import cv2
import numpy as np
from PIL import Image

MODELS = pathlib.Path(__file__).parent.parent / "models"
DET = MODELS / "face_detection_yunet_2023mar.onnx"
REC = MODELS / "face_recognition_sface_2021dec.onnx"


def load_bgr(path, max_side=1024):
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (128, 128, 128, 255))
    im = Image.alpha_composite(bg, im).convert("RGB")
    im.thumbnail((max_side, max_side))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def embed(path, det, rec, min_score):
    img = load_bgr(path)
    det.setInputSize((img.shape[1], img.shape[0]))
    _, faces = det.detect(img)
    if faces is None:
        return None
    faces = faces[faces[:, -1] >= min_score]
    if not len(faces):
        return None
    face = max(faces, key=lambda f: f[2] * f[3])  # largest box
    feat = rec.feature(rec.alignCrop(img, face)).ravel()
    return feat / np.linalg.norm(feat)


def cluster(X, threshold, photos):
    """Average-linkage agglomerative clustering; merge while mean cosine similarity >= threshold.
    Two cutouts from the same photo are never merged: they are different people."""
    groups = [[i] for i in range(len(X))]
    S = X @ X.T
    same = np.array(photos)[:, None] == np.array(photos)[None, :]
    S[same & ~np.eye(len(X), dtype=bool)] = -np.inf
    while len(groups) > 1:
        best, pair = -2, None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                s = S[np.ix_(groups[a], groups[b])].mean()
                if s > best:
                    best, pair = s, (a, b)
        if best < threshold:
            break
        a, b = pair
        groups[a] += groups.pop(b)
    return sorted(groups, key=len, reverse=True)


PAGE = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Name the clusters</title>
<style>
body{{font:16px system-ui,sans-serif;margin:0;padding:1.5rem;background:#1e1e1e;color:#eee}}
section{{margin:0 0 2rem}} h2{{font-size:1rem;margin:0 0 .5rem}}
.row{{display:flex;flex-wrap:wrap;gap:.5rem}} img{{height:160px;background:#555;border-radius:4px}}
label{{display:block;margin:.5rem 0}} input{{font:inherit;padding:.25rem .5rem}}
input:focus,button:focus{{outline:3px solid #6af;outline-offset:2px}}
button{{font:inherit;padding:.5rem 1rem;position:sticky;bottom:1rem}}
</style>
<h1>Name the clusters</h1>
<p>Type a name per group (leave empty to skip), then download <code>names.json</code> and save it next to this page.</p>
{sections}
<button id="dl">Download names.json</button>
<script>
document.getElementById('dl').onclick=()=>{{
  const names={{}};
  document.querySelectorAll('input[data-id]').forEach(i=>{{if(i.value.trim())names[i.dataset.id]=i.value.trim()}});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify(names,null,2)],{{type:'application/json'}}));
  a.download='names.json';a.click();
}};
</script></html>
"""


def section(cid, files, labelled=True):
    imgs = "".join(f'<img src="{html.escape(f)}" alt="{html.escape(pathlib.Path(f).stem)}" loading="lazy">'
                   for f in files)
    field = (f'<label>Name for {cid}: <input data-id="{cid}" autocomplete="off"></label>' if labelled else "")
    return f'<section><h2>{cid} ({len(files)})</h2><div class="row">{imgs}</div>{field}</section>'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=pathlib.Path, help="cutout run folder (e.g. cutouts/v1)")
    ap.add_argument("--threshold", type=float, default=0.363,
                    help="min mean cosine similarity to merge clusters (SFace reference: 0.363)")
    ap.add_argument("--min-score", type=float, default=0.8, help="min face detection score (0.8)")
    o = ap.parse_args()
    if not DET.exists() or not REC.exists():
        sys.exit("face models missing; run scripts/setup.sh")
    run = o.run_dir.resolve()
    files = sorted(p for p in run.rglob("*.png")
                   if not any(part.startswith("_") for part in p.relative_to(run).parts))
    det = cv2.FaceDetectorYN.create(str(DET), "", (320, 320), o.min_score)
    rec = cv2.FaceRecognizerSF.create(str(REC), "")
    feats, keep, noface = [], [], []
    for p in files:
        rel = str(p.relative_to(run))
        e = embed(p, det, rec, o.min_score)
        if e is None:
            noface.append(rel)
        else:
            feats.append(e)
            keep.append(rel)
    photos = [pathlib.Path(k).stem.rsplit("-p", 1)[0] for k in keep]
    groups = cluster(np.array(feats), o.threshold, photos) if feats else []
    clusters = {f"c{k:02d}": [keep[i] for i in g] for k, g in enumerate(groups, 1)}
    (run / "clusters.json").write_text(json.dumps({"clusters": clusters, "noface": noface}, indent=2))
    (run / "names.template.json").write_text(json.dumps({c: "" for c in clusters}, indent=2))
    body = "".join(section(c, fs) for c, fs in clusters.items())
    if noface:
        body += section("no face found", noface, labelled=False)
    (run / "clusters.html").write_text(PAGE.format(sections=body))
    print(f"{len(files)} cutouts: {len(clusters)} clusters "
          f"({', '.join(f'{c}:{len(v)}' for c, v in clusters.items())}), {len(noface)} without face")
    print(f"open {run / 'clusters.html'}")


if __name__ == "__main__":
    main()
