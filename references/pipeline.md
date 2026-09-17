# Pipeline

`cutout.py` handles each person in four stages: find the person, crop, matte, clean up.
Each stage makes up for a weakness in the one before it.

## 1. Who: person instance masks

`instances.py` returns one coarse boolean mask per person, at full image size.

- **vision** (macOS 14+): `scripts/bin/inst`, compiled from `inst.swift`, runs Apple Vision
  `VNGeneratePersonInstanceMaskRequest` and writes one PNG per instance. Masks are cached in
  `<out_base>/_masks/` (plus a `.done` marker), so reruns skip this step.
- **torchvision** (any OS): Mask R-CNN `maskrcnn_resnet50_fpn_v2`, COCO class 1 (person), score > 0.5.
  Slower and heavier, but free of platform lock-in and BSD-licensed.

These masks are good at telling people apart and poor at edges. The pipeline uses them only to decide
*who* and *where*, never as the final alpha.

The photo is opened with `ImageOps.exif_transpose` in memory. Vision applies the same orientation,
so masks and pixels line up; a size mismatch raises an error instead of producing shifted cutouts.

## 2. Which person

- Auto mode: every instance whose bounding-box height is at least `--min-height` (800 px) and at most
  `--max-height`. People are numbered left to right by centroid (`-p1`, `-p2`, ...).
- Select mode: only instances larger than 30 % of the largest count; `C` picks the largest,
  `L`/`R` the leftmost/rightmost centroid. The 30 % filter keeps background people from being "leftmost".

## 3. Crop

Bounding box plus 15 % horizontally and 8 % vertically. Hands, held objects and hair often stick out
of the coarse mask, and the matting model works best when it sees exactly one main subject at high
resolution. Cropping first gives BiRefNet far more pixels per person than a full group shot.

## 4. Edges: BiRefNet

`rembg.remove(crop, session, only_mask=True)` with `birefnet-general`. It is strong on hair and keeps
held objects (cans, fans, cards) fully opaque. One session is created per worker process and reused.

## 5. Limit to the chosen person

BiRefNet happily includes neighbours that reach into the crop. The instance mask is therefore:

1. cropped to the same box and **downscaled to 1/8**,
2. dilated with `MaxFilter` (radius 7 % of the small height, odd),
3. blurred with `GaussianBlur(r/2)`,
4. upscaled bilinearly and multiplied into the alpha.

The 1/8 scale matters: `MaxFilter` cost grows with the square of the radius, so doing this at full
resolution takes minutes per image. The dilation leaves room for props and hair outside the coarse mask.

## 6. Subtract other people

For every other instance: `alpha *= 1 - blur4(mask)`. This removes a neighbour's arm lying in front
of the chosen body, which step 5 alone can't catch because it lies inside the dilated area.

## 7. Optional clean-up, trim, save

- `--close PX`: grey-scale closing (MaxFilter then MinFilter) to fill small notches.
- `--fade-bottom FRAC`: linear fade over the lowest fraction of the trimmed cutout.
- Trim to `getbbox()` and save an optimized RGBA PNG.

## 8. Auto-QA

Computed on the crop before trimming, with `solid = alpha > 0.5`:

| Reason | Rule |
|---|---|
| `empty` | alpha is zero everywhere |
| `tiny` | solid area < 2 % of the crop |
| `wrong_person` | < 50 % of the solid area lies inside the chosen instance mask |

Rejected files go to `_rejected/<reason>-<name>.png` so they can still be inspected. Every row lands
in `report.csv` (file, reason, size, seconds).

## 9. Sorting by person

`cluster_faces.py` composites each cutout on grey, downsizes it to 1024 px, detects faces with YuNet
and keeps the largest, then aligns and embeds it with SFace (128-d, L2-normalised). Average-linkage
agglomerative clustering merges groups while their mean cosine similarity is at least `--threshold`
(0.363, the SFace reference value). The user names clusters in `clusters.html`; `apply_names.py`
moves the files. Both face models come from OpenCV Zoo (Apache-2.0) and run on CPU.
