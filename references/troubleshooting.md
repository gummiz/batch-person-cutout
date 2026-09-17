# Troubleshooting

## Setup

- **`swiftc: command not found`**: install the Xcode command line tools (`xcode-select --install`),
  or skip Vision and run `setup.sh --with-torch`, then use `--backend torchvision`.
- **Vision helper fails or returns nothing**: `VNGeneratePersonInstanceMaskRequest` needs macOS 14+.
  Use the torchvision backend on older systems.
- **No wheels for your Python** (torch or onnxruntime lag behind new Python releases): delete `.venv`
  and recreate it with an older interpreter, e.g. `uv venv --python 3.12 .venv`, then rerun setup.
- **Face model download fails** (proxy, offline): fetch the two ONNX files from
  `github.com/opencv/opencv_zoo` (`models/face_detection_yunet`, `models/face_recognition_sface`) and
  put them into `models/`. Files under 100 kB are Git LFS pointers, not models.
- **BiRefNet download**: about 1 GB, stored in `~/.rembg/models/` (or `$U2NET_HOME`). Copy it from
  another machine if the download is blocked.

## Speed

Reference (Apple M2 Max, CPU provider): 12-23 s per person, depending on crop size, plus
about 5 s per photo for loading and instance masks. Six photos with ten people took 193 s.

- **CoreML provider**: `--provider coreml` did not work for `birefnet-general`. The Neural Engine
  compiler ran for about 14 minutes and the run then failed with
  `Unable to compute the prediction using a neural network model`. With `MLComputeUnits=CPUAndGPU`
  the session didn't finish loading within 8 minutes. That is why `auto` means CPU. Other rembg models may behave differently.
- **`--workers`**: onnxruntime already uses all cores, so two workers on one machine brought no
  throughput gain in testing (each person took about twice as long). Each worker also holds its own
  ~1 GB model. Use more than one only on machines with many cores and plenty of RAM.
- **torchvision backend**: loading Mask R-CNN and running it on a 24 MP photo adds roughly 20-30 s per
  photo on CPU; the weights (~170 MB) download to `~/.cache/torch` on first use.
- Always try `--limit 2` first, then start the full run in the background.

## Bad cutouts

| Symptom | Try |
|---|---|
| Notches or dark blocks at the bottom (table, cans, chair backs) | `--close 9` to `--close 21`; `--model birefnet-portrait`; `--fade-bottom .08` only if the user agrees |
| Hands or props cut off at the side | larger `--margin-x`, e.g. `.25` |
| Neighbour's shoulder remains | expected at the edges of the dilated mask; retouch by hand or crop tighter |
| Background people cut out | raise `--min-height` |
| Many `tiny` rejects | usually people partly hidden or at the image edge; check `_rejected/` contact sheet |
| `wrong_person` rejects | overlapping people; rerun those photos with `--select` and `L`/`R` |
| `no_person` rows | the backend found nobody; try the other backend |
| Everything looks shifted | orientation mismatch; the script raises an error on size mismatch, report it |

`RuntimeWarning: overflow encountered in exp` from rembg is harmless.

## Clustering

- **One person split into several clusters**: normal with profile shots or sunglasses. The user gives
  them the same name, and `apply_names.py` merges them into one folder.
- **Different people in one cluster**: raise `--threshold` (0.40-0.50). Two cutouts from the same
  photo are never merged, whatever the threshold.
- **Many `no face found`**: people seen from behind or faces smaller than about 20 px after the
  1024 px downscale. Lower `--min-score` to 0.6, or let the user sort them by hand from `_unassigned/`.
- **Thumbnails missing in `clusters.html`**: the page uses relative links. Keep it inside the run
  folder, and don't run `apply_names.py --apply` before naming is done.
