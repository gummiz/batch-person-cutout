---
name: batch-person-cutout
description: Batch background removal for photos of people, done entirely on the local machine. Cuts every person out of a folder of photos into transparent RGBA PNGs, runs automatic QA, groups the cutouts by face and sorts them into one folder per person. Use this whenever the user wants cutouts, background removal, "Freisteller", "freistellen", isolated or masked people, transparent PNGs of people, or wants to sort event/shoot photos by person, even if they only say "cut out the people in this folder" or "remove the backgrounds from these shots". Prefer it over cloud tools whenever privacy matters or many photos are involved.
---

# batch-person-cutout

Cuts people out of a photo folder into transparent PNGs, locally, and sorts them by person.
Everything lives in this skill folder; `SKILL_DIR` below means the folder containing this file.

## Privacy rule (read first)

All processing runs on this machine. **Never** send photos, crops, cutouts, masks or contact
sheets to an LLM or any cloud service, and don't open them with an image-viewing tool, unless the user
has explicitly agreed for this specific job. Many projects forbid putting real people into AI tools,
and a cutout is still a photo of a person.

The consequence: you do not decide who is who. The face clustering step groups the cutouts, and the
user names the groups in a local HTML page. You work only with file names, counts and CSV/JSON.
If a project CLAUDE.md contains a stricter rule, it wins.

## Setup

Run once per machine (idempotent, safe to re-run):

```bash
bash SKILL_DIR/scripts/setup.sh              # venv, rembg, OpenCV, Vision helper (macOS), face models
bash SKILL_DIR/scripts/setup.sh --with-torch # additionally torch/torchvision on macOS
```

On Linux and Windows (WSL/Git Bash) setup always installs CPU torch, because Apple Vision is not
available there. The first cutout run downloads the BiRefNet model (~1 GB) to `~/.rembg`.
Use `PY=SKILL_DIR/.venv/bin/python` for all scripts below.

## Workflow

Tell the user the plan (input folder, output folder, rough runtime: ~15-25 s per person on a CPU)
before starting a long run. Source photos are never modified; outputs are versioned.

1. **Cut out** (auto mode: every person at least `--min-height` px tall):
   ```bash
   $PY SKILL_DIR/scripts/cutout.py <photos> [<out_base>]
   ```
   Output goes to `<out_base>/vN/unsorted/<photo>-p<N>.png` (N counts people left to right),
   default `<out_base>` is `<photos>/../cutouts`. A new `vN` is created on every run, so nothing is
   overwritten. For long runs, start in the background and test first with `--limit 2`.
2. **QA review.** Read `vN/report.csv`. Rows with a reason other than `ok` were moved to
   `vN/_rejected/` (`empty`, `tiny` = under 2 % of the crop, `wrong_person` = under 50 % overlap with
   the chosen instance; `no_person` = photo without people). Report the counts. Build contact sheets
   so the user can check visually:
   ```bash
   $PY SKILL_DIR/scripts/contact_sheet.py <out_base>/vN
   ```
3. **Cluster faces:**
   ```bash
   $PY SKILL_DIR/scripts/cluster_faces.py <out_base>/vN [--threshold 0.363]
   ```
   Writes `clusters.json`, `names.template.json` and `clusters.html`. Report the cluster sizes.
   If one person is split across clusters, that's fine (the user gives both the same name). If
   different people are merged, rerun with a higher threshold (e.g. 0.45).
4. **User names the clusters.** Ask the user to open `clusters.html` in a browser, type a name
   per group and save the downloaded `names.json` into `vN/`. Wait for them; don't guess names.
   Alternatively they can fill in `names.template.json` and save it as `names.json`.
5. **Rename.** Show the dry run first, then apply after the user agrees:
   ```bash
   $PY SKILL_DIR/scripts/apply_names.py <out_base>/vN            # prints the plan
   $PY SKILL_DIR/scripts/apply_names.py <out_base>/vN --apply
   ```
   Result: `vN/<name>/<name>-<photo>-p<N>.png`; unnamed or faceless cutouts go to `vN/_unassigned/`.
   Existing targets are never overwritten. Finish with a fresh contact sheet per folder.

**Known names up front:** if the user already knows who stands where, skip steps 3-5 with a
selection file (one line per person: `<photo-stem> <name> <C|L|R>`, C = largest, L = leftmost,
R = rightmost of the large people):
```bash
$PY SKILL_DIR/scripts/cutout.py <photos> --select selection.txt   # -> vN/<name>/<name>-<photo>.png
```
Only the user writes this file, since filling it in means looking at the photos.

## Options (cutout.py)

| Option | Default | Use |
|---|---|---|
| `--min-height` / `--max-height` | 800 / off | skip background people / skip huge ones |
| `--margin-x` / `--margin-y` | .15 / .08 | crop padding so hands, props and hair stay in |
| `--model` | birefnet-general | `birefnet-portrait` for close-ups; any rembg model name works |
| `--backend` | auto | `vision` (macOS) or `torchvision` (Mask R-CNN, any OS) |
| `--provider` | auto (= cpu) | `coreml` failed with BiRefNet in testing; see troubleshooting |
| `--workers` | 1 | parallel processes; gave no speed-up on one machine in testing, each loads a ~1 GB model |
| `--close PX` | off | morphological closing; fills small notches in the alpha |
| `--fade-bottom FRAC` | off | fades out the lower edge where tables or chairs cut in |
| `--limit N`, `--only STEM...` | | test runs and reruns |

Bottom-edge artefacts (tables, cans, chair backs biting into the body) are the most common flaw.
Ask the user before using `--fade-bottom`; some want hard edges and fix them by hand.

## Object mode

Not part of this skill. It handles people only; cutting out products or props is planned as a
separate project. Say so if asked, and don't improvise it.

## References

- `references/pipeline.md`: every step and why it works this way. Read it before changing the code.
- `references/dead-ends.md`: approaches that were tried and failed. Read it before "improving" anything.
- `references/troubleshooting.md`: install problems, slow runs, bad cutouts, clustering issues.
