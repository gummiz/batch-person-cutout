#!/usr/bin/env python3
"""Move clustered cutouts into <name>/<name>-<stem>.png using names.json.

Dry run by default; pass --apply to move files. Files in unnamed clusters or
without a detected face go to _unassigned/.
"""
import argparse
import json
import pathlib
import re
import sys


def safe(name):
    return re.sub(r"[^\w.-]+", "-", name.strip().lower()).strip("-.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=pathlib.Path, help="cutout run folder with clusters.json")
    ap.add_argument("--names", type=pathlib.Path, help="names file (default: <run_dir>/names.json)")
    ap.add_argument("--apply", action="store_true", help="actually move files (default: dry run)")
    o = ap.parse_args()
    run = o.run_dir.resolve()
    data = json.loads((run / "clusters.json").read_text())
    names = json.loads((o.names or run / "names.json").read_text())
    plan = []
    for cid, files in data["clusters"].items():
        name = safe(names.get(cid, ""))
        for f in files:
            stem = pathlib.Path(f).stem
            plan.append((f, f"{name}/{name}-{stem}.png" if name else f"_unassigned/{stem}.png"))
    plan += [(f, f"_unassigned/{pathlib.Path(f).name}") for f in data["noface"]]

    targets = [t for _, t in plan]
    clash = {t for t in targets if targets.count(t) > 1 or (run / t).exists()}
    for src, dst in plan:
        if (run / src).exists():
            print(f"{src} -> {dst}{'  [EXISTS, skipped]' if dst in clash else ''}")
        else:
            print(f"{src} missing, skipped")
    if not o.apply:
        print("dry run; pass --apply to move files")
        return
    for src, dst in plan:
        s, d = run / src, run / dst
        if s.exists() and dst not in clash:
            d.parent.mkdir(parents=True, exist_ok=True)
            s.rename(d)
    for d in [run / "unsorted"]:
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    print("moved" if not clash else f"moved; {len(clash)} skipped because target exists", file=sys.stderr)


if __name__ == "__main__":
    main()
