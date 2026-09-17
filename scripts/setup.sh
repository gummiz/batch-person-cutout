#!/usr/bin/env bash
# Sets up the batch-cutout environment inside the skill folder. Safe to re-run.
#   scripts/setup.sh               default install
#   scripts/setup.sh --with-torch  also install torch/torchvision on macOS (torchvision backend)
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$SKILL_DIR/.venv"
PY="$VENV/bin/python"
WITH_TORCH=0
[[ "${1:-}" == "--with-torch" ]] && WITH_TORCH=1
OS="$(uname -s)"
[[ "$OS" != "Darwin" ]] && WITH_TORCH=1

# 1. Virtual environment (uv if available, else venv + pip)
if [[ ! -x "$PY" ]]; then
  echo "Creating venv at $VENV"
  if command -v uv >/dev/null; then uv venv "$VENV"; else python3 -m venv "$VENV"; fi
fi
install() {
  if command -v uv >/dev/null; then uv pip install --python "$PY" "$@"; else "$PY" -m pip install -q "$@"; fi
}
install "rembg[cpu]" pillow numpy opencv-python
if [[ $WITH_TORCH == 1 ]]; then
  if [[ "$OS" == "Darwin" ]]; then install torch torchvision
  else install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  fi
fi

# 2. Apple Vision helper (macOS only)
if [[ "$OS" == "Darwin" ]]; then
  mkdir -p "$SKILL_DIR/scripts/bin"
  BIN="$SKILL_DIR/scripts/bin/inst"
  if [[ ! -x "$BIN" || "$SKILL_DIR/scripts/inst.swift" -nt "$BIN" ]]; then
    echo "Compiling Vision helper"
    swiftc -O "$SKILL_DIR/scripts/inst.swift" -o "$BIN"
  fi
fi

# 3. Face models (OpenCV Zoo, Apache-2.0)
ZOO="https://github.com/opencv/opencv_zoo/raw/main/models"
mkdir -p "$SKILL_DIR/models"
fetch() {
  local out="$SKILL_DIR/models/$(basename "$1")"
  # Git LFS pointer files are tiny; treat anything under 100 kB as missing
  if [[ ! -f "$out" || $(wc -c <"$out") -lt 100000 ]]; then
    echo "Downloading $(basename "$1")"
    curl -fsSL "$ZOO/$1" -o "$out.part" && mv "$out.part" "$out"
  fi
}
fetch face_detection_yunet/face_detection_yunet_2023mar.onnx
fetch face_recognition_sface/face_recognition_sface_2021dec.onnx

"$PY" -c "import rembg, cv2, PIL, numpy; print('OK: rembg', rembg.__version__ if hasattr(rembg,'__version__') else '', 'opencv', cv2.__version__)"
echo "Setup done. Python: $PY"
echo "Note: the matting model (~1 GB) downloads to ~/.rembg (or \$U2NET_HOME) on first run."
