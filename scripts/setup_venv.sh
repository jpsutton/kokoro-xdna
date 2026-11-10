#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Create the runtime venv (Python 3.13): onnxruntime, kokoro-onnx deps, pyxrt (built
# from source against the system XRT), FastAPI server deps.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
git submodule update --init third_party/kokoro-onnx  # no-op outside a git checkout
uv venv -p 3.13 venv
uv pip install -p venv/bin/python -r requirements.txt
scripts/build_pyxrt.sh "$ROOT/venv"
scripts/build_libs.sh
