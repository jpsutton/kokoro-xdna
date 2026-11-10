#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fetch the Kokoro v1.0 ONNX model and voice embeddings (kokoro-onnx release).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/models"
cd "$ROOT/models"
BASE=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
[ -f kokoro-v1.0.onnx ] || curl -L --fail -o kokoro-v1.0.onnx "$BASE/kokoro-v1.0.onnx"
[ -f voices-v1.0.bin ] || curl -L --fail -o voices-v1.0.bin "$BASE/voices-v1.0.bin"
sha256sum kokoro-v1.0.onnx voices-v1.0.bin
