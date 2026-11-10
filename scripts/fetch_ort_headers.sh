#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Download the ONNX Runtime C/C++ API headers matching the pip wheel (needed only to
# build csrc/npu_ops.cc). They are MIT-licensed upstream and are not vendored here.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VER="${ORT_VERSION:-1.31.0}"
DEST="$ROOT/third_party/onnxruntime-include"
mkdir -p "$DEST"
for f in onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_ep_c_api.h \
         onnxruntime_error_code.h onnxruntime_experimental_c_api.h onnxruntime_float16.h; do
  [ -f "$DEST/$f" ] || curl -sfL -o "$DEST/$f" \
    "https://raw.githubusercontent.com/microsoft/onnxruntime/v$VER/include/onnxruntime/core/session/$f"
done
