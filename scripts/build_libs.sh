#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the ORT custom-op library and the host packing helpers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/scripts/fetch_ort_headers.sh"
mkdir -p "$ROOT/build"
g++ -O2 -shared -fPIC -std=c++17 -fvisibility=hidden -I"$ROOT/third_party/onnxruntime-include" \
  "$ROOT/csrc/npu_ops.cc" -o "$ROOT/build/libnpu_ops.so"
g++ -O3 -march=native -shared -fPIC -std=c++17 -fvisibility=hidden \
  "$ROOT/csrc/npu_host.cc" -o "$ROOT/build/libnpu_host.so"
ls -la "$ROOT/build"/*.so
