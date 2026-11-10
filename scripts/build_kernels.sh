#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compile the NPU kernels (xclbin + dispatch library) with the mlir-aie toolchain.
#
# Requires the IRON environment created by scripts/setup_ironenv.sh
# (mlir-aie/ironenv, Python 3.12). Artifacts land in kernels/build/.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/mlir-aie"
# shellcheck disable=SC1091
source ironenv/bin/activate
# shellcheck disable=SC1091
source utils/env_setup.sh >/dev/null
cd "$ROOT/kernels"
mkdir -p build
COLS="${NPU_COLS:-8}"
python3 conv1d_npu.py --dev npu2 -m 32 -k 64 -n 64 --n-aie-cols "$COLS" \
  --xclbin-path="build/conv1d_m32k64n64c${COLS}.xclbin"
python3 snake_npu.py --dev npu2 -L 1008 --n-aie-cols "$COLS" \
  --xclbin-path="build/snake_L1008c${COLS}.xclbin"
for V in 2 4 6; do
  python3 snake_pack_npu.py --dev npu2 -L 1008 -V "$V" --n-aie-cols "$COLS" \
    --xclbin-path="build/snake_pack_L1008v${V}c${COLS}.xclbin"
done
ls -la build/*.xclbin build/*.prj/dispatch-*.so
