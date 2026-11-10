#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Create mlir-aie/ironenv (Python 3.12) with the mlir-aie v1.4.4 wheel + Peano.
# Compile-time only; the runtime venv (scripts/setup_venv.sh) does not need it.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ ! -d "$ROOT/mlir-aie" ]; then
  git clone https://github.com/Xilinx/mlir-aie.git "$ROOT/mlir-aie"
fi
cd "$ROOT/mlir-aie"
git checkout -q v1.4.4
PY312="${PYTHON312:-$(uv python find 3.12 2>/dev/null || command -v python3.12)}"
export PYTHON="$PY312"
# shellcheck disable=SC1091
source utils/env_install.sh ironenv
