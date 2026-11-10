#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the pyxrt extension module for the project venv's Python version.
#
# Arch's `xrt` package only ships pyxrt for the system Python (3.14). The
# project venv uses 3.13 (onnxruntime-extensions has no 3.14 wheel), so build
# pyxrt from the matching XRT source tag against the system XRT headers/libs.
#
# Usage: scripts/build_pyxrt.sh [venv-dir]
set -euo pipefail

VENV="${1:-$(cd "$(dirname "$0")/.." && pwd)/venv}"
PY="$VENV/bin/python"
XRT_VER="$(pacman -Q xrt | awk '{print $2}' | sed -E 's/^[0-9]+://; s/-[0-9]+$//')"
WORK="${TMPDIR:-/tmp}/pyxrt-build"
mkdir -p "$WORK"

if [ ! -f "$WORK/xrt-src/src/python/pybind11/src/pyxrt.cpp" ]; then
  git clone -q --depth 1 --branch "$XRT_VER" https://github.com/Xilinx/XRT.git "$WORK/xrt-src"
fi

# XRT public headers pull in boost/format.hpp and boost/any.hpp (header-only).
if [ ! -f "$WORK/boost/boost/format.hpp" ]; then
  mkdir -p "$WORK/boost"
  curl -sL https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.gz \
    | tar -xz -C "$WORK/boost" --strip-components=1 boost_1_86_0/boost
fi

"$PY" -m pip install -q pybind11
PB_INC="$("$PY" -m pybind11 --includes)"
EXT="$("$PY" -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')"
SITE="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"

# shellcheck disable=SC2086
g++ -O2 -shared -fPIC -std=c++17 $PB_INC -I"$WORK/boost" -I/usr/include/xrt \
  -Wno-deprecated-declarations \
  "$WORK/xrt-src/src/python/pybind11/src/pyxrt.cpp" \
  -o "$SITE/pyxrt$EXT" -lxrt_coreutil -luuid -lpthread

"$PY" -c 'import pyxrt; print("pyxrt OK:", pyxrt.device(0).get_info(pyxrt.xrt_info_device.name))'
