#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Start the OpenAI-compatible speech server (no API key required).
#
#   scripts/serve.sh [--cpu] [--stream] [--host HOST] [--port PORT] [--threads N]
#
# Endpoint: POST http://HOST:PORT/v1/audio/speech  (any Authorization header is ignored)
# --stream (or STREAM_DEFAULT=1): stream audio sentence by sentence unless the request says "stream": false.
# Logging: LOG_LEVEL=DEBUG (adds request headers), LOG_BODY_CHARS=N (request body preview, default 0=off).
# Defaults: 127.0.0.1:8880, NPU on, 2 ORT threads. Use --host 0.0.0.0 to serve the LAN
# (there is no authentication, so only do that on a trusted network).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8880}"
export ORT_THREADS="${ORT_THREADS:-2}"
export KOKORO_NPU="${KOKORO_NPU:-1}"
export STREAM_DEFAULT="${STREAM_DEFAULT:-0}"

while [ $# -gt 0 ]; do
  case "$1" in
    --cpu) KOKORO_NPU=0 ;;
    --stream) STREAM_DEFAULT=1 ;;
    --host) HOST="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    --threads) ORT_THREADS="$2"; shift ;;
    -h|--help) sed -n '3,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
export HOST PORT ORT_THREADS KOKORO_NPU STREAM_DEFAULT

PY="$ROOT/venv/bin/python"
[ -x "$PY" ] || { echo "missing venv: run scripts/setup_venv.sh" >&2; exit 1; }
for f in models/kokoro-v1.0.onnx models/voices-v1.0.bin; do
  [ -f "$f" ] || { echo "missing $f: run scripts/download_models.sh" >&2; exit 1; }
done

if [ "$KOKORO_NPU" != "0" ]; then
  [ -e /dev/accel/accel0 ] || { echo "no /dev/accel/accel0: NPU driver not loaded (use --cpu to run without)" >&2; exit 1; }
  for f in build/libnpu_ops.so build/libnpu_host.so kernels/build/conv1d_m32k64n64c8.xclbin; do
    [ -f "$f" ] || { echo "missing $f: run scripts/build_libs.sh and scripts/build_kernels.sh" >&2; exit 1; }
  done
  # XRT needs to pin buffers; a small memlock limit makes it fail with mmap err=-11.
  if [ "$(ulimit -l)" != "unlimited" ]; then
    echo "warning: memlock limit is $(ulimit -l) KiB; set 'memlock unlimited' in /etc/security/limits.conf and re-login" >&2
  fi
fi

echo "Kokoro TTS: http://$HOST:$PORT/v1/audio/speech  (npu=$KOKORO_NPU, threads=$ORT_THREADS, stream default=$STREAM_DEFAULT, no API key)"
exec "$PY" server.py
