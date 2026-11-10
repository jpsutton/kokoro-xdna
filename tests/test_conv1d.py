#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate NpuConv1d against a numpy reference conv1d."""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from kokoro_npu.npu_conv import NpuConv1d

def ref_conv1d(x, w, dil, pad):
    cout, cin, taps = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad)))
    t_out = xp.shape[1] - (taps - 1) * dil
    y = np.zeros((cout, t_out), dtype=np.float64)
    for j in range(taps):
        y += w[:, :, j].astype(np.float64) @ xp[:, j * dil : j * dil + t_out].astype(np.float64)
    return y.astype(np.float32)

conv = NpuConv1d(sys.argv[1], m=32, k=64, n=int(sys.argv[3]) if len(sys.argv) > 3 else 32, cols=int(sys.argv[2]) if len(sys.argv) > 2 else 4, precision=sys.argv[4] if len(sys.argv) > 4 else 'bf16')
rng = np.random.default_rng(0)
cases = [(128, 128, 11, 5, 49081), (128, 128, 3, 1, 49081), (256, 256, 7, 3, 8180), (256, 256, 11, 1, 2000),
         (1024, 1090, 3, 1, 818), (512, 512, 5, 1, 818), (1024, 1090, 1, 1, 818)]
for ci, (cout, cin, taps, dil, t) in enumerate(cases):
    pad = (taps - 1) * dil // 2
    x = rng.standard_normal((cin, t), dtype=np.float32)
    w = (rng.standard_normal((cout, cin, taps), dtype=np.float32) / np.sqrt(cin * taps))
    b = rng.standard_normal(cout).astype(np.float32)
    g = conv.geometry(cout, cin, taps, dil, t, pad, pad)
    wm = conv.prepare_weight(w, g)
    y = conv.run(x, ("test", ci), wm, g, bias=b)  # warm
    ts = []
    for _ in range(3):
        t0 = time.perf_counter(); y = conv.run(x, ("test", ci), wm, g, bias=b); ts.append(time.perf_counter() - t0)
    # reference with bf16-rounded inputs
    import ml_dtypes
    xr = x if conv.precision in ("bf16x2", "bf16x3") else x.astype(ml_dtypes.bfloat16).astype(np.float32)
    wr = w if conv.precision == "bf16x3" else w.astype(ml_dtypes.bfloat16).astype(np.float32)
    r = ref_conv1d(xr, wr, dil, pad) + b[:, None]
    rel = np.linalg.norm(y - r) / np.linalg.norm(r)
    flops = 2 * cout * cin * taps * t
    print(f"cout={cout} cin={cin} k={taps} d={dil} T={t}: y{y.shape} rel err {rel:.2e}  {min(ts)*1e3:.1f} ms  {flops/min(ts)/1e9:.0f} GFLOP/s  (n_pad={g.n_pad} ldx={g.ldx})")
