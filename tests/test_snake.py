#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from kokoro_npu.npu_snake import NpuSnake
sn = NpuSnake(sys.argv[1], L=int(sys.argv[2]) if len(sys.argv) > 2 else 512, cols=int(sys.argv[3]) if len(sys.argv) > 3 else 4)
rng = np.random.default_rng(0)
for C, T, scale in [(128, 49081, 3.0), (256, 8180, 10.0), (128, 2100, 30.0), (256, 49081, 1.0), (32, 49081, 1.0)]:
    x = (rng.standard_normal((C, T)) * scale).astype(np.float32)
    alpha = np.exp(rng.standard_normal(C)).astype(np.float32); beta = (1.0 / (alpha + 1e-9)).astype(np.float32)
    y = sn.run(x, ("t", C, T, scale), alpha, beta)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter(); y = sn.run(x, ("t", C, T, scale), alpha, beta); ts.append(time.perf_counter() - t0)
    r = x + beta[:, None] * np.sin(alpha[:, None] * x.astype(np.float64)) ** 2
    err = np.abs(y - r); rel = np.linalg.norm(y - r) / np.linalg.norm(r)
    print(f"C={C} T={T} |ax|max={np.abs(alpha[:,None]*x).max():.0f}: max abs err {err.max():.2e} rel {rel:.2e}  {min(ts)*1e3:.2f} ms ({C*T*4*2/min(ts)/1e9:.1f} GB/s)  npu {sn.stats['npu_s']:.3f}s host {sn.stats['host_s']:.3f}s")
