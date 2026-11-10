#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fused snake -> conv on NPU vs numpy conv(snake(x))."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, ml_dtypes
from kokoro_npu.npu_conv import NpuConv1d
from kokoro_npu.npu_snake_pack import NpuSnakePack

precision = sys.argv[1] if len(sys.argv) > 1 else "bf16x2"
V = {"bf16": 2, "bf16x2": 4, "bf16x3": 6}[precision]
conv = NpuConv1d("kernels/build/conv1d_m32k64n64c8.xclbin", m=32, k=64, n=64, cols=8, precision=precision)
sp = NpuSnakePack(f"kernels/build/snake_pack_L1008v{V}c8.xclbin", L=1008, V=V, cols=8, device=conv.dev)

def ref_conv1d(x, w, dil, pad):
    cout, cin, taps = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad))); t_out = xp.shape[1] - (taps - 1) * dil
    y = np.zeros((cout, t_out))
    for j in range(taps): y += w[:, :, j].astype(np.float64) @ xp[:, j * dil : j * dil + t_out].astype(np.float64)
    return y.astype(np.float32)

rng = np.random.default_rng(0)
for ci, (cout, cin, taps, dil, t) in enumerate([(128, 128, 11, 5, 49081), (128, 128, 3, 1, 49081), (256, 256, 7, 3, 8180), (128, 128, 11, 1, 2100), (256, 256, 3, 5, 8180)]):
    pad = (taps - 1) * dil // 2
    x = (rng.standard_normal((cin, t)) * 2).astype(np.float32)
    alpha = np.exp(rng.standard_normal(cin)).astype(np.float32); beta = (1 / (alpha + 1e-9)).astype(np.float32)
    w = (rng.standard_normal((cout, cin, taps), dtype=np.float32) / np.sqrt(cin * taps))
    if precision != "bf16x3": w = w.astype(ml_dtypes.bfloat16).astype(np.float32)
    b = rng.standard_normal(cout).astype(np.float32)
    g0 = conv.geometry(cout, cin, taps, dil, t, pad, pad)
    Tp, nt, min_ldx = sp.plan(g0)
    g = conv.geometry(cout, cin, taps, dil, t, pad, pad, min_ldx=min_ldx)
    wm = conv.prepare_weight(w, g)
    def producer(xs, gg, ci0, cin_pad, _key=("sp", ci)):
        return sp.produce(xs, _key, alpha, beta, gg, conv.bo_b, cin_pad, ci0)
    adj = lambda gg: conv.geometry(gg.cout, gg.cin, gg.taps, gg.dil, gg.t_in, gg.pad_l, gg.pad_r, min_ldx=sp.plan(gg)[2])
    y = conv.run(x, ("w", ci), wm, g, z_producer=producer, geom_adjust=adj, bias=b)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter(); y = conv.run(x, ("w", ci), wm, g, z_producer=producer, geom_adjust=adj, bias=b); ts.append(time.perf_counter() - t0)
    ys = x + beta[:, None] * np.sin(alpha[:, None] * x.astype(np.float64)) ** 2
    if precision == "bf16": ys = ys.astype(ml_dtypes.bfloat16).astype(np.float64)
    r = ref_conv1d(ys.astype(np.float32), w, dil, pad) + b[:, None]
    rel = np.linalg.norm(y - r) / np.linalg.norm(r)
    print(f"cout={cout} cin={cin} k={taps} d={dil} T={t}: rel err {rel:.2e}  {min(ts)*1e3:.1f} ms  (Tp={Tp} nt={nt} ldx={g.ldx})  snake npu {sp.stats['npu_s']:.3f}s host {sp.stats['host_s']:.3f}s")
