#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare NPU-offloaded synthesis against the CPU-only reference (af_heart)."""
import sys, os, time, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(level=logging.INFO)
import numpy as np
from kokoro_npu.engine import KokoroNPU

text = sys.argv[1] if len(sys.argv) > 1 else "Hello! This is a test of NPU accelerated Kokoro text to speech. The quick brown fox jumps over the lazy dog."
cpu = KokoroNPU(use_npu=False)
npu = KokoroNPU(use_npu=True)
for name, eng in (("cpu", cpu), ("npu", npu)):
    eng.synthesize("warm up.", "af_heart")
res = {}
for name, eng in (("cpu", cpu), ("npu", npu)):
    t0 = time.perf_counter(); a, sr = eng.synthesize(text, "af_heart"); dt = time.perf_counter() - t0
    res[name] = a
    print(f"{name}: {len(a)/sr:.2f}s audio in {dt:.3f}s  RTF {dt/(len(a)/sr):.3f}  stats={eng.npu_stats()}")
a, b = res["cpu"], res["npu"]
n = min(len(a), len(b))
print("len cpu/npu", len(a), len(b))
d = a[:n] - b[:n]
print(f"max abs diff {np.abs(d).max():.4f}  rel rms {np.sqrt((d**2).mean())/np.sqrt((a[:n]**2).mean()):.3e}  corr {np.corrcoef(a[:n], b[:n])[0,1]:.6f}")
import soundfile as sf
sf.write("/tmp/npu_af_heart.wav", b, sr); sf.write("/tmp/cpu_af_heart.wav", a, sr)
