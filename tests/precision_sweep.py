#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare precision policies against the CPU reference: spectrogram corr, LSD (dB), wall time.

usage: precision_sweep.py POLICY   (as in KOKORO_NPU_PRECISION); run once per policy
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from kokoro_npu.engine import KokoroNPU

TEXTS = [
    "Hello! This is a test of NPU accelerated Kokoro text to speech. The quick brown fox jumps over the lazy dog.",
    "In the beginning the universe was created. This has made a lot of people very angry and been widely regarded as a bad move. "
    "Far out in the uncharted backwaters of the unfashionable end of the western spiral arm of the Galaxy lies a small unregarded yellow sun. "
    "Orbiting this at a distance of roughly ninety-two million miles is an utterly insignificant little blue green planet whose ape-descended life forms are so amazingly primitive that they still think digital watches are a pretty neat idea.",
]

def spec(a, n=1024, hop=256):
    w = np.hanning(n)
    fr = np.stack([a[i:i + n] * w for i in range(0, len(a) - n, hop)])
    return np.abs(np.fft.rfft(fr, axis=1))

def metrics(ref, a):
    n = min(len(ref), len(a))
    s0, s1 = spec(ref[:n]), spec(a[:n])
    l0, l1 = 20 * np.log10(s0 + 1e-5), 20 * np.log10(s1 + 1e-5)
    lsd = np.sqrt(((l0 - l1) ** 2).mean(axis=1)).mean()
    return np.corrcoef(s0.ravel(), s1.ravel())[0, 1], lsd

VOICE = os.environ.get("VOICE", "af_heart")
REF = f"/tmp/kokoro_precision_refs_{VOICE}.npz"
if not os.path.exists(REF):
    cpu = KokoroNPU(use_npu=False, intra_op_threads=4)
    np.savez(REF, *[cpu.synthesize(t, VOICE)[0] for t in TEXTS])
refs = [np.load(REF)[f"arr_{i}"] for i in range(len(TEXTS))]
# one NPU engine per process (a second hw context on the same xclbin is refused)
for pol in sys.argv[1:2]:
    eng = KokoroNPU(use_npu=True, precision=pol, intra_op_threads=2)
    eng.synthesize(TEXTS[0], VOICE)
    out = []
    for t, r in zip(TEXTS, refs):
        t0 = time.perf_counter(); a, sr = eng.synthesize(t, VOICE); dt = time.perf_counter() - t0
        c, l = metrics(r, a)
        out.append(f"{len(a)/sr:5.1f}s wall {dt:.2f} corr {c:.4f} LSD {l:.2f}")
    print(f"{pol:60s} | " + " | ".join(out), flush=True)
