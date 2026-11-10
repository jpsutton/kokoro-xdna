#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wall time and process CPU time (user+sys, all threads) per synthesis, CPU-only vs NPU-offloaded."""
import sys, os, time, resource, logging, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kokoro_npu.engine import KokoroNPU
text = "Hello! This is a test of NPU accelerated Kokoro text to speech. The quick brown fox jumps over the lazy dog."
modes = sys.argv[1:] or ["cpu", "npu"]
def cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
for mode in modes:
    eng = KokoroNPU(use_npu=(mode == "npu"), intra_op_threads=int(os.environ.get("ORT_THREADS", "0")) or None)
    eng.synthesize("warm up please.", "af_heart"); eng.synthesize(text, "af_heart")
    n = 3; c0 = cpu_s(); t0 = time.perf_counter()
    for _ in range(n): a, sr = eng.synthesize(text, "af_heart")
    wall = (time.perf_counter() - t0) / n; cpu = (cpu_s() - c0) / n
    dur = len(a) / sr
    print(f"{mode}: audio {dur:.2f}s  wall {wall:.3f}s (RTF {wall/dur:.3f})  CPU {cpu:.3f}s  cpu/wall {cpu/wall:.2f}  stats {json.dumps({k: round(v,3) for k,v in eng.npu_stats().items()})}")
