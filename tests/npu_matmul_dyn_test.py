#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Test a DispatchTime whole_array xclbin + dispatch .so from plain pyxrt (no mlir-aie at runtime)."""
import ctypes
import sys
import time
import numpy as np
import pyxrt
import ml_dtypes

xclbin_path, so_path = sys.argv[1], sys.argv[2]
shapes = [tuple(int(x) for x in s.split("x")) for s in sys.argv[3:]] or [(512, 512, 512)]

lib = ctypes.CDLL(so_path)
lib.dispatch_abi.restype = ctypes.c_char_p
print("abi:", lib.dispatch_abi().decode())
lib.dispatch_generate.restype = ctypes.c_int64
lib.dispatch_generate.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))]
lib.dispatch_last_refusal.restype = ctypes.c_char_p


def gen_insts(M, K, N):
    out = ctypes.POINTER(ctypes.c_uint32)()
    n = lib.dispatch_generate(M, K, N, ctypes.byref(out))
    if n < 0:
        raise RuntimeError(f"dispatch_generate={n}: {lib.dispatch_last_refusal()}")
    return np.ctypeslib.as_array(out, shape=(n,)).copy()


dev = pyxrt.device(0)
xclbin = pyxrt.xclbin(xclbin_path)
dev.register_xclbin(xclbin)
ctx = pyxrt.hw_context(dev, xclbin.get_uuid())
kernel = pyxrt.kernel(ctx, xclbin.get_kernels()[0].get_name())

for M, K, N in shapes:
    insts = gen_insts(M, K, N)
    bo_insts = pyxrt.bo(dev, insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
    bo_insts.write(insts.tobytes(), 0)
    bo_insts.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    rng = np.random.default_rng(0)
    A16 = rng.standard_normal((M, K), dtype=np.float32).astype(ml_dtypes.bfloat16)
    B16 = rng.standard_normal((K, N), dtype=np.float32).astype(ml_dtypes.bfloat16)
    bo_a = pyxrt.bo(dev, A16.nbytes, pyxrt.bo.host_only, kernel.group_id(3))
    bo_b = pyxrt.bo(dev, B16.nbytes, pyxrt.bo.host_only, kernel.group_id(4))
    bo_c = pyxrt.bo(dev, M * N * 4, pyxrt.bo.host_only, kernel.group_id(5))
    bo_a.write(A16.tobytes(), 0); bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_b.write(B16.tobytes(), 0); bo_b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    ts = []
    for i in range(4):
        t = time.perf_counter()
        st = kernel(3, bo_insts, insts.nbytes, bo_a, bo_b, bo_c).wait()
        ts.append(time.perf_counter() - t)
    bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    C = np.frombuffer(bo_c.read(M * N * 4, 0), dtype=np.float32).reshape(M, N)
    ref = A16.astype(np.float32) @ B16.astype(np.float32)
    rel = np.linalg.norm(C - ref) / np.linalg.norm(ref)
    gflops = 2 * M * K * N / min(ts) / 1e9
    print(f"{M}x{K}x{N}: insts={len(insts)} words  {st}  best {min(ts)*1e3:.2f} ms ({gflops:.0f} GFLOP/s)  rel err {rel:.2e}")
