#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone pyxrt test of an mlir-aie whole_array matmul xclbin (bf16 in, f32 out)."""
import sys
import time
import numpy as np
import pyxrt
import ml_dtypes

xclbin_path, insts_path = sys.argv[1], sys.argv[2]
M = K = N = 512

dev = pyxrt.device(0)
xclbin = pyxrt.xclbin(xclbin_path)
dev.register_xclbin(xclbin)
ctx = pyxrt.hw_context(dev, xclbin.get_uuid())
kname = xclbin.get_kernels()[0].get_name()
kernel = pyxrt.kernel(ctx, kname)
print("kernel", kname, "nargs", xclbin.get_kernels()[0].get_num_args())

insts = np.fromfile(insts_path, dtype=np.uint32)
bo_insts = pyxrt.bo(dev, insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
bo_insts.write(insts.tobytes(), 0)
bo_insts.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

rng = np.random.default_rng(0)
A = rng.standard_normal((M, K), dtype=np.float32)
B = rng.standard_normal((K, N), dtype=np.float32)
A16 = A.astype(ml_dtypes.bfloat16)
B16 = B.astype(ml_dtypes.bfloat16)

bo_a = pyxrt.bo(dev, A16.nbytes, pyxrt.bo.host_only, kernel.group_id(3))
bo_b = pyxrt.bo(dev, B16.nbytes, pyxrt.bo.host_only, kernel.group_id(4))
bo_c = pyxrt.bo(dev, M * N * 4, pyxrt.bo.host_only, kernel.group_id(5))
bo_a.write(A16.tobytes(), 0)
bo_b.write(B16.tobytes(), 0)
bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
bo_b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

for i in range(3):
    t = time.perf_counter()
    run = kernel(3, bo_insts, insts.nbytes, bo_a, bo_b, bo_c)
    st = run.wait()
    dt = time.perf_counter() - t
    print(f"run {i}: {st} {dt*1e3:.2f} ms")

bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
C = np.frombuffer(bo_c.read(M * N * 4, 0), dtype=np.float32).reshape(M, N)
ref = A16.astype(np.float32) @ B16.astype(np.float32)
err = np.abs(C - ref)
rel = np.linalg.norm(C - ref) / np.linalg.norm(ref)
print(f"max abs err {err.max():.4f}  rel fro err {rel:.2e}  C[0,:4]={C[0,:4]} ref={ref[0,:4]}")
print("neg fraction in A:", (A < 0).mean())
