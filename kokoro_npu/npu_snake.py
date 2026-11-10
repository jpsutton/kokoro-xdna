# SPDX-License-Identifier: Apache-2.0
"""XRT runtime for the NPU snake activation kernel (kernels/snake_npu.py).

y[c, t] = x[c, t] + beta[c] * sin(alpha[c] * x[c, t])^2, f32.
"""

from __future__ import annotations

import ctypes
import glob
import os
import threading
import time

import numpy as np
import pyxrt

from . import host

N_DISPATCH_PARAMS = 2
MAX_TILES = 1023
X_ELEMS = 32 * 1024 * 1024  # host buffer size the xclbin was compiled with


def _round_up(x: int, q: int) -> int:
    return -(-x // q) * q


class NpuSnake:
    def __init__(self, xclbin_path: str, L: int, cols: int, device_id: int = 0, device=None):
        self.L, self.cols = L, cols
        self.rows_per_group = 4 * cols
        prj = os.path.splitext(xclbin_path)[0] + ".prj"
        so = glob.glob(os.path.join(prj, "dispatch-*.so"))
        if len(so) != 1:
            raise FileNotFoundError(f"expected one dispatch-*.so in {prj}, found {so}")
        self._lib = ctypes.CDLL(so[0])
        self._lib.dispatch_abi.restype = ctypes.c_char_p
        abi = self._lib.dispatch_abi().decode().split(",")
        if len(abi) != N_DISPATCH_PARAMS:
            raise RuntimeError(f"unexpected dispatch ABI {abi}")
        self._lib.dispatch_generate.restype = ctypes.c_int64
        self._lib.dispatch_generate.argtypes = [ctypes.c_int32] * N_DISPATCH_PARAMS + [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))
        ]
        self._lib.dispatch_last_refusal.restype = ctypes.c_char_p

        self.dev = device if device is not None else pyxrt.device(device_id)
        self.xclbin = pyxrt.xclbin(xclbin_path)
        self.dev.register_xclbin(self.xclbin)
        self.ctx = pyxrt.hw_context(self.dev, self.xclbin.get_uuid())
        self.kernel = pyxrt.kernel(self.ctx, self.xclbin.get_kernels()[0].get_name())
        self._lock = threading.Lock()
        self._insts_cache: dict[tuple, tuple[pyxrt.bo, int]] = {}
        self._bo_p: dict[object, pyxrt.bo] = {}
        self._bo_x: pyxrt.bo | None = None
        self._bo_y: pyxrt.bo | None = None
        self._cap = 0
        self.stats = {"calls": 0, "npu_s": 0.0, "host_s": 0.0}

    def _insts(self, C: int, T: int):
        key = (C, T)
        hit = self._insts_cache.get(key)
        if hit is not None:
            return hit
        out = ctypes.POINTER(ctypes.c_uint32)()
        cnt = self._lib.dispatch_generate(C, T, ctypes.byref(out))
        if cnt < 0:
            why = self._lib.dispatch_last_refusal()
            raise RuntimeError(f"dispatch_generate{key} = {cnt}: {why.decode() if why else ''}")
        words = np.ctypeslib.as_array(out, shape=(cnt,)).copy()
        bo = pyxrt.bo(self.dev, words.nbytes, pyxrt.bo.cacheable, self.kernel.group_id(1))
        bo.write(words.tobytes(), 0)
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self._insts_cache[key] = (bo, words.nbytes)
        return bo, words.nbytes

    def _ensure(self, nbytes: int):
        if nbytes > self._cap:
            cap = max(nbytes, 2 * self._cap)
            self._bo_x = pyxrt.bo(self.dev, cap, pyxrt.bo.host_only, self.kernel.group_id(3))
            self._bo_y = pyxrt.bo(self.dev, cap, pyxrt.bo.host_only, self.kernel.group_id(5))
            self._cap = cap

    def params_bo(self, key, alpha: np.ndarray, beta: np.ndarray) -> pyxrt.bo:
        bo = self._bo_p.get(key)
        if bo is None:
            C = alpha.shape[0]
            p = np.zeros((C, self.L), dtype=np.float32)
            p[:, 0] = alpha
            p[:, 1] = beta
            bo = pyxrt.bo(self.dev, p.nbytes, pyxrt.bo.host_only, self.kernel.group_id(4))
            bo.write(p.tobytes(), 0)
            bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._bo_p[key] = bo
        return bo

    def run(self, x: np.ndarray, p_key, alpha: np.ndarray, beta: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """x: [C, T] f32 -> y [C, T] f32."""
        C, T = x.shape
        if C % self.rows_per_group:
            raise ValueError(f"C={C} not a multiple of {self.rows_per_group}")
        Tp = _round_up(T, self.L)
        max_t = min(MAX_TILES, X_ELEMS // (C * self.L)) * self.L
        if T > max_t:
            # elementwise: split along T
            if out is None:
                out = np.empty((C, T), dtype=np.float32)
            for t0 in range(0, T, max_t):
                t1 = min(T, t0 + max_t)
                self.run(x[:, t0:t1], p_key, alpha, beta, out=out[:, t0:t1])
            return out
        with self._lock:
            t0 = time.perf_counter()
            bo_p = self.params_bo(p_key, alpha, beta)
            nbytes = C * Tp * 4
            self._ensure(nbytes)
            xv = np.frombuffer(self._bo_x.map(), dtype=np.float32, count=C * Tp).reshape(C, Tp)
            host.copy_rows_pad(x, xv, C, T)
            self._bo_x.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, nbytes, 0)
            bo_i, n_i = self._insts(C, Tp)
            t1 = time.perf_counter()
            st = self.kernel(3, bo_i, n_i, self._bo_x, bo_p, self._bo_y).wait()
            t2 = time.perf_counter()
            if st != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(f"NPU snake kernel returned {st}")
            self._bo_y.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, nbytes, 0)
            yv = np.frombuffer(self._bo_y.map(), dtype=np.float32, count=C * Tp).reshape(C, Tp)
            if out is None:
                out = np.empty((C, T), dtype=np.float32)
            host.copy_rows(yv, out, C, T)
            t3 = time.perf_counter()
            self.stats["calls"] += 1
            self.stats["npu_s"] += t2 - t1
            self.stats["host_s"] += (t1 - t0) + (t3 - t2)
            return out
