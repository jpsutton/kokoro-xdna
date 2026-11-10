# SPDX-License-Identifier: Apache-2.0
"""XRT runtime for the fused snake + conv-packing kernel (kernels/snake_pack_npu.py).

Writes snake(x) as bf16 row variants straight into the conv1d kernel's Z buffer
(see kokoro_npu/npu_conv.py for the layout):

    Z[(c*V + v)*ldx + e + t] = part_v( y[c, t + sh_v] ),   y = snake(x)
    e = pad_l rounded up to even,  sh_v = (e - pad_l) + shift_v * dil
"""

from __future__ import annotations

import ctypes
import glob
import os
import threading
import time

import ml_dtypes
import numpy as np
import pyxrt

from . import host
from .npu_conv import ConvGeom

N_DISPATCH_PARAMS = 5
MAXV = 6  # params tile layout: p[2:2+MAXV] shifts, p[2+MAXV:2+2*MAXV] activation parts
MAX_NT = 64
X_ELEMS = 32 * 1024 * 1024
PARAMS_LEN = 16


def _round_up(x: int, q: int) -> int:
    return -(-x // q) * q


class NpuSnakePack:
    def __init__(self, xclbin_path: str, L: int, V: int, cols: int, device, group_ids=(3, 4, 5)):
        self.L, self.V, self.cols = L, V, cols
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

        self.dev = device
        self.xclbin = pyxrt.xclbin(xclbin_path)
        self.dev.register_xclbin(self.xclbin)
        self.ctx = pyxrt.hw_context(self.dev, self.xclbin.get_uuid())
        self.kernel = pyxrt.kernel(self.ctx, self.xclbin.get_kernels()[0].get_name())
        self._lock = threading.Lock()
        self._insts_cache: dict[tuple, tuple[pyxrt.bo, int]] = {}
        self._bo_p: dict[object, pyxrt.bo] = {}
        self._bo_x: pyxrt.bo | None = None
        self._x_cap = 0
        self.stats = {"calls": 0, "npu_s": 0.0, "host_s": 0.0}

    def _insts(self, args):
        hit = self._insts_cache.get(args)
        if hit is not None:
            return hit
        out = ctypes.POINTER(ctypes.c_uint32)()
        cnt = self._lib.dispatch_generate(*args, ctypes.byref(out))
        if cnt < 0:
            why = self._lib.dispatch_last_refusal()
            raise RuntimeError(f"dispatch_generate{args} = {cnt}: {why.decode() if why else ''}")
        words = np.ctypeslib.as_array(out, shape=(cnt,)).copy()
        bo = pyxrt.bo(self.dev, words.nbytes, pyxrt.bo.cacheable, self.kernel.group_id(1))
        bo.write(words.tobytes(), 0)
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self._insts_cache[args] = (bo, words.nbytes)
        return bo, words.nbytes

    def _ensure_x(self, nbytes: int):
        if nbytes > self._x_cap:
            cap = max(nbytes, 2 * self._x_cap)
            self._bo_x = pyxrt.bo(self.dev, cap, pyxrt.bo.host_only, self.kernel.group_id(3))
            self._x_cap = cap

    @staticmethod
    def shifts(g: ConvGeom) -> tuple[int, list[int]]:
        e = _round_up(g.pad_l, 2)
        return e, [(e - g.pad_l) + v[0] * g.dil for v in g.variants]

    def params_bo(self, key, alpha: np.ndarray, beta: np.ndarray, g: ConvGeom) -> pyxrt.bo:
        """Params tile per channel: [alpha, beta, sh_0..sh_3, part_0..part_3, ...] (floats)."""
        e, sh = self.shifts(g)
        k = (key, tuple(sh))
        bo = self._bo_p.get(k)
        if bo is None:
            C = alpha.shape[0]
            p = np.zeros((C, self.L), dtype=np.float32)
            p[:, 0] = alpha
            p[:, 1] = beta
            for v, s in enumerate(sh):
                p[:, 2 + v] = s
                p[:, 2 + MAXV + v] = g.variants[v][1]
            bo = pyxrt.bo(self.dev, p.nbytes, pyxrt.bo.host_only, self.kernel.group_id(4))
            bo.write(p.tobytes(), 0)
            bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._bo_p[k] = bo
        return bo

    def plan(self, g: ConvGeom) -> tuple[int, int, int]:
        """(Tp, nt, min_ldx) for a conv geometry: Tp covers t_in + max lookahead."""
        e, sh = self.shifts(g)
        need = g.t_in + max(sh) + 1
        tiles0 = -(-need // self.L)
        best = None
        for extra in range(0, 8):
            tiles = tiles0 + extra
            nt = max(d for d in range(1, MAX_NT + 1) if tiles % d == 0)
            cost = extra * self.L + (tiles // nt) * 64
            if best is None or cost < best[0]:
                best = (cost, tiles, nt)
        Tp = best[1] * self.L
        return Tp, best[2], e + Tp

    def produce(self, x: np.ndarray, p_key, alpha, beta, g: ConvGeom, bo_z: pyxrt.bo, cin_pad: int, ci0: int = 0) -> float:
        """Fill Z (the conv's B buffer) for channel chunk x [cin_chunk, t_in] = channels [ci0, ci0+C).
        alpha/beta are the full per-channel arrays. Returns NPU seconds."""
        C, T = x.shape
        alpha = alpha[ci0 : ci0 + C]
        beta = beta[ci0 : ci0 + C]
        p_key = (p_key, ci0)
        if C % self.rows_per_group:
            raise ValueError(f"C={C} not a multiple of {self.rows_per_group}")
        if cin_pad != C:
            raise ValueError("fused snake requires cin == cin_pad (single chunk, multiple of k)")
        Tp, nt, min_ldx = self.plan(g)
        if g.ldx < min_ldx or C * Tp > X_ELEMS:
            raise ValueError(f"geometry {g} not compatible with fused snake plan Tp={Tp}")
        e, sh = self.shifts(g)
        with self._lock:
            t0 = time.perf_counter()
            bo_p = self.params_bo(p_key, alpha, beta, g)
            nbytes = C * Tp * 4
            self._ensure_x(nbytes)
            xv = np.frombuffer(self._bo_x.map(), dtype=np.float32, count=C * Tp).reshape(C, Tp)
            host.copy_rows_pad(x, xv, C, T)
            self._bo_x.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, nbytes, 0)
            # zero the Z columns the kernel does not write: [0, e) and [e + Tp, ldx)
            rows = C * self.V
            z = np.frombuffer(bo_z.map(), dtype=np.uint16, count=rows * g.ldx).reshape(rows, g.ldx)
            host.zero_strips(z, rows, g.ldx, e, Tp)
            bo_z.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, rows * g.ldx * 2, 0)
            bo_i, n_i = self._insts((C, Tp, nt, g.ldx, e))
            t1 = time.perf_counter()
            st = self.kernel(3, bo_i, n_i, self._bo_x, bo_p, bo_z).wait()
            t2 = time.perf_counter()
            if st != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(f"NPU snake_pack kernel returned {st}")
            # The DMA cannot write below the even column e, but the columns
            # [pad_l - shift_v*dil, e) hold real data: fill those few on the host.
            nhead = max(sh)
            if nhead:
                yh = x[:, :nhead].astype(np.float64)
                yh = (yh + beta[:, None] * np.sin(alpha[:, None] * yh) ** 2).astype(np.float32)
                hi = yh.astype(ml_dtypes.bfloat16)
                lo = (yh - hi.astype(np.float32)).astype(ml_dtypes.bfloat16)
                for v, (shift, part, _wpart) in enumerate(g.variants):
                    start = max(0, g.pad_l - shift * g.dil)
                    n = e - start
                    if n <= 0:
                        continue
                    j0 = start - g.pad_l + shift * g.dil  # y index landing at column `start`
                    src = (lo if part else hi)[:, j0 : j0 + n].view(np.uint16)
                    z[v::self.V, start:e] = src
                bo_z.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, rows * g.ldx * 2, 0)
            self.stats["calls"] += 1
            self.stats["npu_s"] += t2 - t1
            self.stats["host_s"] += t1 - t0
            return t2 - t1
