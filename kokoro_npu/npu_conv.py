# SPDX-License-Identifier: Apache-2.0
"""XRT runtime for the NPU conv1d kernel (kernels/conv1d_npu.py).

No mlir-aie dependency at runtime: the xclbin is loaded with pyxrt and the
per-call instruction stream comes from the compiled dispatch library via ctypes.

Per conv call (one "channel chunk" of cin_pad input channels):

  A  Wmat[cout, K] bf16   K = taps2 * cin2, cin2 = V * cin_pad. Column i*cin2 + c*V + v
                          holds W[co, c, 2i + shift_v] (0 if that tap does not exist).
  B  Z[cin2, ldx]  bf16   row c*V + v holds variant v of input channel c:
                            Z[c*V+v, col] = part_v(Xp[c, col + shift_v*dil])
                          where Xp is the zero-padded input (x at column pad_l) and
                          part_v is the bf16 "hi" rounding or the "lo" residual.
  C  Y[cout, n_pad] f32   first t_out columns are the conv output (no bias).

Variants: precision "bf16"   -> (shift 0, hi), (shift 1, hi)
          precision "bf16x2" -> + (shift 0, lo), (shift 1, lo)         [activation split]
          precision "bf16x3" -> + (shift 0, hi), (shift 1, hi) x W_lo  [weight split too]
Tap pairing exists because DMA offsets/strides are 4-byte granules and every
Kokoro dilation is odd: super-tap i covers taps 2i (shift 0) and 2i+1 (shift 1).

Convs with more input channels than fit `cin2 <= 1023` are split along Cin and
the partial outputs summed on the host (only the small decode-stage convs).
"""

from __future__ import annotations

import ctypes
import glob
import os
import threading
import time
from dataclasses import dataclass

import ml_dtypes
import numpy as np
import pyxrt

from . import host

BF16 = ml_dtypes.bfloat16
CHUNK_Q = 8  # how many extra column tiles of padding to consider when choosing a chunk size
MAX_NC = 64  # shim DMA iteration-count limit
MAX_D1 = 1023  # shim DMA d1 size limit (= cin2)
N_DISPATCH_PARAMS = 7
# Host buffer sizes the xclbin was compiled with (kernels/conv1d_npu.py defaults), in elements.
B_ELEMS = 32 * 1024 * 1024
C_ELEMS = 32 * 1024 * 1024
PIECE_SLACK = 4096  # frames of Z row headroom when splitting long convs into pieces

# A-tile stream layout of the matmul microkernel for m=32,k=64 (kernels.mm stream_dims.A):
# L1 position j holds flat A index A_ORDER[j]. The first 2*m positions carry the bias
# object's hi/lo halves (see kernels/conv_bias_unpack.cc).
_A_SIZES, _A_STRIDES = (8, 8, 4, 8), (256, 8, 64, 1)


def _a_order(sizes, strides):
    out = []

    def rec(level, base):
        if level == len(sizes):
            out.append(base)
            return
        for i in range(sizes[level]):
            rec(level + 1, base + i * strides[level])

    rec(0, 0)
    return np.array(out)


A_ORDER = _a_order(_A_SIZES, _A_STRIDES)

# (tap shift in units of dil, activation part, weight part) per Z row variant.
# part 0 = bf16 rounding of the value, part 1 = bf16 of the residual.
VARIANTS = {
    "bf16": ((0, 0, 0), (1, 0, 0)),
    "bf16x2": ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)),  # + activation residual
    "bf16x3": ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1)),  # + weight residual
}

PRECISION_OF = {v: k for k, v in VARIANTS.items()}


def _round_up(x: int, q: int) -> int:
    return -(-x // q) * q


@dataclass(frozen=True)
class ConvGeom:
    """Padded geometry of one conv (whole Cin; chunking is derived)."""

    cout: int
    cin: int
    taps: int
    taps2: int
    dil: int
    t_in: int
    pad_l: int
    pad_r: int
    t_out: int
    n_pad: int  # t_out rounded up
    ldx: int  # Z row stride (even)
    nc: int  # column tiles per shim column per DMA chunk
    cin_chunks: tuple[tuple[int, int], ...]  # (ci0, cin_pad) per NPU run
    variants: tuple[tuple[int, int], ...]

    @property
    def nvar(self) -> int:
        return len(self.variants)

    def dispatch_args(self, cin_pad: int) -> tuple[int, ...]:
        return (self.cout, self.nvar * cin_pad, self.taps2, 2 * self.dil, self.n_pad, self.ldx, self.nc)

    def z_shape(self, cin_pad: int) -> tuple[int, int]:
        return (self.nvar * cin_pad, self.ldx)


class NpuConv1d:
    """One xclbin (tile config) + one hw context; runs any conv geometry that fits."""

    def __init__(self, xclbin_path: str, m: int, k: int, n: int, cols: int, device_id: int = 0, precision: str = "bf16"):
        self.m, self.k, self.n, self.cols = m, k, n, cols
        self.variants = VARIANTS[precision]
        self.precision = precision
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

        self.dev = pyxrt.device(device_id)
        self.xclbin = pyxrt.xclbin(xclbin_path)
        self.dev.register_xclbin(self.xclbin)
        self.ctx = pyxrt.hw_context(self.dev, self.xclbin.get_uuid())
        self.kernel = pyxrt.kernel(self.ctx, self.xclbin.get_kernels()[0].get_name())
        self._lock = threading.Lock()
        self._insts_cache: dict[tuple, tuple[pyxrt.bo, int]] = {}
        self._bo_a: dict[object, list[pyxrt.bo]] = {}
        self._bo_d: dict[object, pyxrt.bo] = {}
        self._bo_d_zero: pyxrt.bo | None = None
        self._bo_b: pyxrt.bo | None = None
        self._bo_c: pyxrt.bo | None = None
        self._b_cap = 0
        self._c_cap = 0
        self.stats = {"calls": 0, "npu_s": 0.0, "host_s": 0.0}

    # -- helpers ---------------------------------------------------------
    def _insts(self, args: tuple[int, ...]):
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

    def ensure_bc(self, b_bytes: int, c_bytes: int):
        if b_bytes > self._b_cap:
            cap = max(b_bytes, 2 * self._b_cap)
            self._bo_b = pyxrt.bo(self.dev, cap, pyxrt.bo.host_only, self.kernel.group_id(4))
            self._b_cap = cap
        if c_bytes > self._c_cap:
            cap = max(c_bytes, 2 * self._c_cap)
            self._bo_c = pyxrt.bo(self.dev, cap, pyxrt.bo.host_only, self.kernel.group_id(5))
            self._c_cap = cap

    @property
    def bo_b(self) -> pyxrt.bo:
        return self._bo_b

    def _choose_padding(self, t_out: int) -> tuple[int, int]:
        """Pick (n_pad, nc): n_pad a multiple of n*cols with few wasted frames and a
        chunk size nc (tiles per shim column per DMA chunk) that divides the tile
        count, preferring large chunks (fewer BDs) over a little extra padding."""
        unit = self.n * self.cols
        base = -(-t_out // unit)
        best = None
        for extra in range(0, CHUNK_Q):
            tiles = base + extra
            nc = max(d for d in range(1, MAX_NC + 1) if tiles % d == 0)
            waste = extra * unit
            cost = waste + (tiles // nc) * 64
            if best is None or cost < best[0]:
                best = (cost, tiles * unit, nc)
        return best[1], best[2]

    def max_cin(self, nvar: int) -> int:
        """Largest cin_pad per chunk so that nvar * cin_pad <= MAX_D1."""
        return (MAX_D1 // nvar) // self.k * self.k

    def geometry(self, cout, cin, taps, dil, t_in, pad_l, pad_r, min_ldx: int = 0, precision: str | None = None) -> ConvGeom:
        variants = VARIANTS[precision or self.precision]
        if cout % (self.m * 4):
            raise ValueError(f"cout={cout} not a multiple of {self.m * 4}")
        taps2 = (taps + 1) // 2
        t_out = t_in + pad_l + pad_r - (taps - 1) * dil
        n_pad, nc = self._choose_padding(t_out)
        # Furthest element read: n_pad - 1 + (taps2-1)*2*dil, plus dil for the shifted variants.
        ldx = _round_up(max(n_pad + (taps2 - 1) * 2 * dil + dil, min_ldx), 2)
        n_chunks = -(-cin // self.max_cin(len(variants)))
        per = _round_up(-(-cin // n_chunks), self.k)
        chunks = []
        ci0 = 0
        while ci0 < cin:
            w = min(per, cin - ci0)
            chunks.append((ci0, _round_up(w, self.k)))
            ci0 += w
        return ConvGeom(cout, cin, taps, taps2, dil, t_in, pad_l, pad_r, t_out, n_pad, ldx, nc, tuple(chunks), variants)

    def fits(self, g: ConvGeom) -> bool:
        cin_pad_max = max(cp for _, cp in g.cin_chunks)
        return g.nvar * cin_pad_max * g.ldx <= B_ELEMS and g.cout * g.n_pad <= C_ELEMS

    def pieces(self, g: ConvGeom) -> list[tuple[int, int, int, int, int, int]]:
        """Split a too-long conv into output ranges: (o0, o1, i0, i1, pad_l, pad_r)."""
        cin_pad_max = max(cp for _, cp in g.cin_chunks)
        reach = (g.taps - 1) * g.dil
        # slack: the fused snake producer rounds Z rows up to whole tiles plus lookahead
        max_out = min(B_ELEMS // (g.nvar * cin_pad_max) - reach - 2 * g.dil - 2 - PIECE_SLACK, C_ELEMS // g.cout)
        max_out = max_out // (self.n * self.cols) * (self.n * self.cols)
        if max_out <= 0:
            raise ValueError(f"conv geometry {g} cannot fit the NPU buffers")
        out = []
        o0 = 0
        while o0 < g.t_out:
            o1 = min(g.t_out, o0 + max_out)
            i0 = o0 - g.pad_l
            i1 = o1 - g.pad_l + reach
            pl = max(0, -i0)
            pr = max(0, i1 - g.t_in)
            out.append((o0, o1, max(0, i0), min(g.t_in, i1), pl, pr))
            o0 = o1
        return out

    # -- weights -----------------------------------------------------------
    def prepare_weight(self, w: np.ndarray, g: ConvGeom) -> list[np.ndarray]:
        """W[cout, cin, taps] (f32) -> one Wmat[cout, K] bf16 per channel chunk."""
        cout, cin, taps = w.shape
        assert (cout, cin, taps) == (g.cout, g.cin, g.taps)
        wt = np.transpose(w, (0, 2, 1))  # [cout, taps, cin]
        mats = []
        for ci0, cin_pad in g.cin_chunks:
            wc = wt[:, :, ci0 : min(ci0 + cin_pad, cin)]
            ncin = wc.shape[2]
            w_hi = wc.astype(BF16).astype(np.float32)
            w_lo = wc - w_hi
            wmat = np.zeros((cout, g.taps2, cin_pad, g.nvar), dtype=np.float32)
            for v, (shift, _act_part, w_part) in enumerate(g.variants):
                taps_v = (w_lo if w_part else w_hi)[:, shift::2, :]  # taps 2i + shift
                wmat[:, : taps_v.shape[1], :ncin, v] = taps_v
            mats.append(np.ascontiguousarray(wmat.reshape(cout, g.taps2 * cin_pad * g.nvar)).astype(BF16))
        return mats

    def bias_bo(self, key, bias: np.ndarray | None, cout: int) -> pyxrt.bo:
        """Bias objects: one A-shaped tile per core row (cout/m of them), hi+lo bf16 in A stream order."""
        if bias is None:
            if self._bo_d_zero is None:
                z = np.zeros(64 * self.m * self.k, dtype=np.uint16)
                self._bo_d_zero = pyxrt.bo(self.dev, z.nbytes, pyxrt.bo.host_only, self.kernel.group_id(6))
                self._bo_d_zero.write(z.tobytes(), 0)
                self._bo_d_zero.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            return self._bo_d_zero
        bo = self._bo_d.get(key)
        if bo is None:
            n_rows = cout // self.m
            d = np.zeros((n_rows, self.m * self.k), dtype=np.uint16)
            b = bias.astype(np.float32).reshape(n_rows, self.m)
            hi = b.astype(BF16)
            lo = (b - hi.astype(np.float32)).astype(BF16)
            for r in range(n_rows):
                d[r, A_ORDER[: self.m]] = hi[r].view(np.uint16)
                d[r, A_ORDER[self.m : 2 * self.m]] = lo[r].view(np.uint16)
            bo = pyxrt.bo(self.dev, d.nbytes, pyxrt.bo.host_only, self.kernel.group_id(6))
            bo.write(d.tobytes(), 0)
            bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._bo_d[key] = bo
        return bo

    def weight_bos(self, key, wmats: list[np.ndarray]) -> list[pyxrt.bo]:
        bos = self._bo_a.get(key)
        if bos is None:
            bos = []
            for wm in wmats:
                bo = pyxrt.bo(self.dev, wm.nbytes, pyxrt.bo.host_only, self.kernel.group_id(3))
                bo.write(wm.tobytes(), 0)
                bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                bos.append(bo)
            self._bo_a[key] = bos
        return bos

    # -- run ----------------------------------------------------------------
    def run(self, x: np.ndarray, w_key, wmats: list[np.ndarray], g: ConvGeom, out: np.ndarray | None = None, z_producer=None, geom_adjust=None, bias: np.ndarray | None = None) -> np.ndarray:
        """x: [cin, t_in] f32. Returns y: [cout, t_out] f32 (+ bias if given), written into ``out`` if given.

        ``z_producer(x_slice, g, ci0, cin_pad)`` may fill the Z buffer instead of the
        host packer (used by the fused snake->conv path); it is called after
        ``ensure_bc`` with ``self.bo_b`` sized for the chunk.
        """
        if not self.fits(g):
            if out is None:
                out = np.empty((g.cout, g.t_out), dtype=np.float32)
            for o0, o1, i0, i1, pl, pr in self.pieces(g):
                gp = self.geometry(g.cout, g.cin, g.taps, g.dil, i1 - i0, pl, pr, precision=PRECISION_OF[g.variants])
                if geom_adjust is not None:
                    gp = geom_adjust(gp)
                assert gp.t_out == o1 - o0, (gp, o0, o1)
                self._run_fitting(x[:, i0:i1], w_key, wmats, gp, out[:, o0:o1], z_producer, bias)
            return out
        return self._run_fitting(x, w_key, wmats, g, out, z_producer, bias)

    def _run_fitting(self, x, w_key, wmats, g: ConvGeom, out, z_producer, bias):
        with self._lock:
            t0 = time.perf_counter()
            bos_a = self.weight_bos(w_key, wmats)
            bo_d = self.bias_bo(w_key, bias, g.cout)
            bo_d0 = self.bias_bo(None, None, g.cout)
            c_bytes = g.cout * g.n_pad * 4
            first = True
            npu_s = 0.0
            for chunk_i, ((ci0, cin_pad), bo_a) in enumerate(zip(g.cin_chunks, bos_a)):
                cin2 = g.nvar * cin_pad
                ncin = min(cin_pad, g.cin - ci0)
                b_bytes = cin2 * g.ldx * 2
                self.ensure_bc(b_bytes, c_bytes)
                if z_producer is not None:
                    npu_s += z_producer(x[ci0 : ci0 + ncin], g, ci0, cin_pad)
                    # the producer must have written the whole Z region of this chunk
                else:
                    z = np.frombuffer(self._bo_b.map(), dtype=np.uint16, count=cin2 * g.ldx).reshape(cin2, g.ldx)
                    host.pack_conv_z(x[ci0 : ci0 + ncin], cin_pad, g.ldx, g.pad_l, g.dil, g.variants, z)
                    self._bo_b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, b_bytes, 0)

                bo_i, n_i = self._insts(g.dispatch_args(cin_pad))
                t1 = time.perf_counter()
                # bias only in the first channel chunk (partial outputs are summed)
                st = self.kernel(3, bo_i, n_i, bo_a, self._bo_b, self._bo_c, bo_d if chunk_i == 0 else bo_d0).wait()
                npu_s += time.perf_counter() - t1
                if st != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                    raise RuntimeError(f"NPU conv kernel returned {st}")
                self._bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, c_bytes, 0)
                y = np.frombuffer(self._bo_c.map(), dtype=np.float32, count=g.cout * g.n_pad).reshape(
                    g.cout, g.n_pad
                )
                if first:
                    if out is None:
                        out = np.empty((g.cout, g.t_out), dtype=np.float32)
                    host.copy_rows(y, out, g.cout, g.t_out)
                    first = False
                else:
                    host.add_rows(y, out, g.cout, g.t_out)
            t3 = time.perf_counter()
            self.stats["calls"] += 1
            self.stats["npu_s"] += npu_s
            self.stats["host_s"] += (t3 - t0) - npu_s
            return out
