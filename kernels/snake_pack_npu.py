# SPDX-License-Identifier: Apache-2.0
#
# Fused Snake activation + conv input packing on AMD XDNA NPU via IRON (mlir-aie).
#
"""Snake(x) evaluated in f32 and written straight into the conv1d kernel's bf16
``Z`` buffer (see kernels/conv1d_npu.py and kokoro_npu/npu_conv.py):

    Z[(c*V + v)*ldx + e + t] = part_v( y[c, t + sh_v] )        y = snake(x)

Each compute core owns whole channel rows: core (row, col) processes channels
c = grp*4*cols + col*4 + row. The stream for a core is, per DMA chunk of ``nt``
tiles, one params tile (p[0]=alpha, p[1]=beta, p[2+v]=sh_v, p[2+4+v]=part_v, as
floats) followed by ``nt`` data tiles of L f32; the core emits V bf16 tiles per
data tile (p[2+6+v] = activation part). Shifted variants need the start of the next tile, so the core keeps
the previous tile's result and lags one tile behind (the last tile of a row
uses zeros as lookahead, matching the zero-padded input).

DispatchTime: C, T (= Tp, a multiple of L), nt (tiles per chunk; divides T/L
and nt <= 64), ldx, e (even). L, V and n_aie_cols are CompileTime.
"""

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Buffer,
    CompileTime,
    DispatchTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
    WorkerRuntimeBarrier,
    require,
)
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from ml_dtypes import bfloat16

N_AIE_ROWS = 4
HERE = Path(__file__).resolve().parent


@iron.jit
def snake_pack(
    X: In,
    P: In,
    Z: Out,
    *,
    C: DispatchTime[np.int32],
    T: DispatchTime[np.int32],
    nt: DispatchTime[np.int32],
    ldx: DispatchTime[np.int32],
    e: DispatchTime[np.int32],
    L: CompileTime[int],
    V: CompileTime[int],
    n_aie_cols: CompileTime[int],
):
    n_aie_rows = N_AIE_ROWS
    fifo_depth = 2
    tile_ty = np.ndarray[(L,), np.dtype[np.float32]]
    col_ty = np.ndarray[(L * n_aie_rows,), np.dtype[np.float32]]
    otile_ty = np.ndarray[(L,), np.dtype[bfloat16]]
    ocol_ty = np.ndarray[(L * n_aie_rows,), np.dtype[bfloat16]]
    params_ty = np.ndarray[(16,), np.dtype[np.float32]]
    rtp_ty = np.ndarray[(4,), np.dtype[np.int32]]

    flags = [f"-DSNAKE_TILE={L}"]
    inc = [str(HERE)]
    k_eval = ExternalFunction("snake_eval_f32", source_file=str(HERE / "snake_eval.cc"), arg_types=[tile_ty, tile_ty, params_ty], compile_flags=flags, include_dirs=inc)
    k_emit = ExternalFunction(
        "snake_emit_bf16", source_file=str(HERE / "snake_emit.cc"), arg_types=[tile_ty, tile_ty, otile_ty, params_ty, np.int32], compile_flags=flags, include_dirs=inc
    )
    k_zero = ExternalFunction("zero_f32", source_file=str(HERE / "snake_zero.cc"), arg_types=[tile_ty], compile_flags=flags, include_dirs=inc)
    k_copy = ExternalFunction("copy_f32", source_file=str(HERE / "snake_copy.cc"), arg_types=[tile_ty, tile_ty], compile_flags=flags, include_dirs=inc)

    dev = iron.get_current_device()
    assert dev is not None
    if n_aie_cols > dev.cols:
        raise ValueError(f"n_aie_cols={n_aie_cols} but the device has {dev.cols} columns")

    x_l3l2, x_l2l1, z_l2l3, z_l1l2 = [], [], [], []
    for col in range(n_aie_cols):
        xin = ObjectFifo(col_ty, name=f"X_L3L2_{col}", depth=fifo_depth)
        x_l3l2.append(xin)
        x_l2l1.append(
            xin.cons().split(
                [L * r for r in range(n_aie_rows)],
                obj_types=[tile_ty] * n_aie_rows,
                names=[f"X_L2L1_{col}_{r}" for r in range(n_aie_rows)],
                depths=[fifo_depth] * n_aie_rows,
            )
        )
        zout = ObjectFifo(ocol_ty, name=f"Z_L2L3_{col}", depth=fifo_depth)
        z_l2l3.append(zout)
        z_l1l2.append(
            zout.prod().join(
                [L * r for r in range(n_aie_rows)],
                obj_types=[otile_ty] * n_aie_rows,
                names=[f"Z_L1L2_{col}_{r}" for r in range(n_aie_rows)],
                depths=[fifo_depth] * n_aie_rows,
            )
        )

    rtps = [[Buffer(rtp_ty, name=f"rtp_{r}_{c}", use_write_rtp=True) for c in range(n_aie_cols)] for r in range(n_aie_rows)]
    params = [[Buffer(params_ty, name=f"params_{r}_{c}") for c in range(n_aie_cols)] for r in range(n_aie_rows)]
    y_prev = [[Buffer(tile_ty, name=f"yprev_{r}_{c}") for c in range(n_aie_cols)] for r in range(n_aie_rows)]
    y_cur = [[Buffer(tile_ty, name=f"ycur_{r}_{c}") for c in range(n_aie_cols)] for r in range(n_aie_rows)]
    barriers = [[WorkerRuntimeBarrier() for _ in range(n_aie_cols)] for _ in range(n_aie_rows)]

    def core_fn(in_x, out_z, f_eval, f_emit, f_zero, f_copy, rtp, prm, yp, yc, barrier):
        barrier.wait_for_value(1)
        n_groups = rtp[0]
        n_chunks = rtp[1]
        n_tiles = rtp[2]
        barrier.release_with_value(1)

        def take_params():
            p = in_x.acquire(1)
            for i in range(2 + 2 * 6):
                prm[i] = p[i]
            in_x.release(1)

        def step():
            # one data tile: evaluate into yc, emit variants of yp (lookahead = yc), then yp <- yc
            x = in_x.acquire(1)
            f_eval(x, yc, prm)
            in_x.release(1)
            for v in range(V):
                o = out_z.acquire(1)
                f_emit(yp, yc, o, prm, v)
                out_z.release(1)
            f_copy(yc, yp)

        for _ in range_(n_groups):
            # first chunk of the row: params, then the first tile primes yp
            take_params()
            x0 = in_x.acquire(1)
            f_eval(x0, yp, prm)
            in_x.release(1)
            for _ in range_(n_tiles - 1):
                step()
            for _ in range_(n_chunks - 1):
                take_params()
                for _ in range_(n_tiles):
                    step()
            # flush the last tile with zero lookahead
            f_zero(yc)
            for v in range(V):
                o = out_z.acquire(1)
                f_emit(yp, yc, o, prm, v)
                out_z.release(1)

    workers = Worker.grid(
        n_aie_rows,
        n_aie_cols,
        lambda row, col: Worker(
            core_fn,
            [
                x_l2l1[col][row].cons(), z_l1l2[col][row].prod(), k_eval, k_emit, k_zero, k_copy,
                rtps[row][col], params[row][col], y_prev[row][col], y_cur[row][col], barriers[row][col],
            ],
            stack_size=0x1000,
        ),
    )
    flat_workers = [w for row in workers for w in row]
    x_prods = [f.prod() for f in x_l3l2]
    z_conses = [f.cons() for f in z_l2l3]

    def sequence(X, P, Z, C, T, nt, ldx, e, X_hs, Z_hs):
        rows_per_group = n_aie_rows * n_aie_cols
        require(C > 0, "C must be positive")
        require(T > 0, "T must be positive")
        require(C % rows_per_group == 0, "C must be a multiple of 4 * n_aie_cols")
        require(T % L == 0, "T must be a multiple of L")
        require(nt > 0, "nt must be positive")
        require(nt <= 64, "nt must be <= 64")
        require((T // L) % nt == 0, "T / L must be a multiple of nt")
        require(e % 2 == 0, "e must be even")
        require(ldx % 2 == 0, "ldx must be even")
        require(ldx >= e + T, "ldx must cover e + T")
        n_groups = C // rows_per_group
        n_tiles = T // L
        n_chunks = n_tiles // nt
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                rtps[row][col][0] = n_groups
                rtps[row][col][1] = n_chunks
                rtps[row][col][2] = nt
                barriers[row][col].set(1)

        for step, tg in TaskGroup.pipelined(n_groups * n_chunks, depth=2):
            grp = step // n_chunks
            ch = step % n_chunks
            for col in range(n_aie_cols):
                c0 = grp * rows_per_group + col * n_aie_rows
                p_tap = TensorAccessPattern((C * L,), c0 * L, [n_aie_rows, L], [L, 1])
                X_hs[col].fill(P, tap=p_tap, group=tg)
                x_tap = TensorAccessPattern((C * T,), c0 * T + ch * nt * L, [nt, n_aie_rows, L], [L, T, 1])
                X_hs[col].fill(X, tap=x_tap, group=tg)
                # Z rows c*V + v, columns [e, e + T): per chunk [nt tiles, V variants, 4 rows, L]
                z_tap = TensorAccessPattern(
                    (C * V * ldx,),
                    c0 * V * ldx + e + ch * nt * L,
                    [nt, V, n_aie_rows, L],
                    [L, ldx, V * ldx, 1],
                )
                Z_hs[col].drain(Z, tap=z_tap, wait=True, group=tg)

    rt = Runtime(sequence, [X, P, Z, C, T, nt, ldx, e, x_prods, z_conses])
    return Program(dev, rt, workers=flat_workers).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="NPU fused snake + conv packing")
    add_compile_args(p, short_dev=None)
    p.add_argument("-L", type=int, default=1008)
    p.add_argument("-V", type=int, default=4, choices=[2, 4, 6])
    p.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=8)
    p.add_argument("--x-elems", type=int, default=32 * 1024 * 1024)
    p.add_argument("--p-elems", type=int, default=1024 * 1024)
    p.add_argument("--z-elems", type=int, default=32 * 1024 * 1024)
    return p


def _compile_kwargs(opts):
    return dict(
        X=np.ndarray[(opts.x_elems,), np.dtype[np.float32]],
        P=np.ndarray[(opts.p_elems,), np.dtype[np.float32]],
        Z=np.ndarray[(opts.z_elems,), np.dtype[bfloat16]],
        L=opts.L,
        V=opts.V,
        n_aie_cols=opts.n_aie_cols,
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        snake_pack,
        opts,
        compile_kwargs=_compile_kwargs,
        device=lambda o: from_name(o.dev, n_cols=o.n_aie_cols if o.dev == "npu" else None),
    )


if __name__ == "__main__":
    main()
