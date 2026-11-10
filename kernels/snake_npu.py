# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Snake activation on AMD XDNA NPU via IRON (mlir-aie).
#
# Data movement derived from mlir-aie programming_examples (whole_array split/join).
#
"""y[c, t] = x[c, t] + beta[c] * sin(alpha[c] * x[c, t])^2, f32, X is [C, T].

Each compute core owns whole channel rows: core (row, col) of the 4 x cols
grid processes channels c = grp*4*cols + col*4 + row for grp in [0, C/(4*cols)).
A row is streamed as one params tile (p[0] = alpha, p[1] = beta, rest unused,
read from the host buffer P[C, L]) followed by T/L data tiles of L floats.
``C`` and ``T`` are DispatchTime; ``L`` (tile) and ``n_aie_cols`` CompileTime.
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

N_AIE_ROWS = 4
HERE = Path(__file__).resolve().parent


@iron.jit
def snake(
    X: In,
    P: In,
    Y: Out,
    *,
    C: DispatchTime[np.int32],  # channels (multiple of 4 * n_aie_cols)
    T: DispatchTime[np.int32],  # frames per channel (multiple of L)
    L: CompileTime[int],
    n_aie_cols: CompileTime[int],
):
    n_aie_rows = N_AIE_ROWS
    fifo_depth = 2
    tile_ty = np.ndarray[(L,), np.dtype[np.float32]]
    col_ty = np.ndarray[(L * n_aie_rows,), np.dtype[np.float32]]
    params_ty = np.ndarray[(16,), np.dtype[np.float32]]
    rtp_ty = np.ndarray[(2,), np.dtype[np.int32]]

    kernel = ExternalFunction(
        "snake_f32",
        source_file=str(HERE / "snake.cc"),
        arg_types=[tile_ty, tile_ty, params_ty],
        compile_flags=[f"-DSNAKE_TILE={L}"],
    )

    dev = iron.get_current_device()
    assert dev is not None
    if n_aie_cols > dev.cols:
        raise ValueError(f"n_aie_cols={n_aie_cols} but the device has {dev.cols} columns")

    x_l3l2 = []
    x_l2l1 = []  # [col][row]
    y_l2l3 = []
    y_l1l2 = []  # [col][row]
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
        yout = ObjectFifo(col_ty, name=f"Y_L2L3_{col}", depth=fifo_depth)
        y_l2l3.append(yout)
        y_l1l2.append(
            yout.prod().join(
                [L * r for r in range(n_aie_rows)],
                obj_types=[tile_ty] * n_aie_rows,
                names=[f"Y_L1L2_{col}_{r}" for r in range(n_aie_rows)],
                depths=[fifo_depth] * n_aie_rows,
            )
        )

    rtps = [
        [Buffer(rtp_ty, name=f"rtp_{row}_{col}", use_write_rtp=True) for col in range(n_aie_cols)]
        for row in range(n_aie_rows)
    ]
    params = [
        [Buffer(params_ty, name=f"params_{row}_{col}") for col in range(n_aie_cols)]
        for row in range(n_aie_rows)
    ]
    barriers = [[WorkerRuntimeBarrier() for _ in range(n_aie_cols)] for _ in range(n_aie_rows)]

    def core_fn(in_x, out_y, fn, rtp, prm, barrier):
        barrier.wait_for_value(1)
        n_groups = rtp[0]
        n_tiles = rtp[1]
        barrier.release_with_value(1)
        for _ in range_(n_groups):
            p = in_x.acquire(1)
            prm[0] = p[0]
            prm[1] = p[1]
            in_x.release(1)
            for _ in range_(n_tiles):
                x = in_x.acquire(1)
                y = out_y.acquire(1)
                fn(x, y, prm)
                in_x.release(1)
                out_y.release(1)

    workers = Worker.grid(
        n_aie_rows,
        n_aie_cols,
        lambda row, col: Worker(
            core_fn,
            [x_l2l1[col][row].cons(), y_l1l2[col][row].prod(), kernel, rtps[row][col], params[row][col], barriers[row][col]],
            stack_size=0x1000,
        ),
    )
    flat_workers = [w for row in workers for w in row]

    x_prods = [f.prod() for f in x_l3l2]
    y_conses = [f.cons() for f in y_l2l3]

    def sequence(X, P, Y, C, T, X_hs, Y_hs):
        rows_per_group = n_aie_rows * n_aie_cols
        require(C > 0, "C must be positive")
        require(T > 0, "T must be positive")
        require(C % rows_per_group == 0, "C must be a multiple of 4 * n_aie_cols")
        require(T % L == 0, "T must be a multiple of L")
        n_groups = C // rows_per_group
        n_tiles = T // L
        require(n_tiles <= 1023, "T / L must be <= 1023")
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                rtps[row][col][0] = n_groups
                rtps[row][col][1] = n_tiles
                barriers[row][col].set(1)

        for grp, tg in TaskGroup.pipelined(n_groups, depth=2):
            for col in range(n_aie_cols):
                c0 = grp * rows_per_group + col * n_aie_rows
                p_tap = TensorAccessPattern((C * L,), c0 * L, [n_aie_rows, L], [L, 1])
                X_hs[col].fill(P, tap=p_tap, group=tg)
                x_tap = TensorAccessPattern((C * T,), c0 * T, [n_tiles, n_aie_rows, L], [L, T, 1])
                X_hs[col].fill(X, tap=x_tap, group=tg)
                Y_hs[col].drain(Y, tap=x_tap, wait=True, group=tg)

    rt = Runtime(sequence, [X, P, Y, C, T, x_prods, y_conses])
    return Program(dev, rt, workers=flat_workers).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="NPU snake activation")
    add_compile_args(p, short_dev=None)
    p.add_argument("-L", type=int, default=1024)
    p.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
    p.add_argument("--x-elems", type=int, default=32 * 1024 * 1024)
    p.add_argument("--p-elems", type=int, default=1024 * 1024)
    return p


def _compile_kwargs(opts):
    return dict(
        X=np.ndarray[(opts.x_elems,), np.dtype[np.float32]],
        P=np.ndarray[(opts.p_elems,), np.dtype[np.float32]],
        Y=np.ndarray[(opts.x_elems,), np.dtype[np.float32]],
        L=opts.L,
        n_aie_cols=opts.n_aie_cols,
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        snake,
        opts,
        compile_kwargs=_compile_kwargs,
        device=lambda o: from_name(o.dev, n_cols=o.n_aie_cols if o.dev == "npu" else None),
    )


if __name__ == "__main__":
    main()
