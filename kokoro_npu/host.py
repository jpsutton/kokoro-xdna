# SPDX-License-Identifier: Apache-2.0
"""ctypes bindings for build/libnpu_host.so (vectorized packing helpers)."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parent.parent / "build" / "libnpu_host.so"
_lib = ctypes.CDLL(str(_LIB_PATH))

_fp = ctypes.POINTER(ctypes.c_float)
_u16p = ctypes.POINTER(ctypes.c_uint16)
_i64 = ctypes.c_int64

_i64p = ctypes.POINTER(ctypes.c_int64)
_lib.npu_pack_conv_z.argtypes = [_fp, _i64, _i64, _i64, _i64, _i64, _i64, _i64, _i64, _i64p, _i64p, _u16p]
_lib.npu_pack_conv_z.restype = None
for _f in (_lib.npu_copy_rows, _lib.npu_add_rows, _lib.npu_copy_rows_pad):
    _f.argtypes = [_fp, _i64, _fp, _i64, _i64, _i64]
    _f.restype = None


def _fptr(a: np.ndarray):
    return a.ctypes.data_as(_fp)


def pack_conv_z(x: np.ndarray, cin_pad: int, ldx: int, pad_l: int, dil: int, variants, z: np.ndarray):
    """x [cin, t_in] f32 (rows contiguous, any row stride) -> z [nvar*cin_pad, ldx] (uint16 view of bf16)."""
    assert x.dtype == np.float32 and x.strides[1] == 4
    cin, t_in = x.shape
    shifts = np.array([v[0] for v in variants], dtype=np.int64)
    parts = np.array([v[1] for v in variants], dtype=np.int64)  # activation part
    _lib.npu_pack_conv_z(
        _fptr(x), x.strides[0] // 4, cin, t_in, cin_pad, ldx, pad_l, dil, len(variants),
        shifts.ctypes.data_as(_i64p), parts.ctypes.data_as(_i64p), z.ctypes.data_as(_u16p),
    )


def copy_rows(src: np.ndarray, dst: np.ndarray, rows: int, cols: int):
    _lib.npu_copy_rows(_fptr(src), src.strides[0] // 4, _fptr(dst), dst.strides[0] // 4, rows, cols)


def add_rows(src: np.ndarray, dst: np.ndarray, rows: int, cols: int):
    _lib.npu_add_rows(_fptr(src), src.strides[0] // 4, _fptr(dst), dst.strides[0] // 4, rows, cols)


def copy_rows_pad(src: np.ndarray, dst: np.ndarray, rows: int, cols: int):
    """dst[:, :cols] = src[:, :cols]; dst[:, cols:dst_ld] = 0."""
    _lib.npu_copy_rows_pad(_fptr(src), src.strides[0] // 4, _fptr(dst), dst.strides[0] // 4, rows, cols)


_lib.npu_zero_strips.argtypes = [_u16p, _i64, _i64, _i64, _i64]
_lib.npu_zero_strips.restype = None


def zero_strips(z: np.ndarray, rows: int, ld: int, e: int, t: int):
    """Zero columns [0, e) and [e + t, ld) of the first `rows` rows of z (uint16 view)."""
    _lib.npu_zero_strips(z.ctypes.data_as(_u16p), rows, ld, e, t)
