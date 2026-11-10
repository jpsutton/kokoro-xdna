# SPDX-License-Identifier: Apache-2.0
"""Kokoro TTS engine: kokoro-onnx front end + ONNX Runtime with NPU-offloaded ops."""

from __future__ import annotations

import ctypes
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "kokoro-onnx" / "src"))

from kokoro_onnx import Kokoro  # noqa: E402
from kokoro_onnx.trim import trim as trim_audio  # noqa: E402

from .graph import ConvSpec, FusedSpec, SnakeSpec, fuse_snake_conv, rewrite, rewrite_stft  # noqa: E402
from .npu_conv import PRECISION_OF, VARIANTS, NpuConv1d  # noqa: E402
from .npu_snake import NpuSnake  # noqa: E402
from .npu_snake_pack import NpuSnakePack  # noqa: E402

log = logging.getLogger("kokoro_npu")

DEFAULT_MODEL = ROOT / "models" / "kokoro-v1.0.onnx"
DEFAULT_VOICES = ROOT / "models" / "voices-v1.0.bin"
DEFAULT_XCLBIN = ROOT / "kernels" / "build" / "conv1d_m32k64n64c8.xclbin"
DEFAULT_SNAKE_XCLBIN = ROOT / "kernels" / "build" / "snake_L1008c8.xclbin"
DEFAULT_SNAKE_PACK_XCLBIN = {V: ROOT / "kernels" / "build" / f"snake_pack_L1008v{V}c8.xclbin" for V in (2, 4, 6)}
DEFAULT_OPS_LIB = ROOT / "build" / "libnpu_ops.so"
DEFAULT_VOICE = "af_heart"
# Measured with tests/precision_sweep.py: the text encoder / F0 / N predictors need bf16x3
# (the model is chaotic there); the vocoder generator is indifferent to activation/weight
# residuals, the rest of the decoder only needs the activation residual.
DEFAULT_PRECISION = "decoder/decoder/generator=bf16,decoder/=bf16x2,bf16x3"
SAMPLE_RATE = 24000

_CALLBACK_T = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int64,
    ctypes.c_int64,
)


STREAM_GAP_END = 0.18  # s of silence after a sentence in streaming mode
STREAM_GAP_MID = 0.08  # ... after a piece that ends on , or ; or runs on
STREAM_MIN_PHONEMES = 30  # merge shorter sentences into their neighbour (one clip per tiny word sounds choppy)
STREAM_MAX_PHONEMES = 400


def split_sentences(phonemes: str, long_splitter) -> list[str]:
    """Split a phoneme string at sentence ends; long sentences go through ``long_splitter``."""
    parts = re.split(r"(?<=[.!?])\s+", phonemes.strip())
    merged, cur = [], ""
    for part in parts:
        cur = f"{cur} {part}".strip() if cur else part
        if len(cur) >= STREAM_MIN_PHONEMES:
            merged.append(cur)
            cur = ""
    if cur:
        if merged and len(cur) < STREAM_MIN_PHONEMES:
            merged[-1] = f"{merged[-1]} {cur}"
        else:
            merged.append(cur)
    out = []
    for m in merged:
        out.extend(long_splitter(m) if len(m) > STREAM_MAX_PHONEMES else [m])
    return [m for m in out if m.strip()]


def parse_precision(setting: str):
    """Precision policy -> f(spec or None) -> precision name.

    ``setting`` is a comma list. A bare name sets the default; ``substr=name`` applies
    to convs whose ONNX node name contains ``substr`` (first match wins). Example:
    ``generator/resblocks=bf16x2,bf16x3``.
    """
    rules, default = [], None
    for tok in setting.split(","):
        tok = tok.strip()
        if "=" in tok:
            pat, name = tok.split("=", 1)
            rules.append((pat, name))
        else:
            default = tok
    if default is None:
        raise ValueError(f"precision setting {setting!r} has no default")
    for name in [default] + [n for _, n in rules]:
        if name not in VARIANTS:
            raise ValueError(f"unknown precision {name!r}")

    def precision_of(spec):
        if spec is not None:
            for pat, name in rules:
                if pat in spec.name:
                    return name
        return default

    return precision_of


class NpuOpTable:
    """Maps NpuOp op_ids to prepared NPU work (weights, geometry cache)."""

    def __init__(self, conv: NpuConv1d, specs: list[ConvSpec], snake: NpuSnake | None = None, snakes: list[SnakeSpec] = (),
                 snake_packs: dict[int, NpuSnakePack] | None = None, fused: list[FusedSpec] = (), precision=None):
        self.conv = conv
        self.snake = snake
        self.snake_packs = snake_packs or {}
        self.precision = precision or (lambda spec: conv.precision)
        self.specs = {s.op_id: s for s in specs}
        self.snake_specs = {s.op_id: s for s in snakes}
        self.fused_specs = {s.op_id: s for s in fused}
        self._wmats: dict[int, list[np.ndarray]] = {}
        self._geom: dict[tuple[int, int], object] = {}
        self._lock = threading.Lock()
        self._tls = threading.local()

    @property
    def errors(self) -> list[str]:
        """NPU op failures raised on the calling thread (ORT runs the op on that thread)."""
        errs = getattr(self._tls, "errors", None)
        if errs is None:
            errs = self._tls.errors = []
        return errs

    def _fused_geom(self, g):
        """Recompute a conv geometry with the Z row length the fused snake needs."""
        _, _, min_ldx = self.snake_packs[g.nvar].plan(g)
        return self.conv.geometry(g.cout, g.cin, g.taps, g.dil, g.t_in, g.pad_l, g.pad_r, min_ldx=min_ldx, precision=PRECISION_OF[g.variants])

    def _prepared(self, spec: ConvSpec, t: int, fused: bool = False):
        key = (spec.op_id, t, fused)
        g = self._geom.get(key)
        if g is None:
            g = self.conv.geometry(spec.cout, spec.cin, spec.taps, spec.dil, t, spec.pad_l, spec.pad_r, precision=self.precision(spec))
            if fused:
                g = self._fused_geom(g)
            self._geom[key] = g
        wm = self._wmats.get(spec.op_id)
        if wm is None:
            wm = self.conv.prepare_weight(spec.weight, g)
            self._wmats[spec.op_id] = wm
        return g, wm

    def preload(self):
        """Prepare + upload every weight once (geometry needs a T; use a dummy one)."""
        for spec in self.specs.values():
            g, wm = self._prepared(spec, 1024)
            self.conv.weight_bos(spec.op_id, wm)
        for sp in self.snake_specs.values():
            self.snake.params_bo(sp.op_id, sp.alpha, sp.beta)
        for fs in self.fused_specs.values():
            g, wm = self._prepared(fs.conv, 1024, fused=True)
            self.conv.weight_bos(fs.conv.op_id, wm)

    def __call__(self, op_id, x_ptr, c, t, y_ptr, cout, tout) -> int:
        try:
            x = np.ctypeslib.as_array(x_ptr, shape=(c, t))
            y = np.ctypeslib.as_array(y_ptr, shape=(cout, tout))
            sp = self.snake_specs.get(op_id)
            if sp is not None:
                self.snake.run(x, sp.op_id, sp.alpha, sp.beta, out=y)
                return 0
            fs = self.fused_specs.get(op_id)
            if fs is not None:
                spec = fs.conv
                g, wm = self._prepared(spec, t, fused=True)
                if g.t_out != tout or spec.cin != c:
                    raise RuntimeError(f"{spec.name}: shape mismatch c={c} t={t} tout={tout} geom={g}")
                snake = fs.snake

                def producer(xs, gg, ci0, cin_pad):
                    return self.snake_packs[gg.nvar].produce(xs, snake.op_id, snake.alpha, snake.beta, gg, self.conv.bo_b, cin_pad, ci0)

                self.conv.run(x, spec.op_id, wm, g, out=y, z_producer=producer, geom_adjust=self._fused_geom, bias=spec.bias)
                return 0
            spec = self.specs[op_id]
            g, wm = self._prepared(spec, t)
            if g.t_out != tout or spec.cin != c:
                raise RuntimeError(f"{spec.name}: shape mismatch c={c} t={t} tout={tout} geom={g}")
            self.conv.run(x, spec.op_id, wm, g, out=y, bias=spec.bias)
            return 0
        except Exception as e:  # never let an exception cross the C boundary
            log.exception("NpuOp %s failed", op_id)
            self.errors.append(repr(e))
            return 1


class KokoroNPU:
    def __init__(
        self,
        model_path=DEFAULT_MODEL,
        voices_path=DEFAULT_VOICES,
        xclbin=DEFAULT_XCLBIN,
        snake_xclbin=DEFAULT_SNAKE_XCLBIN,
        ops_lib=DEFAULT_OPS_LIB,
        use_npu: bool = True,
        use_snake: bool = True,
        fuse: bool | None = None,
        precision: str | None = None,
        name_filter=None,
        intra_op_threads: int | None = None,
        allow_spinning: bool = False,
    ):
        self.use_npu = use_npu
        so = ort.SessionOptions()
        if intra_op_threads:
            so.intra_op_num_threads = intra_op_threads
        # ORT worker threads busy-spin between parallel sections by default; that
        # is pure wasted CPU while the NPU works, so turn it off.
        so.add_session_config_entry("session.intra_op.allow_spinning", "1" if allow_spinning else "0")
        self.table = None
        if use_npu:
            t0 = time.time()
            model = onnx.load(str(model_path))
            model, specs, snakes = rewrite(model, cout_mult=128, name_filter=name_filter, snake=use_snake)
            n_stft = rewrite_stft(model) if os.environ.get("KOKORO_NPU_STFT", "0") == "1" else 0  # ORT's STFT kernel is ~20x slower than the equivalent strided Conv
            precision = precision or os.environ.get("KOKORO_NPU_PRECISION", DEFAULT_PRECISION)
            precision_of = parse_precision(precision)
            self.conv = NpuConv1d(str(xclbin), m=32, k=64, n=64, cols=8, precision=precision_of(None))
            self.snake = NpuSnake(str(snake_xclbin), L=1008, cols=8, device=self.conv.dev) if use_snake else None
            if fuse is None:
                fuse = os.environ.get("KOKORO_NPU_FUSE", "1") != "0"
            fused = fuse_snake_conv(model, specs, snakes) if (use_snake and fuse) else []
            self.snake_packs = {}
            for V in sorted({len(VARIANTS[precision_of(fs.conv)]) for fs in fused}):
                self.snake_packs[V] = NpuSnakePack(str(DEFAULT_SNAKE_PACK_XCLBIN[V]), L=1008, V=V, cols=8, device=self.conv.dev)
            self.table = NpuOpTable(self.conv, specs, self.snake, snakes, self.snake_packs, fused, precision_of)
            self.table.preload()
            self._lib = ctypes.CDLL(str(ops_lib))
            self._cb = _CALLBACK_T(self.table)  # keep a reference: C holds the pointer
            self._lib.npu_ops_set_callback(self._cb)
            so.register_custom_ops_library(str(ops_lib))
            sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
            log.info("NPU session ready: %d convs + %d snakes offloaded (%d fused snake+conv), %d STFT rewritten, precision=%s (%.1fs)", len(specs), len(snakes), len(fused), n_stft, precision, time.time() - t0)
        else:
            sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.sess = sess
        # kokoro-onnx's from_session() validates session._model_path (an ORT
        # private attribute, None for sessions built from bytes).
        if getattr(sess, "_model_path", None) is None:
            sess._model_path = str(model_path)
        self.kokoro = Kokoro.from_session(sess, str(voices_path))
        # espeak (phonemizer) is not thread-safe; everything after it is, so concurrent
        # requests overlap their CPU-side ops with each other's NPU time.
        self._phonemize_lock = threading.Lock()
        tok = self.kokoro.tokenizer
        phonemize = tok.phonemize

        def locked_phonemize(*a, **kw):
            with self._phonemize_lock:
                return phonemize(*a, **kw)

        tok.phonemize = locked_phonemize

    @property
    def voices(self) -> list[str]:
        return self.kokoro.get_voices()

    def synthesize(self, text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0, lang: str = "en-us"):
        """Returns (audio float32 [-1,1], sample_rate)."""
        if self.table is not None:
            self.table.errors.clear()
        audio, sr = self.kokoro.create(text, voice=voice, speed=speed, lang=lang)
        if self.table is not None and self.table.errors:
            raise RuntimeError("NPU op errors: " + "; ".join(self.table.errors))
        return audio.astype(np.float32), sr

    def synthesize_stream(self, text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0, lang: str = "en-us"):
        """Yield float32 audio chunks (sentence by sentence) as they are generated.

        The first chunk is ready after one sentence instead of the whole text. Sentences
        are synthesized separately, so a short silence is inserted between them (the
        batch path of ``synthesize`` lets the model place its own pauses).
        """
        k = self.kokoro
        style = k.get_voice_style(voice)
        phonemes = k.tokenizer.phonemize(text, lang)  # already serialized by the locked wrapper
        pieces = split_sentences(phonemes, k._split_phonemes)
        for i, piece in enumerate(pieces):
            if self.table is not None:
                self.table.errors.clear()
            audio, sr = k._create_audio(piece, style, speed)
            if self.table is not None and self.table.errors:
                raise RuntimeError("NPU op errors: " + "; ".join(self.table.errors))
            audio, _ = trim_audio(audio)
            yield audio.astype(np.float32)
            if i + 1 < len(pieces):
                yield np.zeros(int(sr * (STREAM_GAP_END if piece[-1:] in ".!?" else STREAM_GAP_MID)), dtype=np.float32)

    def npu_stats(self) -> dict:
        if self.table is None:
            return {}
        d = {"conv_" + k: v for k, v in self.conv.stats.items()}
        if self.snake is not None:
            d.update({"snake_" + k: v for k, v in self.snake.stats.items()})
        for sp in getattr(self, "snake_packs", {}).values():
            for k, v in sp.stats.items():
                d["fused_" + k] = d.get("fused_" + k, 0) + v
        return d
