# kokoro-xdna: Kokoro TTS on the AMD XDNA2 NPU (Strix Halo)

Kokoro-82M text-to-speech served through an OpenAI-compatible `/v1/audio/speech`
endpoint, with the heavy part of the vocoder (iSTFTNet generator) executed on the
AMD Ryzen AI NPU via custom MLIR-AIE (IRON) kernels. Goal: move CPU load to the
NPU, not raw speed.

Status: works end to end on one machine (Ryzen AI MAX+ 395, Arch Linux, XRT 2.21,
kernel 7.2 `amdxdna`). See [Requirements and limits](#requirements-and-limits) before
trying another setup.

Started from [Unicorn-Commander/magic-unicorn-tts](https://github.com/Unicorn-Commander/magic-unicorn-tts)
and [Unicorn-Orator](https://github.com/Unicorn-Commander/Unicorn-Orator); the
upstream "NPU acceleration" turned out to be a CPU fallback (see
[UPSTREAM_BUGS.md](UPSTREAM_BUGS.md)), so the NPU path here is new.

## What runs where

ORT profile of Kokoro on CPU: `Sin` 33% + `Conv` 29% + Snake elementwise ~20% of
the time, all in `/decoder/decoder/generator`. Those are now NPU custom ops:

| Op (ONNX)                                      | Count | NPU kernel                                              |
|------------------------------------------------|------:|---------------------------------------------------------|
| `Conv` 1-D, stride 1, group 1, odd dilation (+bias) | 81/88 | `kernels/conv1d_npu.py` (bf16 matmul, DMA im2col, bias init) |
| Snake `Mul→Sin→Pow→Mul→Add` → feeding a conv   | 48/48 | `kernels/snake_pack_npu.py` fused: snake writes the conv's bf16 input tiles |

Everything else (BERT, LSTM, STFT, ConvTranspose, the remaining elementwise ops)
stays on the ONNX Runtime CPU EP.

Result for a 7.8 s af_heart utterance (Ryzen AI MAX+ 395, spin-wait off, `tests/bench.py`):

| path                                  | ORT threads | wall    | RTF   | process CPU time |
|---------------------------------------|------------:|--------:|------:|-----------------:|
| CPU only                              | 4           | 0.95 s  | 0.122 | 2.81 s           |
| CPU only                              | 2           | 1.40 s  | 0.180 | 2.52 s           |
| NPU, default per-stage precision      | 2           | 1.08 s  | 0.139 | 0.63 s           |
| NPU, default per-stage precision      | 1           | 1.22 s  | 0.157 | 0.57 s           |
| NPU, uniform `bf16x3`                 | 2           | 1.68 s  | 0.217 | 0.71 s           |

About 4x less CPU time than the CPU path at equal wall time. Of the remaining
~0.6 CPU-s, ~0.05 is the f32 copy out of the XRT output buffer, ~0.2 is the
host side of the NPU ops (Z packing, dispatch), and the rest is ONNX Runtime's
own elementwise ops, LSTM, BERT matmuls, ConvTranspose and STFT.
(ORT's default of 32 spinning threads burned 28 CPU-s per utterance; turning the
spin-wait off alone brought the CPU path from 7.4 to 2.8 CPU-s.)

### Precision

`KOKORO_NPU_PRECISION` is a comma list: a bare name is the default, `substr=name`
applies to convs whose ONNX node name contains `substr` (first match wins).
Names: `bf16` rounds conv activations and weights to bf16; `bf16x2` splits
activations into bf16 hi+lo (2 products); `bf16x3` also adds the weight residual
(3 products, each a separate K block of the same matmul, so no extra kernels).
All three use the same conv xclbin; only the packed Z/weight layout differs.

The default is `decoder/decoder/generator=bf16,decoder/=bf16x2,bf16x3`. It comes
from `tests/precision_sweep.py` (spectrogram correlation and log-spectral distance
against the CPU path, two texts of 8 s and 32 s, voices af_heart / bm_george /
af_bella):

| policy                                  | 8 s wall | 32 s wall | corr (32 s) | LSD, af_heart 8 s / 32 s |
|-----------------------------------------|---------:|----------:|------------:|-------------------------:|
| uniform `bf16x3`                        | 1.66 s   | 6.3 s     | 0.9993      | 1.46 / 1.39 dB           |
| uniform `bf16x2`                        | 1.44 s   | 5.3 s     | 0.9969      | 2.05 / 2.98 dB           |
| generator `bf16x2`, rest `bf16x3`       | 1.47 s   | 5.5 s     | 0.9993      | 1.47 / 1.39 dB           |
| generator `bf16`, decoder `bf16x2`, text encoder / F0 / N `bf16x3` (default) | 1.06 s | 3.9 s | 0.9993 | 1.49 / 1.42 dB |
| decoder all `bf16`, rest `bf16x3`       | 1.06 s   | 3.7 s     | 0.9984      | 1.51 / 1.47 dB           |

Precision only matters in the text encoder and the F0/N predictors (they feed a
chaotic path); the vocoder is indifferent to the residual terms. The model itself
is chaotic at the ~1.2-1.4 dB level: perturbing the CPU path's STFT output by 1e-7
already gives LSD 1.4 dB. Samples in `samples/`.

Concurrent requests overlap: only phonemization (espeak) is serialized, so one
request's CPU-side ops run while another uses the NPU (+12-15% throughput with 2-3
clients; output is bit-identical to serial).

## Layout

```
server.py              FastAPI: POST /v1/audio/speech, GET /v1/audio/voices, /v1/models, /health
kokoro_npu/engine.py   KokoroNPU: graph rewrite + ORT session + custom-op dispatch
kokoro_npu/graph.py    ONNX rewrite: Conv -> NpuOp(+Add bias); Snake chain -> NpuOp
kokoro_npu/npu_conv.py XRT runtime for conv1d (pyxrt + dispatch .so via ctypes)
kokoro_npu/npu_snake.py XRT runtime for the standalone snake (used only with KOKORO_NPU_FUSE=0)
kokoro_npu/npu_snake_pack.py XRT runtime for the fused snake -> conv input packing
kokoro_npu/host.py     ctypes bindings for build/libnpu_host.so (packing helpers)
csrc/npu_ops.cc        ORT custom op "kokoro.npu:NpuOp" -> C callback (zero-copy)
csrc/npu_host.cc       f32->bf16 packing / row copies
kernels/conv1d_npu.py  IRON design: C = Wmat @ im2col(Z) with the im2col done by the shim DMA
kernels/snake_npu.py   IRON design: standalone per-channel snake (f32 in/out)
kernels/snake_pack_npu.py IRON design: snake -> bf16 Z row variants (plain/dil-shifted, hi/lo)
kernels/snake_*.cc, snake_common.h  AIE2P kernels: f32 sin (range reduction + polynomial), bf16 emit with lookahead shift
kernels/conv_bias_*.cc AIE2P kernels: C tile init from a bias vector (bias rides the A fifo)
kernels/build/         xclbin + dispatch-*.so (built, not checked in)
third_party/kokoro-onnx  vendored kokoro-onnx 0.4.9 (tokenizer, voices, chunking)
tests/                 kernel validation, parity, benchmark
ref/                   upstream clones (reference only)
```

## Setup (Arch Linux, XRT 2.21 from the `xrt` package, kernel 7.2 `amdxdna`)

```bash
git clone --recurse-submodules <this repo> && cd <this repo>   # kokoro-onnx is a submodule
# 1. memlock: XRT mmaps the 64 MB device heap with MAP_LOCKED. Add to /etc/security/limits.conf
#    and re-login:    <user>  -  memlock  unlimited
# 2. compile-time toolchain (Python 3.12 venv with mlir-aie v1.4.4 wheel + Peano)
scripts/setup_ironenv.sh
# 3. runtime venv (Python 3.13: onnxruntime, pyxrt built from XRT source, fastapi)
scripts/setup_venv.sh
# 4. models + kernels
scripts/download_models.sh
scripts/build_kernels.sh      # ~10 s
# 5. run
scripts/serve.sh               # http://127.0.0.1:8880, no API key; --cpu, --host, --port, --threads
```

```bash
curl -X POST localhost:8880/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"Hello from the NPU.","voice":"af_heart","response_format":"wav"}' -o out.wav
```

`response_format`: `wav`, `flac`, `mp3`, `opus` (Ogg), `aac` (needs ffmpeg), `pcm` (s16le mono 24 kHz). OpenAI voice names (alloy, nova, ...) map to Kokoro voices; null fields and extra fields (`instructions`, ...) are accepted. Env: `KOKORO_NPU=0`
(CPU only), `KOKORO_NPU_PRECISION=<policy>` (see Precision), `KOKORO_NPU_FUSE=0` (separate
snake kernel), `KOKORO_NPU_STFT=1` (replace ORT's STFT by an equivalent strided Conv:
−25 ms CPU per utterance, but see below), `ORT_THREADS` (default 2), `HOST`, `PORT`.

### Streaming

By default the whole text is synthesized, encoded and returned in one response. With
streaming on, audio is generated sentence by sentence and sent as chunked transfer
as soon as each sentence is ready, so the first audio arrives after ~0.6 s whatever
the text length (a 24 s passage: first byte 0.6 s, total 3.7 s; non-streaming sends
nothing for 2.7 s).

* Per request: `"stream": true` or OpenAI's `"stream_format": "audio"`. `"stream": false` forces
  the single-response mode. `"stream_format": "sse"` is rejected (400).
* Server-wide default for clients that cannot send the flag (Patchbay): `scripts/serve.sh --stream`
  or `STREAM_DEFAULT=1`.
* Works for every `response_format`. `pcm` and `wav` are produced in process (the WAV header
  carries an unknown length, which players accept for streams); `mp3`, `opus`, `aac` and `flac`
  go through one long-running ffmpeg per request so the stream is a single continuous bitstream.
* Costs: sentences are synthesized separately, so total time is ~35% longer than the batched
  path (RTF ~0.15 vs ~0.11), and a fixed 0.18 s (0.08 s after a comma-ended piece) pause is
  inserted between sentences instead of the model's own. Very short sentences are merged
  (< 30 phonemes) to avoid choppy clips.
* A client that disconnects stops generation after the sentence in flight and kills its ffmpeg.
  An error mid-stream ends the body early and is logged (`stream ABORTED`).
* Patchbay reads the whole body (`resp.bytes()`) before playing, so it gains nothing from
  streaming until its client plays chunks as they arrive.

Tests: `venv/bin/python tests/test_conv1d.py kernels/build/conv1d_m32k64n64c8.xclbin 8 64 bf16x3`,
`tests/test_snake_pack.py bf16x3`, `tests/test_snake.py kernels/build/snake_L1008c8.xclbin 1008 8`,
`tests/parity_test.py [text]`, `tests/bench.py`.

## How the conv kernel works

`Y[co,t] = Σ_{j,ci} W[co,ci,j] · Xp[ci, t + j·dil]` is `C = A @ B` with
`A = Wmat[Cout, taps·Cin]` and `B = im2col`. `B` is never materialised: the shim
DMA walks the padded input with a 4-D access pattern `[n-tile, tap, ci, t]` whose
tap stride is the dilation. Two hardware constraints shaped the design:

* DMA offsets/strides are 4-byte granules, so a bf16 stream cannot be shifted by an
  odd element count — and every Kokoro dilation is odd. Taps are paired: the host
  stores `Z = [Xp ; Xp shifted by dil]` (2·Cin rows) and the kernel runs
  `ceil(taps/2)` "super-taps" of stride `2·dil`.
* DMA dimension limits (iteration ≤ 64, d1/d0 ≤ 1023, d0 contiguous) → column
  tiles are issued in chunks of ≤ 64, and `2·cin_pad ≤ 1023` (convs with more input
  channels are split along Cin and summed on the host; only the small decode-stage
  convs).

All shape parameters are `DispatchTime`: one xclbin per tile config, and the
instruction stream is generated per call by the compiled `dispatch-*.so`
(loaded with ctypes — no mlir-aie at runtime). The compute tiles run the stock
bf16→f32 matmul microkernel; 8 columns × 4 rows; ~1.5 TFLOP/s on the big convs.

## Notes on the remaining CPU time (~0.6 CPU-s per 7.8 s)

* ORT's own ops: residual `Add`s, STFT (54 ms, single-threaded naive DFT in ORT),
  ConvTranspose, LSTM, BERT matmuls, InstanceNorm chains — each ≤ 4% of the
  original time.
* Host side of the NPU ops (~0.2 CPU-s): one f32 copy of each conv input into an
  XRT buffer and one copy of the output back. Avoiding those would need ORT to
  allocate tensors in NPU-visible memory (custom allocator).
* `KOKORO_NPU_STFT=1` replaces the STFT with a strided Conv (exact to 1e-6), but
  the model derives a phase via `Atan(im/re)` + sign branches and ~44% of the bins
  have |re| < 1e-6, so the phase there is rounding noise in *any* implementation;
  ORT's STFT also produces a non-zero DC imaginary part whose sign happens to make
  the exported branch structure agree with `torch.atan2`. The rewrite reproduces
  that for DC/Nyquist, but the other noise bins differ, which costs ~1 dB against
  the ORT reference (not necessarily against the PyTorch original). Off by default.

## Where the NPU time goes (and why it is not faster than the CPU)

Measured, not estimated:

* The conv kernel executes at 2.6–2.9 TFLOP/s bf16 on all 8 columns, the same
  rate as mlir-aie's stock `whole_array` matmul at equal M/K/N and across tile
  configs (m32k64n64/n96, m16k64n128). The DMA-side im2col costs nothing extra;
  the design is compute-bound at the microkernel's ceiling (Peano-compiled
  `aie::mmul<4,8,8>` bf16; the Chess toolchain is reported faster but needs a
  Vitis licence).
* Executed work per utterance is large because of the precision split: K =
  super-taps × V × Cin, so `bf16x3` runs 3× the MACs of `bf16`. For the heaviest
  conv (k=11, d=5, 128 ch, T=49k) that is 38.6 GFLOP → 14.8 ms at peak.
* The 32-core Zen 5 CPU does the same convs in f32 at a comparable effective
  rate, so wall time is roughly equal at `bf16` and ~1.7× longer at `bf16x3`.
  The default per-stage precision keeps `bf16x3` only where it matters, which puts
  the NPU path ahead of the same-thread-count CPU path on wall time as well.

Things tried that did not help end-to-end: bfp16 emulation of the bf16 mmul
(`emulate_bf16_mmul_with_bfp16`: 0-20% faster, but relative error 6e-3 instead of
2e-6 and the residual splits cannot recover it); zero-copy ORT tensors (the output
copy is only ~0.05 CPU-s); dropping the zero half of the last super-tap for odd
tap counts (needs a fifth DMA dimension: the B walk is already n-tile x super-tap x
row x t); 8 vs 4 columns (+20% on big convs),
wider output tiles n=128 with single-buffered C (−10–20% on big convs, lost
again to padding granularity on small ones), row-stride alignment (none).
Weights are already broadcast to all columns per core row, and a sliding-window
conv would only remove the im2col re-read, which is not the bottleneck.

## Known limitations

* `ConvTranspose`, STFT, LSTM, BERT remain on CPU.
* Convs with more than 170 (bf16x3) / 255 (bf16x2) / 511 (bf16) input channels
  are split along Cin (DMA d1 ≤ 1023 rows) and summed on the host; only the
  256-channel resblocks and the small decode-stage convs are affected.
* One conv xclbin, one standalone snake xclbin and up to three fused snake_pack
  xclbins (one per precision variant count, 2/4/6) → up to five hw contexts on the
  same device; no switch penalty was measurable.

## Requirements and limits

* **Hardware**: built and tested only on a Strix Halo NPU (`npu2`, 8 columns x 4 core
  rows, firmware 1.1.2.65). The kernels' tile parameters and the DMA limits they work
  around are Strix-specific. Other XDNA2 parts with fewer columns need `NPU_COLS` and
  likely retuning; XDNA1 (Phoenix/Hawk Point, `npu1`) is untested and would need new
  kernels.
* **Software**: Linux with the `amdxdna` driver and XRT 2.21 (`/dev/accel/accel0`),
  `uv`, Python 3.12 (compile-time toolchain) and 3.13 (runtime), a C++17 compiler,
  boost headers (to build `pyxrt`), espeak-ng (via `espeakng-loader`), and `ffmpeg`
  only for `aac` and for streaming `mp3`/`opus`/`aac`/`flac`.
* **Memlock**: XRT needs `memlock unlimited` (see Setup) or it fails with `mmap ... err=-11`.
* **Precision**: results differ from the CPU path at the ~1.5 dB log-spectral-distance
  level (spectrogram correlation 0.999). The model is chaotic enough that tiny
  perturbations give the same size of difference; see the Precision section.
* **No authentication**: the server has none. Keep it on localhost or a trusted network.

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). `kernels/conv1d_npu.py` and
`kernels/snake_npu.py` are derived from mlir-aie examples and carry
`Apache-2.0 WITH LLVM-exception`. Model weights are downloaded separately and are
Apache-2.0 (Kokoro-82M).
