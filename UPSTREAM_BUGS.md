# Upstream bugs found while building this project, and an upstreaming plan

Status of each upstream as of 2026-10-26 (all unarchived, issues enabled):

| Project | Last push | Activity |
|---|---|---|
| Unicorn-Commander/magic-unicorn-tts | 2026-10-25 (licence PR only; code last touched 2025-07) | low |
| Unicorn-Commander/Unicorn-Orator | 2026-10-25 (licence PR; code 2025-11) | low |
| magicunicorn/kokoro-npu-quantized (Hugging Face) | 2025-07-10 | dormant |
| thewh1teagle/kokoro-onnx | 2026-09-01 | active |
| Xilinx/mlir-aie | daily | very active |
| Xilinx/XRT (and Arch `xrt` package) | weekly | active |

---

## 1. magic-unicorn-tts

### 1.1 Repository does not build: required modules are not in the tree
`kokoro_mlir_integration.py` imports `kokoro_mlir_npu` (class
`KokoroNPUAcceleratorMLIR`) and `web_interface_magic_unicorn.py` imports
`vitisai_onnxruntime_wrapper`. Neither file is in the repository or any public
repository of the organisation. `kokoro_mlir_npu.py` exists in Unicorn-Orator
(`kokoro-tts/`), and that in turn imports `npu.npu_runtime`, which only exists in the
author's personal mirror `mastercda/Unicorn-Orator` (`kokoro-tts/npu/npu_runtime.py`),
never in the org repo. `vitisai_onnxruntime_wrapper` is nowhere.

### 1.2 The "NPU acceleration" is a CPU fallback
`KokoroNPUAcceleratorMLIR.accelerated_inference()` calls the CPU function
unconditionally (`# TODO: Replace with actual MLIR-AIE NPU execution`).
`SimplifiedNPURuntime.run_inference()` returns `np.random.randn(...)` ("dummy
output for testing") and is never called. "NPU ready" means `/dev/accel/accel0`
opened and answered one ioctl. README benchmarks ("13x faster", "RTF 0.213 in
turbo mode") therefore measure onnxruntime on the CPU.

### 1.3 Hard-coded developer paths
`web_interface_magic_unicorn.py` hard-codes `/home/ucadmin/Development/kokoro_npu_project/...`
for the model, voices, kokoro-onnx source, venv interpreter and `PYTHONPATH`;
synthesis spawns `/home/ucadmin/.../venv/bin/python`.

### 1.4 Model repo on Hugging Face is broken
`magicunicorn/kokoro-npu-quantized`: `kokoro-npu-quantized-int8.onnx`,
`kokoro-npu-fp16.onnx` and `voices-v1.0.bin` are 133/134-byte Git-LFS pointer
files; the blobs were never uploaded. `install.sh` downloads these pointers and
reports success. (`voices-v1.0.bin`'s pointer SHA matches the kokoro-onnx release
file, so that one can be substituted.)

### 1.5 `requirements.txt` lists `kokoro-onnx` from PyPI while the code imports it from `./kokoro-onnx/src`
Neither the vendored directory nor a pin exists; API drift between kokoro-onnx
0.4.x and 0.6.x (`create()`, `from_session()`) makes this fragile.

**Plan:** these are not "fixable" by patch in a meaningful way — the NPU path does
not exist. Open one issue on magic-unicorn-tts summarising 1.1–1.5 with a link to
this project as a working alternative, and one issue on the HF repo (LFS pointers).
Not worth a PR unless the maintainers respond; the code has been untouched since
July 2025.

## 2. Unicorn-Orator

### 2.1 `kokoro-tts/npu/npu_runtime.py` missing from the org repository
See 1.1. The file exists only in `mastercda/Unicorn-Orator`. The `xdna2/` tree also
imports `runtime.kokoro_xdna2_runtime` (`KokoroXDNA2Runtime`), referenced in
`KOKORO_NPU_STATUS.md` as "408 lines – NEW!", which was never committed.

### 2.2 "BF16 signed value bug" claim (`xdna2/utils/bf16_workaround.py`) is wrong
Docstring: "AMD XDNA2 NPU's AIE accumulator in `aie::mmul<...bfloat16...>` doesn't
correctly handle signed BF16 values, resulting in 789–2823% accuracy errors." The
stock mlir-aie v1.4.4 bf16→f32 whole-array matmul on this Strix Halo NPU gives a
relative error of 1.7e-7 with ~50% negative inputs (`tests/npu_matmul_test.py`),
and the conv kernel here (signed activations and weights) matches its reference to
3e-7. The error was in their kernel/driver code, not the hardware; the [0,1]
rescaling workaround degrades accuracy for nothing.

### 2.3 `xdna2/runtime/xdna2_runtime.py` is a stub
Every method is a `TODO`; `server.py` imports and "initialises" it.

**Plan:** issue on Unicorn-Orator with 2.1–2.3, offering `kernels/conv1d_npu.py` +
`kokoro_npu/npu_conv.py` from this repo as a drop-in that actually runs on XDNA2.
PR only if asked.

## 3. kokoro-onnx (thewh1teagle)

### 3.1 `Kokoro.from_session()` crashes for sessions not created from a file path
`from_session()` reads the private `session._model_path` and passes it to
`KoKoroConfig(...).validate()`, which calls `Path(model_path).exists()`. For an
`InferenceSession` built from bytes (needed when the graph is rewritten in memory,
or for any custom-op/EP setup) `_model_path` is `None` → `TypeError: argument
should be a str or an os.PathLike object ... not 'NoneType'`. Present on `main`
(2026-09) and in 0.4.9. Workaround used here: `sess._model_path = str(path)`.

Fix: `from_session()` should not validate the model path at all (the session already
proved the model loads); only `voices_path` needs checking. ~5-line PR.

### 3.2 `Kokoro.__init__` crashes when the package is imported from source
`__init__` logs `importlib.metadata.version('kokoro-onnx')`, which raises
`PackageNotFoundError` when the package is used from a checkout (`sys.path`
insert) instead of an installed distribution — which is exactly what
magic-unicorn-tts does. Wrap in try/except or drop the log line.

### 3.3 `requires-python = ">=3.10,<3.14"` upper bound
Nothing in the package needs it; the cap only blocks installs on 3.14 (where
onnxruntime wheels exist). Minor; mention in the same PR.

### 3.4 (model, not code) exported `atan2` disagrees with PyTorch for exact-zero imaginary parts
The Kokoro v1.0 ONNX export computes the source-signal STFT phase as
`Atan(im/re)` with `Where(re<0, Where(im>0, +pi, -pi), ...)` branches. For
`im == 0` exactly (the DC and Nyquist bins of a real STFT) this yields `-pi`
where `torch.atan2(0, re<0)` gives `+pi`. It is masked today only because ORT's
STFT returns a non-zero garbage imaginary part (see 7.1). Any exact STFT
implementation (an FFT, or the strided-Conv rewrite in this repo) flips those
phases. Worth a note to whoever maintains the export (hexgrad/kokoro or
thewh1teagle/kokoro-onnx model files): use `Atan2`-equivalent handling of
signed zeros, or compute phase differently.

**Plan:** PR to thewh1teagle/kokoro-onnx for 3.1–3.3, with a regression test
that builds a session from `model.SerializeToString()`. Active project, likely to
merge. 3.4 as an issue with the reproduction from `kokoro_npu/graph.py:rewrite_stft`.

## 4. mlir-aie (Xilinx) — observations, not clear bugs

### 4.1 `utils/env_install.sh` hard-requires Python 3.12 although wheels exist for 3.10–3.14
The script refuses any interpreter other than 3.12 while the `latest-wheels-*` /
release assets ship cp310–cp314. A system with only 3.13/3.14 must install 3.12
just to run the script. Suggest: warn, don't fail, when a wheel for the running
interpreter exists.

### 4.2 Documentation gaps hit while writing the conv design
Hardware DMA constraints are enforced only at dispatch time with a refusal message
(good) but not documented anywhere near `TensorAccessPattern`: iteration count
≤ 64, repeat ≤ 256, d1/d2 size ≤ 1023, d0 size ≤ 1023, strides in whole 4-byte
granules (so bf16 patterns cannot have odd element strides/offsets). A short
"DMA pattern limits" section in the IRON docs would save a few iterations.

### 4.3 `whole_array` README-recommended `m=64 k=64 n=64` bf16/f32 on npu2 fails buffer placement
`error: 'aie.buffer' op could not be placed: buffer "B_L2L1_3_3_cons_buff_1" needs
8192 bytes and this tile has no room left` with `n_aie_cols=4`. The test suite uses
`64x64x32` for npu2; the README's generic guidance does not say so.

**Plan:** 4.1 → small PR (relax the check to "a wheel exists for this interpreter").
4.2 → docs PR adding the limits table next to `TensorAccessPattern` /
`npu_dma_memcpy_nd` docs. 4.3 → issue. mlir-aie is very active; all three are
cheap and in scope.

## 5. XRT / Arch Linux packaging

### 5.1 `xrt-smi examine` fails with `mmap(... len=67108864 ... flags=8209) failed (err=-11)` under the default memlock limit
XRT maps the 64 MB XDNA device heap with `MAP_LOCKED`; the Arch default
`RLIMIT_MEMLOCK` is 8 MB, so every device open fails with an opaque error. Neither
the package nor XRT prints a hint. Fix options: XRT could emit "raise memlock
limit" on `EAGAIN` from that mmap; the Arch package could ship a
`/etc/security/limits.d/xrt.conf` or a note in the post-install message.

### 5.2 Arch `xrt` package only ships `pyxrt` for the system Python (3.14)
Anything that needs a different interpreter (here: onnxruntime-extensions has no
3.14 wheel; mlir-aie's installer wants 3.12) has to rebuild `pyxrt` from source
(`scripts/build_pyxrt.sh`, one pybind11 translation unit; needs boost headers that
the package does not pull in as a dependency even though `/usr/include/xrt/*.h`
include `boost/format.hpp` and `boost/any.hpp`).

**Plan:** 5.1 → XRT issue (better error message) + Arch package bug (limits.d
snippet or install note). 5.2 → Arch package bug: add `boost` to `depends`
(public headers need it) or ship a `pyxrt` source/ `python-pyxrt` subpackage.

## 6. onnxruntime-extensions (Microsoft) — limitation, not bug

Python custom ops (`@onnx_op`) return outputs through `tensor.flatten().tolist()`
(`_ocos.py: _on_pyop_invocation`), i.e. a Python list of floats, which makes them
unusable for multi-megabyte tensors. No 3.14 wheel either. This is why the bridge
here is a C++ custom op with a C callback. Worth an issue suggesting a buffer-
protocol return path.

## 7. onnxruntime (Microsoft) — STFT operator

### 7.1 CPU `STFT` imaginary part of the DC bin is not zero
For `onesided=1`, frame length 20, hop 5, hann window, the k=0 imaginary output
is up to ~5e-6 in magnitude (inputs O(1)) with sign anti-correlated to the real
part, and the Nyquist bin's imaginary part is likewise noise. Mathematically both
are exactly 0 for a real input. Downstream code that takes the phase (atan2) of
those bins sees ±pi flips. Reproduction: build a single-node STFT model and
compare with `np.fft.rfft` of the windowed frames.

### 7.2 CPU `STFT` is ~20x slower than an equivalent strided Conv
245k samples, frame 20, hop 5: 65 ms for STFT vs 3.6 ms for a `Conv` with the
windowed DFT basis as weights (4 threads), with identical outputs to 1e-6. The
kernel appears to evaluate the DFT per frame without vectorisation/threading.

**Plan:** one issue for both (7.1 is a correctness bug; 7.2 a perf note with the
Conv formulation as the obvious fix for small frame lengths).
