// SPDX-License-Identifier: Apache-2.0
// Host-side packing helpers for the NPU runtime (called via ctypes).
// Memory-bound loops kept simple so the compiler vectorizes them.

#include <cstdint>
#include <cstring>

namespace {

// f32 -> bf16 with round-to-nearest-even (NaN payloads not preserved).
inline uint16_t f2bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  uint32_t lsb = (u >> 16) & 1u;
  u += 0x7FFFu + lsb;
  return static_cast<uint16_t>(u >> 16);
}

inline void cast_row(const float* __restrict src, uint16_t* __restrict dst, int64_t n) {
  for (int64_t i = 0; i < n; i++) dst[i] = f2bf(src[i]);
}

}  // namespace

extern "C" {

// Pack conv input x[cin, t_in] (row stride x_ld) into Z[nvar*cin_pad, ldx] bf16.
// Row c*nvar + v holds variant v of channel c: part_v(Xp[c, col + shift_v*dil])
// where Xp is x zero-padded with pad_l leading zeros; part 0 = bf16 rounding of
// the value, part 1 = bf16 rounding of the residual (value - part0).
__attribute__((visibility("default"))) void npu_pack_conv_z(const float* x, int64_t x_ld, int64_t cin, int64_t t_in,
                                                            int64_t cin_pad, int64_t ldx, int64_t pad_l, int64_t dil,
                                                            int64_t nvar, const int64_t* shifts, const int64_t* parts,
                                                            uint16_t* z) {
  const int64_t rows = nvar * cin_pad;
  for (int64_t r = 0; r < rows; r++) {
    uint16_t* zr = z + r * ldx;
    const int64_t c = r / nvar;
    const int64_t v = r % nvar;
    if (c >= cin) {
      std::memset(zr, 0, ldx * sizeof(uint16_t));
      continue;
    }
    const float* xr = x + c * x_ld;
    int64_t start = pad_l - shifts[v] * dil;  // column where x[c, 0] lands
    int64_t x0 = 0;
    if (start < 0) {
      x0 = -start;
      start = 0;
    }
    int64_t n = t_in - x0;
    if (start + n > ldx) n = ldx - start;
    if (n < 0) n = 0;
    if (start > 0) std::memset(zr, 0, start * sizeof(uint16_t));
    if (parts[v] == 0) {
      cast_row(xr + x0, zr + start, n);
    } else {
      const float* src = xr + x0;
      uint16_t* dst = zr + start;
      for (int64_t i = 0; i < n; i++) {
        uint16_t hi = f2bf(src[i]);
        uint32_t u = static_cast<uint32_t>(hi) << 16;
        float hf;
        std::memcpy(&hf, &u, 4);
        dst[i] = f2bf(src[i] - hf);
      }
    }
    int64_t end = start + n;
    if (end < ldx) std::memset(zr + end, 0, (ldx - end) * sizeof(uint16_t));
  }
}

// dst[r, :cols] = src[r, :cols] for r in rows, with independent row strides.
__attribute__((visibility("default"))) void npu_copy_rows(const float* src, int64_t src_ld, float* dst,
                                                          int64_t dst_ld, int64_t rows, int64_t cols) {
  for (int64_t r = 0; r < rows; r++) std::memcpy(dst + r * dst_ld, src + r * src_ld, cols * sizeof(float));
}

// dst[r, :cols] += src[r, :cols]
__attribute__((visibility("default"))) void npu_add_rows(const float* src, int64_t src_ld, float* dst,
                                                         int64_t dst_ld, int64_t rows, int64_t cols) {
  for (int64_t r = 0; r < rows; r++) {
    const float* s = src + r * src_ld;
    float* d = dst + r * dst_ld;
    for (int64_t i = 0; i < cols; i++) d[i] += s[i];
  }
}

// Copy x[rows, cols] into dst[rows, dst_ld] and zero the tail columns [cols, dst_ld).
__attribute__((visibility("default"))) void npu_copy_rows_pad(const float* src, int64_t src_ld, float* dst,
                                                              int64_t dst_ld, int64_t rows, int64_t cols) {
  for (int64_t r = 0; r < rows; r++) {
    std::memcpy(dst + r * dst_ld, src + r * src_ld, cols * sizeof(float));
    if (dst_ld > cols) std::memset(dst + r * dst_ld + cols, 0, (dst_ld - cols) * sizeof(float));
  }
}
}

extern "C" {
// Zero columns [0, e) and [e + t, ld) of each of `rows` uint16 rows.
__attribute__((visibility("default"))) void npu_zero_strips(uint16_t* z, int64_t rows, int64_t ld, int64_t e, int64_t t) {
  const int64_t tail0 = e + t;
  for (int64_t r = 0; r < rows; r++) {
    uint16_t* zr = z + r * ld;
    if (e > 0) std::memset(zr, 0, e * sizeof(uint16_t));
    if (tail0 < ld) std::memset(zr + tail0, 0, (ld - tail0) * sizeof(uint16_t));
  }
}
}
