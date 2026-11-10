// SPDX-License-Identifier: Apache-2.0
#include "snake_common.h"

extern "C" {

// Variant v: out[i] = part_v( y_prev[i + sh_v] ) for i + sh_v < L, else part_v( y_cur[i + sh_v - L] ).
// sh_v = (int)p[2 + v] (0 <= sh_v <= 16); part_v = p[2 + SNAKE_MAXV + v] != 0 selects the bf16
// residual (value - bf16(value)) instead of the bf16 value.
void snake_emit_bf16(const float *__restrict y_prev, const float *__restrict y_cur,
                     bfloat16 *__restrict out, const float *__restrict p, int v) {
  const unsigned sh = static_cast<unsigned>(p[2 + v]);
  const bool part = p[2 + SNAKE_MAXV + v] != 0.0f;
  aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr int NB = SNAKE_TILE / VL;
  auto it_o = aie::begin_restrict_vector<VL>(out);
  vf a = aie::load_v<VL>(y_prev);
  for (int i = 0; i < NB; i++) {
    vf b = (i + 1 < NB) ? aie::load_v<VL>(y_prev + (i + 1) * VL) : aie::load_v<VL>(y_cur);
    vf s = aie::shuffle_down_fill(a, b, sh);
    aie::vector<bfloat16, VL> hi = aie::accum<accfloat, VL>(s).template to_vector<bfloat16>();
    if (!part) {
      *it_o++ = hi;
    } else {
      vf r = vsub(s, aie::accum<accfloat, VL>(hi).template to_vector<float>());
      *it_o++ = aie::accum<accfloat, VL>(r).template to_vector<bfloat16>();
    }
    a = b;
  }
}

} // extern "C"
