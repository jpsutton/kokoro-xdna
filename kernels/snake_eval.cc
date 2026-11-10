// SPDX-License-Identifier: Apache-2.0
#include "snake_common.h"

extern "C" {

void snake_eval_f32(const float *__restrict x, float *__restrict y, const float *__restrict p) {
  const float alpha = p[0];
  const float beta = p[1];
  auto it_x = aie::begin_restrict_vector<VL>(x);
  auto it_y = aie::begin_restrict_vector<VL>(y);
  for (int i = 0; i < SNAKE_TILE / VL; i++) {
    vf xv = *it_x++;
    vf s = vsin(vmul(xv, alpha));
    *it_y++ = vmac(xv, vmul(s, s), aie::broadcast<float, VL>(beta));
  }
}

} // extern "C"
