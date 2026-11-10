// SPDX-License-Identifier: Apache-2.0
#include "snake_common.h"

extern "C" {

void copy_f32(const float *__restrict src, float *__restrict dst) {
  auto it_s = aie::begin_restrict_vector<VL>(src);
  auto it_d = aie::begin_restrict_vector<VL>(dst);
  for (int i = 0; i < SNAKE_TILE / VL; i++) *it_d++ = *it_s++;
}

} // extern "C"
