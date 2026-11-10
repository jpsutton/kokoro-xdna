// SPDX-License-Identifier: Apache-2.0
#include "snake_common.h"

extern "C" {

void zero_f32(float *__restrict y) {
  auto it_y = aie::begin_restrict_vector<VL>(y);
  for (int i = 0; i < SNAKE_TILE / VL; i++) *it_y++ = aie::zeros<float, VL>();
}

} // extern "C"
