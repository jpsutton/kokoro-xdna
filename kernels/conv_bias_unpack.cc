// SPDX-License-Identifier: Apache-2.0
//===- conv_bias.cc - bias init for the conv1d C tile -----------*- C++ -*-===//
//
// The C tile in core memory uses the matmul microkernel's blocked layout; the
// generated header conv_bias_layout.h provides, per group of CB_GROUP
// consecutive elements, the output row they belong to.
//
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>
// CB_M, CB_ELEMS, CB_GROUP, CB_ROW_OF_GROUP[] come from the generated layout header (-include).

extern "C" {

// a: an A-fifo object (bf16, blocked layout) whose first CB_M stream elements hold
// bias hi parts and the next CB_M the lo parts. bias[i] = hi[i] + lo[i] (f32).
void conv_bias_unpack(const bfloat16 *__restrict a, float *__restrict bias) {
  const uint16_t *u = reinterpret_cast<const uint16_t *>(a);
  for (int i = 0; i < CB_M; i++) {
    uint32_t hi = static_cast<uint32_t>(u[i]) << 16;
    uint32_t lo = static_cast<uint32_t>(u[CB_M + i]) << 16;
    bias[i] = *reinterpret_cast<float *>(&hi) + *reinterpret_cast<float *>(&lo);
  }
}

} // extern "C"
