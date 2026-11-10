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

// c[q] = bias[row(q)] for the whole tile.
void conv_bias_fill(float *__restrict c, const float *__restrict bias) {
  auto it = aie::begin_restrict_vector<CB_GROUP>(c);
  for (int g = 0; g < CB_ELEMS / CB_GROUP; g++) {
    *it++ = aie::broadcast<float, CB_GROUP>(bias[CB_ROW_OF_GROUP[g]]);
  }
}

} // extern "C"
