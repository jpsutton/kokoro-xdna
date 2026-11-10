// SPDX-License-Identifier: Apache-2.0
#pragma once
//===- snake_common.h - shared code for the fused snake kernels --*- C++ -*-===//
//
// y = x + beta * sin(alpha * x)^2  computed in f32 (see snake.cc), then emitted
// as bf16 "variant" tiles laid out for kernels/conv1d_npu.py:
//
//   variant v of tile t:  out[i] = part_v( y[t*L + i + sh_v] )
//
// where sh_v is a small positive shift (the conv's tap pairing wants a copy of
// the row shifted by dil, and DMA 4-byte alignment adds 0/1), so the kernel
// keeps the previous tile's y and looks ahead into the current one.
// part 0 = bf16 round-to-nearest-even, part 1 = bf16 of the residual.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef SNAKE_TILE
#define SNAKE_TILE 1008
#endif
#define SNAKE_MAXV 6

namespace {
constexpr int VL = 16;
constexpr float INV_2PI = 0.15915494309189535f;
constexpr float TWO_PI_HI = 6.28125f;
constexpr float TWO_PI_LO = 0.0019353071795864769f;
constexpr float PI_HI = 3.140625f;
constexpr float PI_LO = 0.0009676535897932384f;
constexpr float HALF_PI = 1.5707963267948966f;

using vf = aie::vector<float, VL>;

__attribute__((always_inline)) inline vf vmul(vf a, vf b) { return aie::mul(a, b).template to_vector<float>(); }
__attribute__((always_inline)) inline vf vmul(vf a, float b) { return aie::mul(a, b).template to_vector<float>(); }
__attribute__((always_inline)) inline vf vadd(vf a, vf b) { return aie::add(a, b); }
__attribute__((always_inline)) inline vf vsub(vf a, vf b) { return aie::sub(a, b); }
__attribute__((always_inline)) inline vf vmsc(vf acc, vf a, float b) {
  return aie::msc(aie::accum<accfloat, VL>(acc), a, b).template to_vector<float>();
}
__attribute__((always_inline)) inline vf vmac(vf acc, vf a, vf b) {
  return aie::mac(aie::accum<accfloat, VL>(acc), a, b).template to_vector<float>();
}

__attribute__((always_inline)) inline vf vsin(vf t) {
  aie::vector<int32, VL> qi = aie::to_fixed<int32>(vmul(t, INV_2PI), 0);
  vf q = aie::to_float<float>(qi, 0);
  vf r = vmsc(t, q, TWO_PI_HI);
  r = vmsc(r, q, TWO_PI_LO);
  vf pi_v = vadd(aie::broadcast<float, VL>(PI_HI), aie::broadcast<float, VL>(PI_LO));
  vf hi = vsub(pi_v, r);
  vf lo = vsub(aie::broadcast<float, VL>(-PI_HI - PI_LO), r);
  auto m_hi = aie::gt(r, aie::broadcast<float, VL>(HALF_PI));
  auto m_lo = aie::lt(r, aie::broadcast<float, VL>(-HALF_PI));
  r = aie::select(r, hi, m_hi);
  r = aie::select(r, lo, m_lo);
  vf r2 = vmul(r, r);
  vf p = aie::broadcast<float, VL>(-2.5052108385441718e-08f);
  p = vadd(vmul(p, r2), aie::broadcast<float, VL>(2.7557319223985893e-06f));
  p = vadd(vmul(p, r2), aie::broadcast<float, VL>(-1.9841269841269841e-04f));
  p = vadd(vmul(p, r2), aie::broadcast<float, VL>(8.3333333333333332e-03f));
  p = vadd(vmul(p, r2), aie::broadcast<float, VL>(-1.6666666666666666e-01f));
  p = vadd(vmul(p, r2), aie::broadcast<float, VL>(1.0f));
  return vmul(p, r);
}

// f32 -> bf16 round-to-nearest-even, scalar (used on the few tail elements).
__attribute__((always_inline)) inline uint16_t f2bf(float f) {
  uint32_t u = *reinterpret_cast<uint32_t*>(&f);
  uint32_t lsb = (u >> 16) & 1u;
  u += 0x7FFFu + lsb;
  return static_cast<uint16_t>(u >> 16);
}
__attribute__((always_inline)) inline float bf2f(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16;
  return *reinterpret_cast<float*>(&u);
}
} // namespace

