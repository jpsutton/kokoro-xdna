// SPDX-License-Identifier: Apache-2.0
//===- snake.cc - Snake activation for AIE2P --------------------*- C++ -*-===//
//
// y = x + beta * sin(alpha * x)^2       (Kokoro/BigVGAN "Snake", beta = 1/(alpha+eps))
//
// f32 in / f32 out, one tile of SNAKE_TILE elements of a single channel; the
// per-channel (alpha, beta) pair arrives in a separate params tile p[0], p[1].
// sin() is computed in f32 with range reduction to [-pi/2, pi/2] and an odd
// Taylor polynomial through x^11 (abs error < 1e-7 on that range).
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef SNAKE_TILE
#define SNAKE_TILE 1024
#endif

namespace {
constexpr int VL = 16;
constexpr float INV_2PI = 0.15915494309189535f;
constexpr float TWO_PI_HI = 6.28125f;              // exact in f32 (short mantissa)
constexpr float TWO_PI_LO = 0.0019353071795864769f; // 2*pi - TWO_PI_HI
constexpr float PI_HI = 3.140625f;
constexpr float PI_LO = 0.0009676535897932384f;
constexpr float HALF_PI = 1.5707963267948966f;

using vf = aie::vector<float, VL>;

__attribute__((always_inline)) inline vf vmul(vf a, vf b) {
  return aie::mul(a, b).template to_vector<float>();
}
__attribute__((always_inline)) inline vf vmul(vf a, float b) {
  return aie::mul(a, b).template to_vector<float>();
}
__attribute__((always_inline)) inline vf vadd(vf a, vf b) { return aie::add(a, b); }
__attribute__((always_inline)) inline vf vsub(vf a, vf b) { return aie::sub(a, b); }
__attribute__((always_inline)) inline vf vmsc(vf acc, vf a, float b) {
  // acc - a*b
  return aie::msc(aie::accum<accfloat, VL>(acc), a, b).template to_vector<float>();
}
__attribute__((always_inline)) inline vf vmac(vf acc, vf a, vf b) {
  return aie::mac(aie::accum<accfloat, VL>(acc), a, b).template to_vector<float>();
}

// sin(t) for arbitrary f32 t.
__attribute__((always_inline)) inline vf vsin(vf t) {
  // q = round(t / 2pi) via the hardware float->int conversion; r = t - q*2pi in [-pi, pi]
  aie::vector<int32, VL> qi = aie::to_fixed<int32>(vmul(t, INV_2PI), 0);
  vf q = aie::to_float<float>(qi, 0);
  vf r = vmsc(t, q, TWO_PI_HI);
  r = vmsc(r, q, TWO_PI_LO);
  // fold to [-pi/2, pi/2]: r > pi/2 -> pi - r ; r < -pi/2 -> -pi - r
  vf pi_v = vadd(aie::broadcast<float, VL>(PI_HI), aie::broadcast<float, VL>(PI_LO));
  vf hi = vsub(pi_v, r);
  vf lo = vsub(aie::broadcast<float, VL>(-PI_HI - PI_LO), r);
  auto m_hi = aie::gt(r, aie::broadcast<float, VL>(HALF_PI));
  auto m_lo = aie::lt(r, aie::broadcast<float, VL>(-HALF_PI));
  r = aie::select(r, hi, m_hi);
  r = aie::select(r, lo, m_lo);
  // odd Taylor polynomial
  vf r2 = vmul(r, r);
  vf p = aie::broadcast<float, VL>(-2.5052108385441718e-08f); // -1/11!
  p = vadd(vmul(p, r2), aie::broadcast<float, VL>(2.7557319223985893e-06f)); // 1/9!
  p = vadd(vmul(p, r2), aie::broadcast<float, VL>(-1.9841269841269841e-04f)); // -1/7!
  p = vadd(vmul(p, r2), aie::broadcast<float, VL>(8.3333333333333332e-03f)); // 1/5!
  p = vadd(vmul(p, r2), aie::broadcast<float, VL>(-1.6666666666666666e-01f)); // -1/3!
  p = vadd(vmul(p, r2), aie::broadcast<float, VL>(1.0f));
  return vmul(p, r);
}
} // namespace

extern "C" {

void snake_f32(const float *__restrict x, float *__restrict y, const float *__restrict p) {
  const float alpha = p[0];
  const float beta = p[1];
  auto it_x = aie::begin_restrict_vector<VL>(x);
  auto it_y = aie::begin_restrict_vector<VL>(y);
  for (int i = 0; i < SNAKE_TILE / VL; i++) {
    vf xv = *it_x++;
    vf s = vsin(vmul(xv, alpha));
    vf s2 = vmul(s, s);
    vf out = vmac(xv, s2, aie::broadcast<float, VL>(beta));
    *it_y++ = out;
  }
}

} // extern "C"
