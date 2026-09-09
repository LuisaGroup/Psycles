#include "tiny_weight_inputs.h"
// Diagnostic: original GPU weight functions, raw binary32 inputs/outputs.
// No host reference arithmetic; inputs stay dynamic kernel arguments.
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/film/light_passes.h"

#include <array>
#include <cstdio>
#include <cstdlib>

void check(hipError_t e) {
  if (e != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(e)); std::exit(1); }
}
__global__ void probe(unsigned *out, unsigned numerator, unsigned denominator, unsigned zero_bits) {
  const auto n = __uint_as_float(numerator);
  const auto d = __uint_as_float(denominator);
  const auto z = __uint_as_float(zero_bits);
  BsdfEval eval{make_float3(z, n, n), make_float3(n, z, n), make_float3(d)};
  auto a = bsdf_eval_pass_diffuse_weight(&eval);
  auto b = bsdf_eval_pass_glossy_weight(&eval);
  out[0] = __float_as_uint(a.x); out[1] = __float_as_uint(a.y); out[2] = __float_as_uint(a.z);
  out[3] = __float_as_uint(b.x); out[4] = __float_as_uint(b.y); out[5] = __float_as_uint(b.z);
}
int main() {

  unsigned *out{}; check(hipMalloc(&out, 6 * sizeof(unsigned)));
  for (const auto &v : tiny_weight_inputs) {
    probe<<<1, 1>>>(out, v[0], v[1], 0u); check(hipGetLastError());
    std::array<unsigned, 6> result{};
    check(hipMemcpy(result.data(), out, sizeof(result), hipMemcpyDeviceToHost));
    std::printf("%08x %08x", v[0], v[1]);
    for (auto bits : result) { std::printf(" %08x", bits); }
    std::printf("\n");
  }
  check(hipFree(out));
}
