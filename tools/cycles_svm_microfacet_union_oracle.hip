// Cycles itself is the oracle: no closure formula is implemented here.
// hipcc --offload-arch=gfx1201 -DHIPCC -std=c++17 -O3 -ffast-math \
//   -I /path/to/blender/intern/cycles \
//   tools/cycles_svm_microfacet_union_oracle.hip -o /tmp/microfacet-union-oracle
// The four rows are GGX conductor/F82, then Beckmann conductor/F82.
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/closure/bsdf_microfacet.h"

#include <array>
#include <cstdio>
#include <cstdlib>

namespace {

void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(error));
    std::exit(1);
  }
}

__global__ void probe(float4 *values, int *labels) {
  const unsigned int scenario = threadIdx.x;
  if (scenario >= 4u) {
    return;
  }
  const bool f82 = (scenario & 1u) != 0u;
  const bool ggx = scenario < 2u;
  const float3 N = normalize(make_float3(0.2f, -0.3f, 1.0f));
  const float3 wi = normalize(make_float3(0.35f, -0.15f, 0.925f));
  const float3 direct = normalize(make_float3(0.4f, 0.1f, 0.91f));
  float3 T, B;
  make_orthonormals_tangent(N, normalize(make_float3(0.9f, 0.2f, -0.1f)), &T, &B);
  MicrofacetBsdf bsdf{};
  bsdf.N = N;
  bsdf.T = T;
  bsdf.type = ggx ? CLOSURE_BSDF_MICROFACET_GGX_ID : CLOSURE_BSDF_MICROFACET_BECKMANN_ID;
  bsdf.alpha_x = f82 ? 0.42f : 0.25f;
  bsdf.alpha_y = f82 ? 0.18f : 0.52f;
  bsdf.ior = 1.45f;
  bsdf.energy_scale = f82 ? 1.07f : 1.11f;
  FresnelConductor conductor{};
  conductor.thin_film = {0.0f, 1.4f};
  conductor.ior = {make_float3(0.3f, 0.8f, 1.2f), make_float3(3.0f, 2.0f, 1.5f)};
  FresnelF82Tint tint{};
  tint.thin_film = {0.0f, 1.3f};
  tint.f0 = make_float3(0.8f, 0.4f, 0.1f);
  tint.b = make_float3(0.12f, -0.08f, 0.04f);
  bsdf.fresnel_type = f82 ? MicrofacetFresnel::F82_TINT : MicrofacetFresnel::CONDUCTOR;
  bsdf.fresnel = f82 ? static_cast<void *>(&tint) : static_cast<void *>(&conductor);
  const float3 random = f82 ? make_float3(0.67f, 0.19f, 0.5f) : make_float3(0.21f, 0.83f, 0.5f);
  Spectrum eval = zero_spectrum();
  float3 wo = zero_float3();
  float pdf = 0.0f;
  float2 roughness = zero_float2();
  float eta = 0.0f;
  const auto *sc = reinterpret_cast<const ShaderClosure *>(&bsdf);
  labels[scenario] = ggx ?
      bsdf_microfacet_ggx_sample(nullptr, sc, N, wi, random, &eval, &wo, &pdf, &roughness, &eta) :
      bsdf_microfacet_beckmann_sample(nullptr, sc, N, wi, random, &eval, &wo, &pdf, &roughness, &eta);
  values[scenario * 4u] = make_float4(wo.x, wo.y, wo.z, pdf);
  values[scenario * 4u + 1u] = make_float4(eval.x, eval.y, eval.z, 0.0f);
  values[scenario * 4u + 2u] = make_float4(roughness.x, roughness.y, eta, 0.0f);
  pdf = 0.0f;
  eval = ggx ? bsdf_microfacet_ggx_eval(nullptr, sc, wi, direct, &pdf) :
               bsdf_microfacet_beckmann_eval(nullptr, sc, wi, direct, &pdf);
  values[scenario * 4u + 3u] = make_float4(eval.x, eval.y, eval.z, pdf);
}

} // namespace

int main() {
  float4 *values = nullptr;
  int *labels = nullptr;
  check(hipMalloc(&values, 16u * sizeof(float4)));
  check(hipMalloc(&labels, 4u * sizeof(int)));
  probe<<<1u, 32u>>>(values, labels);
  check(hipGetLastError());
  std::array<float4, 16u> host_values{};
  std::array<int, 4u> host_labels{};
  check(hipMemcpy(host_values.data(), values, sizeof(host_values), hipMemcpyDeviceToHost));
  check(hipMemcpy(host_labels.data(), labels, sizeof(host_labels), hipMemcpyDeviceToHost));
  for (unsigned int scenario = 0u; scenario < 4u; ++scenario) {
    for (unsigned int row = 0u; row < 4u; ++row) {
      const auto value = host_values[scenario * 4u + row];
      std::printf("%.9g %.9g %.9g %.9g ", value.x, value.y, value.z, value.w);
    }
    std::printf("%d\n", host_labels[scenario]);
  }
  check(hipFree(labels));
  check(hipFree(values));
}
