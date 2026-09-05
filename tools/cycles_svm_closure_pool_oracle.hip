// Cycles 5.2.1 HIP is the sole layout/allocation oracle. No allocator or
// closure formula is reimplemented here.
// hipcc --offload-arch=gfx1201 -DHIPCC -std=c++17 -O3 \
//   -I /path/to/blender/intern/cycles \
//   tools/cycles_svm_closure_pool_oracle.hip -o /tmp/closure-pool-oracle
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/closure/alloc.h"
#include "kernel/closure/bsdf_ashikhmin_velvet.h"
#include "kernel/closure/bssrdf.h"
#include "kernel/closure/bsdf_hair.h"
#include "kernel/closure/bsdf_microfacet.h"
#include "kernel/closure/bsdf_oren_nayar.h"
#include "kernel/closure/bsdf_principled_hair_huang.h"
#include "kernel/closure/bsdf_ray_portal.h"
#include "kernel/closure/bsdf_sheen.h"
#include "kernel/closure/bsdf_toon.h"

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#define CLOSURE_POOL_LAYOUT(SIZE, FIELD)                                       \
  SIZE(ShaderClosure)                                                          \
  FIELD(ShaderClosure, weight)                                                 \
  FIELD(ShaderClosure, type)                                                   \
  FIELD(ShaderClosure, sample_weight)                                          \
  FIELD(ShaderClosure, N)                                                      \
  SIZE(OrenNayarBsdf)                                                          \
  FIELD(OrenNayarBsdf, param)                                                  \
  FIELD(OrenNayarParam, roughness)                                             \
  FIELD(OrenNayarParam, a)                                                     \
  FIELD(OrenNayarParam, b)                                                     \
  FIELD(OrenNayarParam, multiscatter_term)                                     \
  SIZE(SheenBsdf)                                                              \
  FIELD(SheenBsdf, roughness)                                                  \
  FIELD(SheenBsdf, transformA)                                                 \
  FIELD(SheenBsdf, transformB)                                                 \
  FIELD(SheenBsdf, T)                                                          \
  FIELD(SheenBsdf, B)                                                          \
  SIZE(VelvetBsdf)                                                             \
  FIELD(VelvetBsdf, sigma)                                                     \
  FIELD(VelvetBsdf, invsigma2)                                                 \
  SIZE(ToonBsdf)                                                               \
  FIELD(ToonBsdf, size)                                                        \
  FIELD(ToonBsdf, smooth)                                                      \
  SIZE(RayPortalClosure)                                                       \
  FIELD(RayPortalClosure, P)                                                   \
  FIELD(RayPortalClosure, D)                                                   \
  SIZE(HairBsdf)                                                               \
  FIELD(HairBsdf, T)                                                           \
  FIELD(HairBsdf, roughness1)                                                  \
  FIELD(HairBsdf, roughness2)                                                  \
  FIELD(HairBsdf, offset)                                                      \
  SIZE(ChiangHairBSDF)                                                         \
  FIELD(ChiangHairBSDF, sigma)                                                 \
  FIELD(ChiangHairBSDF, v)                                                     \
  FIELD(ChiangHairBSDF, s)                                                     \
  FIELD(ChiangHairBSDF, alpha)                                                 \
  FIELD(ChiangHairBSDF, eta)                                                   \
  FIELD(ChiangHairBSDF, m0_roughness)                                          \
  FIELD(ChiangHairBSDF, h)                                                     \
  SIZE(HuangHairBSDF)                                                          \
  FIELD(HuangHairBSDF, sigma)                                                  \
  FIELD(HuangHairBSDF, roughness)                                              \
  FIELD(HuangHairBSDF, tilt)                                                   \
  FIELD(HuangHairBSDF, eta)                                                    \
  FIELD(HuangHairBSDF, aspect_ratio)                                           \
  FIELD(HuangHairBSDF, h)                                                      \
  FIELD(HuangHairBSDF, extra)                                                  \
  SIZE(HuangHairExtra)                                                         \
  FIELD(HuangHairExtra, R)                                                     \
  FIELD(HuangHairExtra, TT)                                                    \
  FIELD(HuangHairExtra, TRT)                                                   \
  FIELD(HuangHairExtra, Y)                                                     \
  FIELD(HuangHairExtra, Z)                                                     \
  FIELD(HuangHairExtra, wi)                                                    \
  FIELD(HuangHairExtra, radius)                                                \
  FIELD(HuangHairExtra, e2)                                                    \
  FIELD(HuangHairExtra, pixel_coverage)                                        \
  FIELD(HuangHairExtra, h)                                                     \
  SIZE(Bssrdf)                                                                 \
  FIELD(Bssrdf, radius)                                                        \
  FIELD(Bssrdf, albedo)                                                        \
  FIELD(Bssrdf, anisotropy)                                                    \
  FIELD(Bssrdf, ior)                                                           \
  FIELD(Bssrdf, alpha)                                                         \
  SIZE(MicrofacetBsdf)                                                         \
  FIELD(MicrofacetBsdf, alpha_x)                                               \
  FIELD(MicrofacetBsdf, alpha_y)                                               \
  FIELD(MicrofacetBsdf, ior)                                                   \
  FIELD(MicrofacetBsdf, energy_scale)                                          \
  FIELD(MicrofacetBsdf, fresnel_type)                                          \
  FIELD(MicrofacetBsdf, fresnel)                                               \
  FIELD(MicrofacetBsdf, T)                                                     \
  SIZE(FresnelThinFilm)                                                        \
  FIELD(FresnelThinFilm, thickness)                                            \
  FIELD(FresnelThinFilm, ior)                                                  \
  SIZE(FresnelGeneralizedSchlick)                                              \
  FIELD(FresnelGeneralizedSchlick, thin_film)                                  \
  FIELD(FresnelGeneralizedSchlick, reflection_tint)                            \
  FIELD(FresnelGeneralizedSchlick, transmission_tint)                          \
  FIELD(FresnelGeneralizedSchlick, f0)                                         \
  FIELD(FresnelGeneralizedSchlick, f90)                                        \
  FIELD(FresnelGeneralizedSchlick, exponent)                                   \
  SIZE(FresnelConductor)                                                       \
  FIELD(FresnelConductor, thin_film)                                           \
  FIELD(FresnelConductor, ior)                                                 \
  FIELD(complex<Spectrum>, re)                                                 \
  FIELD(complex<Spectrum>, im)                                                 \
  SIZE(FresnelF82Tint)                                                         \
  FIELD(FresnelF82Tint, thin_film)                                             \
  FIELD(FresnelF82Tint, f0)                                                    \
  FIELD(FresnelF82Tint, b)

namespace {
#define SIZE_NAME(T) #T ".sizeof",
#define FIELD_NAME(T, M) #T "." #M,
constexpr const char *layout_names[]{
    CLOSURE_POOL_LAYOUT(SIZE_NAME, FIELD_NAME)};
#undef SIZE_NAME
#undef FIELD_NAME
constexpr unsigned layout_count = sizeof(layout_names) / sizeof(*layout_names);
constexpr unsigned capacity_count = 9u;
constexpr unsigned steps = 7u;
constexpr unsigned mixed_cases = 3u;
constexpr unsigned mixed_rows = 4u;

void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(error));
    std::exit(1);
  }
}

__global__ void layout_probe(unsigned *out) {
  unsigned i = 0u;
#define SIZE_VALUE(T) out[i++] = sizeof(T);
#define FIELD_VALUE(T, M) out[i++] = offsetof(T, M);
  CLOSURE_POOL_LAYOUT(SIZE_VALUE, FIELD_VALUE)
#undef SIZE_VALUE
#undef FIELD_VALUE
}

__global__ void allocation_probe(uint4 *out) {
  const unsigned capacity = threadIdx.x;
  if (capacity >= capacity_count) {
    return;
  }
  ShaderData sd;
  sd.num_closure = 0;
  sd.num_closure_left = capacity;
  for (unsigned step = 0u; step < steps; ++step) {
    const unsigned wanted = step % 3u;
    auto *ordinary =
        closure_alloc(&sd, sizeof(ShaderClosure), CLOSURE_BSDF_DIFFUSE_ID,
                      make_float3(float(step + 1u)));
    void *extra = ordinary
                      ? closure_alloc_extra(&sd, wanted * sizeof(ShaderClosure))
                      : nullptr;
    const auto ordinary_slot = ordinary ? unsigned(ordinary - sd.closure) : ~0u;
    const auto extra_slot =
        extra ? unsigned(static_cast<ShaderClosure *>(extra) - sd.closure)
              : ~0u;
    out[capacity * steps + step] = make_uint4(
        ordinary_slot, extra_slot, sd.num_closure, sd.num_closure_left);
  }
}

// Exercise both pointer-bearing closure families in the same full pool.
// A varying live prefix keeps owner indices dynamic; all remaining ordinary
// slots are then written before either tail payload is read back.
__global__ void mixed_payload_probe(uint4 *out) {
  const unsigned prefix = threadIdx.x;
  if (prefix >= mixed_cases) {
    return;
  }
  ShaderData sd;
  sd.num_closure = 0;
  sd.num_closure_left = 8;
  for (unsigned i = 0u; i < prefix; ++i) {
    closure_alloc(&sd, sizeof(ShaderClosure), CLOSURE_NONE_ID, zero_float3());
  }
  auto *micro = reinterpret_cast<MicrofacetBsdf *>(closure_alloc(
      &sd, sizeof(MicrofacetBsdf), CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID,
      make_float3(2.0f, 3.0f, 4.0f)));
  micro->sample_weight = 5.0f;
  auto *fresnel = static_cast<FresnelGeneralizedSchlick *>(
      closure_alloc_extra(&sd, sizeof(FresnelGeneralizedSchlick)));
  micro->fresnel = fresnel;
  fresnel->f0 = make_float3(9.0f + float(prefix), 10.0f, 11.0f);

  auto *hair = reinterpret_cast<HuangHairBSDF *>(
      closure_alloc(&sd, sizeof(HuangHairBSDF), CLOSURE_BSDF_HAIR_HUANG_ID,
                    make_float3(12.0f, 13.0f, 14.0f)));
  hair->sample_weight = 15.0f;
  hair->extra = static_cast<HuangHairExtra *>(
      closure_alloc_extra(&sd, sizeof(HuangHairExtra)));
  hair->extra->Y = make_float3(19.0f, 20.0f, 21.0f);
  hair->extra->R = 22.0f;

  while (sd.num_closure_left != 0) {
    const float v = 100.0f + sd.num_closure;
    auto *filler = reinterpret_cast<MicrofacetBsdf *>(
        closure_alloc(&sd, sizeof(MicrofacetBsdf),
                      CLOSURE_BSDF_MICROFACET_GGX_ID, make_float3(v)));
    filler->sample_weight = v;
    filler->N = make_float3(v);
    filler->alpha_x = v;
    filler->alpha_y = v;
    filler->ior = v;
    filler->energy_scale = v;
    filler->fresnel_type = MicrofacetFresnel::NONE;
    filler->T = make_float3(v);
  }
  const float4 values[]{make_float4(micro->weight, micro->sample_weight),
                        make_float4(fresnel->f0, float(sd.num_closure_left)),
                        make_float4(hair->weight, hair->sample_weight),
                        make_float4(hair->extra->Y, hair->extra->R)};
  for (unsigned row = 0u; row < mixed_rows; ++row) {
    const auto v = values[row];
    out[prefix * mixed_rows + row] =
        make_uint4(__float_as_uint(v.x), __float_as_uint(v.y),
                   __float_as_uint(v.z), __float_as_uint(v.w));
  }
}
} // namespace

int main() {
  unsigned *layout{};
  uint4 *allocations{};
  uint4 *payloads{};
  check(hipMalloc(&layout, layout_count * sizeof(unsigned)));
  check(hipMalloc(&allocations, capacity_count * steps * sizeof(uint4)));
  check(hipMalloc(&payloads, mixed_cases * mixed_rows * sizeof(uint4)));
  layout_probe<<<1, 1>>>(layout);
  allocation_probe<<<1, 32>>>(allocations);
  mixed_payload_probe<<<1, 32>>>(payloads);
  check(hipGetLastError());
  std::array<unsigned, layout_count> host_layout{};
  std::array<uint4, capacity_count * steps> host_allocations{};
  std::array<uint4, mixed_cases * mixed_rows> host_payloads{};
  check(hipMemcpy(host_layout.data(), layout, sizeof(host_layout),
                  hipMemcpyDeviceToHost));
  check(hipMemcpy(host_allocations.data(), allocations,
                  sizeof(host_allocations), hipMemcpyDeviceToHost));
  check(hipMemcpy(host_payloads.data(), payloads, sizeof(host_payloads),
                  hipMemcpyDeviceToHost));
  for (unsigned i = 0; i < layout_count; ++i) {
    std::printf("layout %s %u\n", layout_names[i], host_layout[i]);
  }
  for (unsigned capacity = 0; capacity < capacity_count; ++capacity) {
    for (unsigned step = 0; step < steps; ++step) {
      const auto v = host_allocations[capacity * steps + step];
      std::printf("allocation %u %u %u %u %u %u\n", capacity, step, v.x, v.y,
                  v.z, v.w);
    }
  }
  for (unsigned i = 0u; i < host_payloads.size(); ++i) {
    const auto v = host_payloads[i];
    std::printf("payload %u %u %u %u %u\n", i, v.x, v.y, v.z, v.w);
  }
  check(hipFree(payloads));
  check(hipFree(allocations));
  check(hipFree(layout));
}
