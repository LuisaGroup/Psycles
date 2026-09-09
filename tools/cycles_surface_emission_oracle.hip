// Capture original Cycles ShaderData emission and film routing on HIP.
// hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC -std=c++20 -O3 \
//   -ffast-math -I /path/to/blender/intern/cycles THIS -o /tmp/emission-oracle
#include "../tests/cycles_surface_emission_fixture.h"
// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "util/atomic.h"
// clang-format on

namespace f = psycles::test_support::surface_emission_fixture;
__device__ float *observed_film;
__device__ unsigned *write_counts;
__device__ float original_atomic(float *p, float v) {
  return atomic_add_and_fetch_float(p, v);
}
__device__ float observed_atomic(float *p, float v) {
  const auto lane = p - observed_film;
  if (lane < 0 || lane >= f::film_lanes) {
    __builtin_trap();
  }
  atomicAdd(write_counts + lane, 1u);
  return original_atomic(p, v);
}
// Observe the original writer without changing its contribution, address or
// execution predicate. Preserve zero writes: the counts are not RGB tests.
#undef atomic_add_and_fetch_float
#define atomic_add_and_fetch_float(p, v) observed_atomic((p), (v))
#include "kernel/film/light_passes.h"
#undef atomic_add_and_fetch_float
#define atomic_add_and_fetch_float(p, v) original_atomic((p), (v))
#include "kernel/integrator/surface_shader.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(error));
    std::exit(1);
  }
}
template <typename T> T *zero(unsigned count) {
  T *out{};
  check(hipMalloc(&out, count * sizeof(T)));
  check(hipMemset(out, 0, count * sizeof(T)));
  return out;
}

__global__ void probe(float *film, unsigned *counts, float3 *emission,
                      f::Input in) {
  observed_film = film;
  write_counts = counts;
  for (unsigned i = 0; i < f::film_lanes; ++i) {
    film[i] = 0;
    counts[i] = 0;
  }
  const IntegratorState state{0};
  const unsigned path_flag = in.route == 0  ? 0
                             : in.route < 3 ? PATH_RAY_SURFACE_PASS
                                            : PATH_RAY_VOLUME_PASS;
  INTEGRATOR_STATE_WRITE(state, path, flag) = path_flag;
  INTEGRATOR_STATE_WRITE(state, path, bounce) = in.route == 0  ? 0
                                                : in.route % 2 ? 1
                                                               : 3;
  INTEGRATOR_STATE_WRITE(state, path, throughput) =
      in.kind == 4 ? one_spectrum() : make_float3(0.5f, 0.75f, 1.25f);
  INTEGRATOR_STATE_WRITE(state, path, pass_diffuse_weight) =
      make_float3(0.25f, 0.125f, 0.5f);
  INTEGRATOR_STATE_WRITE(state, path, pass_glossy_weight) =
      make_float3(0.125f, 0.25f, 0.25f);
  ShaderDataTinyStorage storage{};
  auto *sd = AS_SHADER_DATA(&storage);
  sd->flag = in.emission ? SD_EMISSION : 0;
  sd->Ng = make_float3(0, 0, 1);
  sd->wi = in.kind == 3 ? make_float3(1, 0, 0) : make_float3(0, 0, 1);
  sd->closure_emission_background = in.kind == 1   ? zero_spectrum()
                                    : in.kind == 2 ? make_float3(4, -2, 1)
                                                   : make_float3(2, 4, 1);
  *emission = surface_shader_emission(sd);
  // Entry eligibility is the directly audited integrate_surface predicate.
  // The harness supplies its state; original surface_shader_emission and
  // film_write_surface_emission own all evaluation and pass routing below.
  if (!in.exit && (sd->flag & SD_EMISSION)) {
    film_write_surface_emission(nullptr, state, *emission, 1.0f, film);
  }
}

int main() {
  KernelParamsHIP params{};
  auto &film = params.data.film;
  film.pass_stride = f::film_lanes;
  film.light_pass_flag = PASS_ANY;
  film.pass_combined = 0;
  film.pass_diffuse_direct = 4;
  film.pass_diffuse_indirect = 8;
  film.pass_glossy_direct = 12;
  film.pass_glossy_indirect = 16;
  film.pass_transmission_direct = 20;
  film.pass_transmission_indirect = 24;
  film.pass_volume_direct = 28;
  film.pass_volume_indirect = 32;
  film.pass_emission = 36;
  film.pass_adaptive_aux_buffer = film.pass_denoising_albedo =
      film.pass_lightgroup = PASS_UNUSED;
  film.pass_volume_transmit = film.pass_volume_scatter = PASS_UNUSED;
  params.data.kernel_features = KERNEL_FEATURE_LIGHT_PASSES;
  params.data.integrator.sample_clamp_direct = f::direct_limit;
  params.data.integrator.sample_clamp_indirect = f::indirect_limit;
  auto &path = params.integrator_state.path;
  path.flag = zero<uint32_t>(1);
  path.bounce = zero<uint16_t>(1);
  path.throughput = zero<PackedSpectrum>(1);
  path.pass_diffuse_weight = zero<PackedSpectrum>(1);
  path.pass_glossy_weight = zero<PackedSpectrum>(1);
  path.render_pixel_index = zero<uint32_t>(1);
  path.visibility = zero<uint8_t>(1);
  path.sample = zero<uint32_t>(1);
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
  auto output = zero<float>(f::film_lanes);
  auto counts = zero<unsigned>(f::film_lanes);
  auto emission = zero<float3>(1);
  for (unsigned i = 0; i < f::oracle_cases; ++i) {
    probe<<<1, 1>>>(output, counts, emission, f::inputs[i]);
    check(hipGetLastError());
    std::array<float, f::film_lanes> values{};
    std::array<unsigned, f::film_lanes> writes{};
    float3 L{};
    check(hipMemcpy(values.data(), output, sizeof(values),
                    hipMemcpyDeviceToHost));
    check(hipMemcpy(writes.data(), counts, sizeof(writes),
                    hipMemcpyDeviceToHost));
    check(hipMemcpy(&L, emission, sizeof(L), hipMemcpyDeviceToHost));
    std::printf("E %u %.9g %.9g %.9g", i, L.x, L.y, L.z);
    for (auto v : values) {
      std::printf(" %.9g", v);
    }
    for (auto v : writes) {
      std::printf(" %u", v);
    }
    std::printf("\n");
  }
  // Only this process's small observer allocations are released by exit.
}
