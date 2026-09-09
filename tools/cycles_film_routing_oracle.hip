// GPU-only observer of native shadow state and film routing. Generate the
// original-source adapter with capture_cycles_film_routing.py first.
#include "../tests/cycles_film_routing_fixture.h"
// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "util/atomic.h"
// clang-format on

namespace f = psycles::test_support::film_routing_fixture;
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
#undef atomic_add_and_fetch_float
#define atomic_add_and_fetch_float(p, v) observed_atomic((p), (v))
#include "kernel/film/light_passes.h"
#undef atomic_add_and_fetch_float
#define atomic_add_and_fetch_float(p, v) original_atomic((p), (v))
#include "film_routing_native.h"

#include <cstdio>
#include <cstdlib>

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
__device__ float3 rgb(f::RGB value) {
  return make_float3(value[0], value[1], value[2]);
}
struct ObservedState {
  unsigned flag, bounce, visibility;
  float3 diffuse, glossy;
};

__global__ void probe(float *film, unsigned *counts, ObservedState *out,
                      f::Input in) {
  observed_film = film;
  write_counts = counts;
  for (unsigned i = 0; i < f::film_lanes; ++i) {
    film[i] = 0;
    counts[i] = 0;
  }
  const IntegratorState state{0};
  const IntegratorShadowState shadow{0};
  unsigned flag = in.route == 0   ? 0u
                  : in.route == 1 ? PATH_RAY_SURFACE_PASS
                  : in.route == 2 ? PATH_RAY_VOLUME_PASS
                  : in.route == 3 ? PATH_RAY_ANY_PASS
                                  : PATH_RAY_TRANSPARENT | PATH_RAY_VOLUME_PRIMARY_TRANSMIT;
  INTEGRATOR_STATE_WRITE(state, path, flag) = flag;
  INTEGRATOR_STATE_WRITE(state, path, bounce) = in.bounce;
  INTEGRATOR_STATE_WRITE(state, path, visibility) = PATH_RAY_VISIBILITY_CAMERA;
  INTEGRATOR_STATE_WRITE(state, path, throughput) = one_spectrum();
  INTEGRATOR_STATE_WRITE(state, path, pass_diffuse_weight) = rgb(in.diffuse_weight);
  INTEGRATOR_STATE_WRITE(state, path, pass_glossy_weight) = rgb(in.glossy_weight);
  const auto contribution = rgb(in.contribution);
  if (in.operation < f::emission) {
    BsdfEval eval{rgb(in.bsdf_diffuse), rgb(in.bsdf_glossy), rgb(in.bsdf_sum)};
    if (in.operation == f::surface_nee) {
      original_surface_shadow_pass(state, shadow, eval);
    } else {
      original_volume_shadow_pass(state, shadow, eval);
    }
    INTEGRATOR_STATE_WRITE(shadow, shadow_path, throughput) = contribution;
    film_write_direct_light(nullptr, shadow, film);
    *out = {INTEGRATOR_STATE(shadow, shadow_path, flag),
            INTEGRATOR_STATE(shadow, shadow_path, bounce),
            INTEGRATOR_STATE(shadow, shadow_path, visibility),
            INTEGRATOR_STATE(shadow, shadow_path, pass_diffuse_weight),
            INTEGRATOR_STATE(shadow, shadow_path, pass_glossy_weight)};
  } else {
    auto value = contribution;
    film_clamp_light(nullptr, &value, max(0, int(in.bounce) - 1));
    film_write_combined_pass(nullptr, PATH_RAY_VISIBILITY_CAMERA, flag, 0, value, film);
    film_write_emission_or_background_pass(
        nullptr, state, value, film,
        in.operation == f::emission ? kernel_data.film.pass_emission
                                   : kernel_data.film.pass_background);
    *out = {flag, in.bounce, PATH_RAY_VISIBILITY_CAMERA,
            rgb(in.diffuse_weight), rgb(in.glossy_weight)};
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
  film.pass_background = 40;
  film.pass_adaptive_aux_buffer = film.pass_denoising_albedo =
      film.pass_lightgroup = PASS_UNUSED;
  film.pass_volume_transmit = film.pass_volume_scatter = PASS_UNUSED;
  params.data.kernel_features = KERNEL_FEATURE_LIGHT_PASSES;
  params.data.integrator.sample_clamp_direct = 2.0f;
  params.data.integrator.sample_clamp_indirect = 1.5f;
  auto allocate = [](auto &p) {
    p.flag = zero<uint32_t>(1);
    p.bounce = zero<uint16_t>(1);
    p.throughput = zero<PackedSpectrum>(1);
    p.pass_diffuse_weight = zero<PackedSpectrum>(1);
    p.pass_glossy_weight = zero<PackedSpectrum>(1);
    p.render_pixel_index = zero<uint32_t>(1);
    p.visibility = zero<uint8_t>(1);
    p.sample = zero<uint32_t>(1);
  };
  allocate(params.integrator_state.path);
  allocate(params.integrator_state.shadow_path);
  params.integrator_state.shadow_path.lightgroup = zero<uint8_t>(1);
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
  auto output = zero<float>(f::film_lanes);
  auto counts = zero<unsigned>(f::film_lanes);
  auto state = zero<ObservedState>(1);
  for (unsigned i = 0; i < f::cases; ++i) {
    probe<<<1, 1>>>(output, counts, state, f::inputs[i]);
    check(hipGetLastError());
    std::array<float, f::film_lanes> values{};
    std::array<unsigned, f::film_lanes> writes{};
    ObservedState s{};
    check(hipMemcpy(values.data(), output, sizeof(values), hipMemcpyDeviceToHost));
    check(hipMemcpy(writes.data(), counts, sizeof(writes), hipMemcpyDeviceToHost));
    check(hipMemcpy(&s, state, sizeof(s), hipMemcpyDeviceToHost));
    std::printf("F %u %u %u %u %.9g %.9g %.9g %.9g %.9g %.9g", i,
                s.flag, s.bounce, s.visibility,
                s.diffuse.x, s.diffuse.y, s.diffuse.z,
                s.glossy.x, s.glossy.y, s.glossy.z);
    for (auto v : values) { std::printf(" %.9g", v); }
    for (auto v : writes) { std::printf(" %u", v); }
    std::printf("\n");
  }
}
