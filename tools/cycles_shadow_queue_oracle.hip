// Queue-boundary oracle from the unmodified external Cycles 5.2.1 integrator.
// Only the BVH calls are replaced with prepared traversal results. There is
// no reference renderer and no copied queue/state transition implementation.
#include "../tests/cycles_shadow_queue_fixture.h"

// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/bvh/bvh.h"
#include "kernel/integrator/state_util.h"
// clang-format on

using namespace psycles::test_support;

__device__ ShadowQueueInput prepared[shadow_queue_inputs.size()];

ccl_device bool prepared_shadow(KernelGlobals, const Ray *ray, uint) {
  return prepared[unsigned(ray->P.x)].blocked != 0u;
}
ccl_device void prepared_shadow_all(KernelGlobals, IntegratorShadowState state,
                                    const Ray *ray, uint, uint, uint *num_hits,
                                    float *throughput) {
  const auto input = prepared[unsigned(ray->P.x)];
  *num_hits = input.count;
  *throughput = input.blocked ? 0.0f : 1.0f;
  for (auto j = 0u; j < input.count; ++j) {
    Intersection isect{};
    isect.t = float(input.count - j); // Exercise genuine Cycles sorting too.
    integrator_state_write_shadow_isect(state, &isect, j);
  }
}

#define scene_intersect_shadow prepared_shadow
#define scene_intersect_shadow_all prepared_shadow_all
#include "kernel/integrator/intersect_shadow.h"
#undef scene_intersect_shadow_all
#undef scene_intersect_shadow

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(error));
    std::exit(1);
  }
}
template <typename T> T *upload(const std::vector<T> &values) {
  T *out{};
  check(hipMalloc(&out, values.size() * sizeof(T)));
  check(hipMemcpy(out, values.data(), values.size() * sizeof(T),
                  hipMemcpyHostToDevice));
  return out;
}
__global__ void probe(uint4 *out) {
  // Serial probe keeps the genuine global queue counters independently
  // observable for each case; state itself still uses the actual GPU SoA.
  for (auto i = 0u; i < shadow_queue_inputs.size(); ++i) {
    const IntegratorShadowState state{int(i)};
    auto *counters = kernel_integrator_state.queue_counter;
    counters->num_queued[DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW] = 1u;
    counters->num_queued[DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW] = 0u;
    INTEGRATOR_STATE_WRITE(state, shadow_path, queued_kernel) =
        DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW;
    INTEGRATOR_STATE_WRITE(state, shadow_path, throughput) = one_spectrum();
    Ray ray{};
    ray.P = make_float3(float(i), 0.0f, 0.0f);
    ray.D = make_float3(0.0f, 0.0f, 1.0f);
    ray.tmax = 100.0f;
    integrator_state_write_shadow_ray(state, &ray);
    integrator_intersect_shadow(nullptr, state);
    out[i] = make_uint4(
        INTEGRATOR_STATE(state, shadow_path, queued_kernel),
        counters->num_queued[DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW],
        counters->num_queued[DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW],
        integrator_shadow_path_is_terminated(state));
  }
}
} // namespace

int main() {
  constexpr auto count = shadow_queue_inputs.size();
  KernelParamsHIP params{};
  params.data.integrator.transparent_max_bounce = 16;
  auto &s = params.integrator_state;
#define KERNEL_STRUCT_BEGIN(name) for (int array_index = 0;; ++array_index) {
#define KERNEL_STRUCT_BEGIN_PACKED(parent, feature) KERNEL_STRUCT_BEGIN(parent)
#define KERNEL_STRUCT_MEMBER(parent, type, name, feature)                      \
  s.parent.name = upload(std::vector<type>(count));
#define KERNEL_STRUCT_MEMBER_PACKED KERNEL_STRUCT_MEMBER
#define KERNEL_STRUCT_ARRAY_MEMBER(parent, type, name, feature)                \
  s.parent[array_index].name = upload(std::vector<type>(count));
#define KERNEL_STRUCT_END(name)                                                \
  (void)array_index;                                                           \
  break;                                                                       \
  }
#define KERNEL_STRUCT_END_ARRAY(name, cpu_size, gpu_size)                      \
  if (array_index >= int(gpu_size) - 1)                                        \
    break;                                                                     \
  }
#define KERNEL_STRUCT_VOLUME_STACK_SIZE 1
#include "kernel/integrator/shadow_state_template.h"
  s.queue_counter = upload(std::vector<IntegratorQueueCounter>(1u));
  check(hipMemcpyToSymbol(HIP_SYMBOL(prepared), shadow_queue_inputs.data(),
                          sizeof(shadow_queue_inputs)));
  uint4 *out{};
  check(hipMalloc(&out, count * sizeof(uint4)));
  std::printf("kernels %u %u\n", DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW,
              DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW);
  for (auto transparent : {0, 1}) {
    params.data.integrator.transparent_shadows = transparent;
    check(
        hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
    probe<<<1u, 1u>>>(out);
    check(hipGetLastError());
    std::array<uint4, count> rows{};
    check(hipMemcpy(rows.data(), out, sizeof(rows), hipMemcpyDeviceToHost));
    for (auto i = 0u; i < count; ++i) {
      const auto r = rows[i];
      std::printf("Q %d %u %u %u %u %u\n", transparent, i, r.x, r.y, r.z, r.w);
    }
  }
  check(hipDeviceReset());
}
