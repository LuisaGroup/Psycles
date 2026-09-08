// Original Cycles 5.2.1 GPU closest -> lamp -> closest state machine.
// The empty scene's BVH result is supplied; lamp intersection, evaluation,
// transparent termination, queue transitions and integrator state are original.
#include "../tests/cycles_lamp_routing_fixture.h"

// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/bvh/bvh.h"
#include "kernel/integrator/state_util.h"
// clang-format on

namespace f = psycles::test_support::lamp_routing;

__device__ unsigned traversal_calls;
__device__ int2 last_traversal_self;

ccl_device bool empty_scene(KernelGlobals, const Ray *ray, uint,
                            Intersection *isect) {
  ++traversal_calls;
  last_traversal_self = make_int2(ray->self.object, ray->self.prim);
  isect->t = ray->tmax;
  isect->type = PRIMITIVE_NONE;
  return false;
}

#define scene_intersect empty_scene
#include "kernel/integrator/intersect_closest.h"
#undef scene_intersect
#include "kernel/integrator/shade_light.h"

#include <cmath>
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
template <typename T> T *zero_state() {
  T *out{};
  check(hipMalloc(&out, sizeof(T)));
  check(hipMemset(out, 0, sizeof(T)));
  return out;
}

__global__ void probe(uint4 *output, float *distance, float *film,
                      unsigned mode) {
  const IntegratorState state{0};
  auto *counters = kernel_integrator_state.queue_counter;
  for (unsigned i = 0; i < DEVICE_KERNEL_NUM; ++i) {
    counters->num_queued[i] = 0;
  }
  counters->num_queued[DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST] = 1;
  INTEGRATOR_STATE_WRITE(state, path, queued_kernel) =
      DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST;
  INTEGRATOR_STATE_WRITE(state, path, flag) =
      PATH_RAY_MIS_SKIP | PATH_RAY_REFLECT;
  INTEGRATOR_STATE_WRITE(state, path, visibility) =
      mode == 0 ? PATH_RAY_VISIBILITY_CAMERA : PATH_RAY_VISIBILITY_DIFFUSE;
  INTEGRATOR_STATE_WRITE(state, path, throughput) = one_spectrum();
  INTEGRATOR_STATE_WRITE(state, path, transparent_bounce) = 0;
  INTEGRATOR_STATE_WRITE(state, path, rng_offset) = f::rng_offset;
  INTEGRATOR_STATE_ARRAY_WRITE(state, volume_stack, 0, shader) = SHADER_NONE;
  INTEGRATOR_STATE_ARRAY_WRITE(state, volume_stack, 0, object) = OBJECT_NONE;
  Intersection isect{};
  isect.object = f::initial_object;
  isect.prim = f::initial_primitive;
  isect.type = PRIMITIVE_TRIANGLE;
  integrator_state_write_isect(state, &isect);
  Ray ray{};
  ray.D = make_float3(0, 0, 1);
  ray.tmax = f::maximum;
  integrator_state_write_ray(state, &ray);
  traversal_calls = 0;
  unsigned lamps = 0;
  int2 last_lamp = make_int2(f::initial_object, f::initial_primitive);
  for (unsigned step = 0; step < 32; ++step) {
    const auto queued = INTEGRATOR_STATE(state, path, queued_kernel);
    if (queued == DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST) {
      integrator_intersect_closest(nullptr, state, film);
    } else if (queued == DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_FORWARD) {
      ++lamps;
      last_lamp = make_int2(INTEGRATOR_STATE(state, isect, object),
                            INTEGRATOR_STATE(state, isect, prim));
      integrator_shade_light_forward(nullptr, state, film);
    } else {
      break;
    }
  }
  output[0] = make_uint4(traversal_calls, lamps,
                         INTEGRATOR_STATE(state, path, queued_kernel) ==
                             DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND,
                         INTEGRATOR_STATE(state, path, transparent_bounce));
  output[1] = make_uint4(INTEGRATOR_STATE(state, path, rng_offset), last_lamp.x,
                         last_lamp.y, 0);
  output[2] = make_uint4(last_traversal_self.x, last_traversal_self.y, 0, 0);
  *distance = INTEGRATOR_STATE(state, ray, tmin);
}
} // namespace

int main() {
  KernelParamsHIP params{};
  params.data.integrator.num_lights = f::light_count;
  params.data.integrator.use_light_mis = true;
  std::vector<KernelLight> lights(f::light_count);
  std::vector<KernelObject> objects(f::initial_object + 1);
  for (unsigned i = 0; i < f::light_count; ++i) {
    auto &light = lights[i];
    light.type = LIGHT_SPOT;
    light.co = make_float3(0, 0, float(i + 1));
    light.object_id = f::first_light_object + i;
    light.shader_id = SHADER_USE_MIS;
    light.spot.radius = f::radius;
    light.spot.is_sphere = true;
    light.spot.dir = make_float3(-f::axis, 0, -f::axis);
    light.spot.cos_half_spot_angle =
        static_cast<float>(std::cos(0.5 * f::spot_angle));
    light.spot.cos_half_larger_spread = light.spot.cos_half_spot_angle;
    light.spot.spot_smooth =
        1.0f / ((1.0f - light.spot.cos_half_spot_angle) * 0.5f);
    light.spot.eval_fac = 1.0f;
    objects[light.object_id].tfm = {
        make_float4(f::axis, 0, f::axis, 0), make_float4(0, 1, 0, 0),
        make_float4(-f::axis, 0, f::axis, float(i + 1))};
    objects[light.object_id].itfm = {
        make_float4(f::axis, 0, -f::axis, f::axis * float(i + 1)),
        make_float4(0, 1, 0, 0),
        make_float4(f::axis, 0, f::axis, -f::axis * float(i + 1))};
  }
  KernelShader shader{};
  shader.flags = SD_HAS_CONSTANT_EMISSION;
  // Enabled lamps, zero attenuation along the fixture ray.
  for (auto &channel : shader.constant_emission) {
    channel = 1.0f;
  }
  params.lights = upload(lights);
  params.objects = upload(objects);
  params.shaders = upload(std::vector<KernelShader>{shader});
  auto &s = params.integrator_state;
#define KERNEL_STRUCT_BEGIN(name) for (int array_index = 0;; ++array_index) {
#define KERNEL_STRUCT_BEGIN_PACKED(parent, feature) KERNEL_STRUCT_BEGIN(parent)
#define KERNEL_STRUCT_MEMBER(parent, type, name, feature)                      \
  s.parent.name = zero_state<type>();
#define KERNEL_STRUCT_MEMBER_PACKED KERNEL_STRUCT_MEMBER
#define KERNEL_STRUCT_ARRAY_MEMBER(parent, type, name, feature)                \
  s.parent[array_index].name = zero_state<type>();
#define KERNEL_STRUCT_END(name)                                                \
  (void)array_index;                                                           \
  break;                                                                       \
  }
#define KERNEL_STRUCT_END_ARRAY(name, cpu_size, gpu_size)                      \
  if (array_index >= int(gpu_size) - 1)                                        \
    break;                                                                     \
  }
#define KERNEL_STRUCT_VOLUME_STACK_SIZE 1
#include "kernel/integrator/state_template.h"
  s.queue_counter = upload(std::vector<IntegratorQueueCounter>(1));
  auto output = upload(std::vector<uint4>(3));
  auto distance = upload(std::vector<float>(1));
  auto film = upload(std::vector<float>(16384));
  for (unsigned mode = 0; mode < f::modes; ++mode) {
    for (unsigned index = 0; index < f::transparent_limits.size(); ++index) {
      params.data.integrator.transparent_max_bounce =
          f::transparent_limits[index];
      check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params,
                              sizeof(params)));
      probe<<<1, 1>>>(output, distance, film, mode);
      check(hipGetLastError());
      std::array<uint4, 3> result{};
      float t{};
      check(hipMemcpy(result.data(), output, sizeof(result),
                      hipMemcpyDeviceToHost));
      check(hipMemcpy(&t, distance, sizeof(t), hipMemcpyDeviceToHost));
      std::printf("Q %u %u", mode, index);
      for (const auto v : result) {
        std::printf(" %u %u %u %u", v.x, v.y, v.z, v.w);
      }
      std::printf(" %.9g\n", t);
    }
  }
  check(hipDeviceReset());
}
