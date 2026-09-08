// Original Cycles 5.2.1 GPU intersection and evaluation, with authored inputs.
// hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC -std=c++20 -O3 \
//   -ffast-math -I /path/to/blender/intern/cycles this_file -o /tmp/oracle
#include "../tests/cycles_light_endpoint_fixture.h"

// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/light/light.h"
// clang-format on

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace f = psycles::test_support::light_endpoint;

void check(hipError_t e) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(e));
    std::exit(1);
  }
}

__global__ void probe(float4 *output, float3 origin, float3 direction,
                      float minimum, unsigned mode) {
  Ray ray{};
  ray.P = origin;
  ray.D = direction;
  ray.tmin = minimum;
  ray.tmax = f::maximum;
  Intersection isect{};
  isect.prim = PRIM_NONE;
  isect.object = OBJECT_NONE;
  isect.t = f::maximum;
  const auto visibility = (mode == 0u || mode == 3u)
                              ? PATH_RAY_VISIBILITY_CAMERA
                              : PATH_RAY_VISIBILITY_DIFFUSE;
  lights_intersect_impl<true>(nullptr, &ray, &isect, PRIM_NONE, OBJECT_NONE,
                              PRIMITIVE_NONE, visibility, 0u, 0u, OBJECT_NONE,
                              nullptr, 0);
  const bool hit = isect.prim != PRIM_NONE;
  LightEval eval{};
  if (hit) {
    eval = light_eval_from_intersection(nullptr, &isect, ray.P, ray.D,
                                        make_float3(0, 0, 1), 0u);
  }
  *output = make_float4(float(hit), isect.t, eval.eval_fac, eval.pdf);
}

__global__ void probe_visibility(unsigned *output) {
  const auto i = threadIdx.x;
  const auto shader = ((i & 1u) ? SHADER_EXCLUDE_DIFFUSE : 0u) |
                      ((i & 2u) ? SHADER_EXCLUDE_GLOSSY : 0u) |
                      ((i & 4u) ? SHADER_EXCLUDE_TRANSMIT : 0u) |
                      ((i & 8u) ? SHADER_EXCLUDE_CAMERA : 0u) |
                      ((i & 16u) ? SHADER_EXCLUDE_SCATTER : 0u);
  const auto flag = i >= 32u ? PATH_RAY_REFLECT : 0u;
  unsigned mask = 0u;
  for (unsigned visibility = 0u; visibility < 32u; ++visibility) {
    if (is_light_shader_visible_to_path(shader, PathRayVisibility(visibility),
                                        flag)) {
      mask |= 1u << visibility;
    }
  }
  output[i] = mask;
}

int main() {
  KernelLight *light{};
  KernelObject *object{};
  float4 *output{};
  check(hipMalloc(&light, sizeof(KernelLight)));
  check(hipMalloc(&object, sizeof(KernelObject)));
  check(hipMalloc(&output, sizeof(float4)));
  KernelObject object_value{};
  object_value.itfm = {make_float4(1, 0, 0, 0), make_float4(0, 1, 0, 0),
                       make_float4(0, 0, 1, 0)};
  check(hipMemcpy(object, &object_value, sizeof(object_value),
                  hipMemcpyHostToDevice));
  KernelParamsHIP params{};
  params.lights = light;
  params.objects = object;
  params.data.integrator.num_lights = 1;
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
  for (unsigned row = 0u; row < f::inputs.size(); ++row) {
    const auto &input = f::inputs[row];
    for (unsigned mode = 0u; mode < f::modes; ++mode) {
      KernelLight value{};
      value.type = input.type == 0u   ? LIGHT_POINT
                   : input.type == 1u ? LIGHT_SPOT
                                      : LIGHT_AREA;
      value.object_id = 0;
      value.shader_id = mode == 4u ? 0u : SHADER_USE_MIS;
      if (mode == 2u) {
        value.shader_id |= SHADER_EXCLUDE_DIFFUSE;
      }
      if (mode == 3u) {
        value.shader_id |= SHADER_EXCLUDE_CAMERA;
      }
      if (input.type == 2u) {
        value.area.axis_u = make_float3(1, 0, 0);
        value.area.axis_v = make_float3(0, 1, 0);
        value.area.dir = make_float3(0, 0, -1);
        value.area.len_u = value.area.len_v = f::length;
        value.area.invarea = input.ellipse ? -1.0f / M_PI_F : 0.25f;
        value.area.tan_half_spread = std::tan(0.5f * f::spread);
        value.area.normalize_spread =
            1.0f / (value.area.tan_half_spread - 0.5f * f::spread);
      } else {
        value.spot.radius = input.radius;
        value.spot.is_sphere = input.sphere;
        value.spot.eval_fac =
            (input.radius == 0.0f
                 ? 0.25f
                 : 1.0f / (4.0f * M_PI_F * input.radius * input.radius)) *
            M_1_PI_F;
        value.spot.dir = make_float3(0, 0, -1);
        value.spot.cos_half_spot_angle = std::cos(0.5f * f::spot_angle);
        value.spot.cos_half_larger_spread = value.spot.cos_half_spot_angle;
        value.spot.spot_smooth =
            1.0f / ((1.0f - value.spot.cos_half_spot_angle) * f::spot_smooth);
      }
      check(hipMemcpy(light, &value, sizeof(value), hipMemcpyHostToDevice));
      probe<<<1u, 1u>>>(
          output,
          make_float3(input.origin[0], input.origin[1], input.origin[2]),
          make_float3(input.direction[0], input.direction[1],
                      input.direction[2]),
          input.minimum, mode);
      check(hipGetLastError());
      float4 result{};
      check(hipMemcpy(&result, output, sizeof(result), hipMemcpyDeviceToHost));
      std::printf("%u %.9g %.9g %.9g %.9g\n", row * f::modes + mode, result.x,
                  result.y, result.z, result.w);
    }
  }
  unsigned *visibility{};
  check(hipMalloc(&visibility, 64u * sizeof(unsigned)));
  probe_visibility<<<1u, 64u>>>(visibility);
  check(hipGetLastError());
  std::array<unsigned, 64> masks{};
  check(hipMemcpy(masks.data(), visibility, sizeof(masks),
                  hipMemcpyDeviceToHost));
  for (unsigned i = 0u; i < masks.size(); ++i) {
    std::printf("V %u %u\n", i, masks[i]);
  }
  check(hipFree(visibility));
  check(hipFree(output));
  check(hipFree(light));
  check(hipFree(object));
}
