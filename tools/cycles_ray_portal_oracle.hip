// Original Cycles 5.2.1 GPU portal integration and path-state observer.
// This includes and calls the unmodified original integrator functions.
#include "../tests/cycles_ray_portal_fixture.h"

// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/bvh/bvh.h"
#include "kernel/integrator/state_util.h"
#include "kernel/svm/svm.h"
#include "kernel/integrator/shade_surface.h"
// clang-format on

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace f = psycles::test_support::ray_portal;
__device__ f::Input prepared[f::inputs.size()];
void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(error));
    std::exit(1);
  }
}
template<typename T> T *allocate(unsigned count) {
  T *out{};
  check(hipMalloc(&out, count * sizeof(T)));
  check(hipMemset(out, 0, count * sizeof(T)));
  return out;
}
template<typename T> T *upload(const std::vector<T> &values) {
  auto out = allocate<T>(unsigned(values.size()));
  check(hipMemcpy(out, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice));
  return out;
}
__device__ float3 rgb(f::RGB value) {
  return make_float3(value[0], value[1], value[2]);
}
__device__ ClosureType type(f::Kind kind) {
  switch (kind) {
    case f::Kind::portal: return CLOSURE_BSDF_RAY_PORTAL_ID;
    case f::Kind::diffuse: return CLOSURE_BSDF_DIFFUSE_ID;
    case f::Kind::transparent: return CLOSURE_BSDF_TRANSPARENT_ID;
    case f::Kind::bssrdf: return CLOSURE_BSSRDF_BURLEY_ID;
    default: return CLOSURE_HOLDOUT_ID;
  }
}

__global__ void probe(float4 *floats, uint4 *integers, unsigned index) {
  const auto in = prepared[index];
  const IntegratorState state{0};
  ShaderData sd{};
  sd.P = rgb(in.point);
  sd.N = sd.Ng = make_float3(0, 0, 1);
  sd.wi = make_float3(0, 0, -1);
  sd.dP = 0.375f;
  sd.type = in.triangle ? PRIMITIVE_TRIANGLE : PRIMITIVE_CURVE;
  sd.object = 0;
  sd.prim = 0;
  sd.object_flag = in.transform_applied ? SD_OBJECT_TRANSFORM_APPLIED : 0;
  sd.flag = SD_BSDF | (in.portal_shader ? SD_RAY_PORTAL : 0);
  sd.num_closure = int(in.count);
  for (unsigned i = 0; i < in.count; ++i) {
    auto &sc = sd.closure[i];
    sc.type = type(in.closures[i].kind);
    sc.weight = rgb(in.closures[i].weight);
    sc.sample_weight = in.closures[i].sample_weight;
    sc.N = sd.N;
    if (CLOSURE_IS_RAY_PORTAL(sc.type)) {
      auto *pc = reinterpret_cast<RayPortalClosure *>(&sc);
      pc->P = rgb(in.portal_position);
      pc->D = rgb(in.portal_direction);
    }
  }
  INTEGRATOR_STATE_WRITE(state, path, flag) = PATH_RAY_REFLECT | PATH_RAY_TRANSPARENT_BACKGROUND;
  INTEGRATOR_STATE_WRITE(state, path, visibility) = PATH_RAY_VISIBILITY_DIFFUSE;
  INTEGRATOR_STATE_WRITE(state, path, bounce) = 0;
  INTEGRATOR_STATE_WRITE(state, path, diffuse_bounce) = 1;
  INTEGRATOR_STATE_WRITE(state, path, glossy_bounce) = 2;
  INTEGRATOR_STATE_WRITE(state, path, transmission_bounce) = 3;
  INTEGRATOR_STATE_WRITE(state, path, transparent_bounce) = in.transparent_bounce;
  INTEGRATOR_STATE_WRITE(state, path, portal_bounce) = in.portal_bounce;
  INTEGRATOR_STATE_WRITE(state, path, rng_offset) = 49;
  INTEGRATOR_STATE_WRITE(state, path, throughput) = make_float3(0.2f, 0.4f, 0.8f);
  INTEGRATOR_STATE_WRITE(state, path, mis_ray_pdf) = 0.33f;
  INTEGRATOR_STATE_WRITE(state, path, min_ray_pdf) = 0.125f;
  INTEGRATOR_STATE_WRITE(state, path, mis_origin_n) = make_float3(1, 0, 0);
  INTEGRATOR_STATE_WRITE(state, path, continuation_probability) = 0.875f;
  INTEGRATOR_STATE_WRITE(state, path, pass_diffuse_weight) = make_float3(0.25f, 0.5f, 0.75f);
  INTEGRATOR_STATE_WRITE(state, path, pass_glossy_weight) = make_float3(0.75f, 0.5f, 0.25f);
  INTEGRATOR_STATE_WRITE(state, isect, object) = 0;
  INTEGRATOR_STATE_WRITE(state, isect, prim) = 17;
  INTEGRATOR_STATE_WRITE(state, isect, type) = PRIMITIVE_TRIANGLE;
  Ray ray{};
  ray.P = make_float3(7, 8, 9);
  ray.D = make_float3(0, 0, 1);
  ray.tmin = 0.125f;
  ray.tmax = 10.0f;
  ray.dP = 0.25f;
  ray.dD = 0.0625f;
  integrator_state_write_ray(state, &ray);
  float3 random = make_float3(0.125f, 0.25f, in.random_z);
  const auto *sc = in.force_first_closure ? &sd.closure[0]
      : surface_shader_bsdf_bssrdf_pick(&sd, &random);
  int label = int(in.transition_label);
  if (label != 0) {
    path_state_next(nullptr, state, label, sd.flag);
  } else {
    if (!CLOSURE_IS_RAY_PORTAL(sc->type)) { __builtin_trap(); }
    label = integrate_surface_ray_portal(nullptr, state, &sd, sc);
  }
  floats[0] = make_float4(INTEGRATOR_STATE(state, ray, P), INTEGRATOR_STATE(state, ray, tmin));
  floats[1] = make_float4(INTEGRATOR_STATE(state, ray, D), INTEGRATOR_STATE(state, ray, tmax));
  floats[2] = make_float4(INTEGRATOR_STATE(state, path, throughput), INTEGRATOR_STATE(state, ray, dP));
  floats[3] = make_float4(INTEGRATOR_STATE(state, ray, dD), INTEGRATOR_STATE(state, path, mis_ray_pdf),
                         INTEGRATOR_STATE(state, path, min_ray_pdf), random.z);
  floats[4] = make_float4(INTEGRATOR_STATE(state, path, mis_origin_n),
                         INTEGRATOR_STATE(state, path, continuation_probability));
  floats[5] = make_float4(INTEGRATOR_STATE(state, path, pass_diffuse_weight), sc->sample_weight);
  floats[6] = make_float4(INTEGRATOR_STATE(state, path, pass_glossy_weight), 0);
  integers[0] = make_uint4(label, unsigned(sc - sd.closure),
                           INTEGRATOR_STATE(state, path, flag), INTEGRATOR_STATE(state, path, visibility));
  integers[1] = make_uint4(INTEGRATOR_STATE(state, path, bounce), INTEGRATOR_STATE(state, path, diffuse_bounce),
                           INTEGRATOR_STATE(state, path, glossy_bounce), INTEGRATOR_STATE(state, path, transmission_bounce));
  integers[2] = make_uint4(INTEGRATOR_STATE(state, path, transparent_bounce), INTEGRATOR_STATE(state, path, portal_bounce),
                           INTEGRATOR_STATE(state, path, rng_offset), INTEGRATOR_STATE(state, isect, object));
  integers[3] = make_uint4(INTEGRATOR_STATE(state, isect, prim), INTEGRATOR_STATE(state, isect, type), label != LABEL_NONE, sd.flag);
}

int main() {
  KernelParamsHIP params{};
  params.data.kernel_features = KERNEL_FEATURE_PATH_TRACING |
      KERNEL_FEATURE_NODE_PORTAL | KERNEL_FEATURE_LIGHT_PASSES;
  params.data.integrator.max_bounce = 4;
  params.data.integrator.max_diffuse_bounce = 2;
  params.data.integrator.max_glossy_bounce = 4;
  params.data.integrator.max_transmission_bounce = 5;
  KernelObject object{};
  object.tfm = {make_float4(2, 0, 0, 0.25f), make_float4(0, 1.5f, 0, -0.5f), make_float4(0, 0, 1, 2)};
  object.itfm = {make_float4(0.5f, 0, 0, -0.125f), make_float4(0, 2.0f / 3.0f, 0, 1.0f / 3.0f), make_float4(0, 0, 1, -2)};
  params.objects = upload(std::vector<KernelObject>{object});
  // Cycles' packed_uint3 constructors are device-only. Upload the immutable
  // three-uint ABI bytes; do not substitute HIP's padded uint3 layout.
  static_assert(sizeof(packed_uint3) == sizeof(std::array<unsigned, 3>));
  params.tri_vindex = reinterpret_cast<const packed_uint3 *>(
      upload(std::vector<std::array<unsigned, 3>>{{0, 1, 2}}));
  // Immutable geometry, not host-computed intersections or expected offsets.
  auto world_vertices = upload(std::vector<packed_float3>{make_float3(-2, -2, 2), make_float3(3, -2, 2), make_float3(-2, 3, 2)});
  auto local_vertices = upload(std::vector<packed_float3>{make_float3(-2, -2, 0), make_float3(3, -2, 0), make_float3(-2, 3, 0)});
  auto edge_first = upload(std::vector<packed_float3>{make_float3(-2, -2, 3), make_float3(2, -2, 3), make_float3(2, 2, 3)});
  auto edge_second = upload(std::vector<packed_float3>{make_float3(-2, -2, 3), make_float3(2, 2, 3), make_float3(-2, 2, 3)});
  auto &s = params.integrator_state;
#define KERNEL_STRUCT_BEGIN(name) for (int array_index = 0;; ++array_index) {
#define KERNEL_STRUCT_BEGIN_PACKED(parent, feature) KERNEL_STRUCT_BEGIN(parent)
#define KERNEL_STRUCT_MEMBER(parent, type, name, feature) s.parent.name = allocate<type>(1);
#define KERNEL_STRUCT_MEMBER_PACKED KERNEL_STRUCT_MEMBER
#define KERNEL_STRUCT_ARRAY_MEMBER(parent, type, name, feature) s.parent[array_index].name = allocate<type>(1);
#define KERNEL_STRUCT_END(name) (void)array_index; break; }
#define KERNEL_STRUCT_END_ARRAY(name, cpu_size, gpu_size) if (array_index >= int(gpu_size) - 1) break; }
#define KERNEL_STRUCT_VOLUME_STACK_SIZE 1
#include "kernel/integrator/state_template.h"
  check(hipMemcpyToSymbol(HIP_SYMBOL(prepared), f::inputs.data(), sizeof(f::inputs)));
  auto floats = allocate<float4>(f::float_rows);
  auto integers = allocate<uint4>(f::integer_rows);
  std::vector<float4> values(f::float_rows);
  std::vector<uint4> metadata(f::integer_rows);
  for (unsigned i = 0; i < f::inputs.size(); ++i) {
    params.data.integrator.transparent_max_bounce = f::inputs[i].transparent_limit;
    params.tri_verts = f::inputs[i].transform_applied ? world_vertices : local_vertices;
    if (f::inputs[i].triangle == 2) params.tri_verts = edge_first;
    if (f::inputs[i].triangle == 3) params.tri_verts = edge_second;
    check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
    probe<<<1, 1>>>(floats, integers, i);
    check(hipGetLastError());
    check(hipMemcpy(values.data(), floats, values.size() * sizeof(float4), hipMemcpyDeviceToHost));
    check(hipMemcpy(metadata.data(), integers, metadata.size() * sizeof(uint4), hipMemcpyDeviceToHost));
    std::printf("%u", i);
    for (auto v : values) { std::printf(" %.9g %.9g %.9g %.9g", v.x, v.y, v.z, v.w); }
    for (auto v : metadata) { std::printf(" %u %u %u %u", v.x, v.y, v.z, v.w); }
    std::printf("\n");
  }
  check(hipDeviceReset());
}
