// Genuine Cycles HIP shader_setup_from_ray, surface_shader_eval and
// integrate_transparent_shadow. No translated reference shading/integrator.
#include "../tests/cycles_svm_native_shadow_fixture.h"

// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/camera/camera.h"
#include "kernel/geom/primitive.h"
#include "kernel/integrator/shade_shadow.h"
// clang-format on

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
using namespace psycles::test_support;
constexpr unsigned count = native_shadow_inputs.size();
constexpr unsigned batch_count = native_shadow_batches.size();

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
__global__ void pack_normals(const NativeShadowInput *inputs,
                             packed_normal *out) {
  const auto i = threadIdx.x;
  if (i < count * 3u) {
    const auto n = inputs[i / 3u].geometry.normals[i % 3u];
    out[i] = packed_normal{make_float3(n.x, n.y, n.z)};
  }
}
ccl_device void init_state(IntegratorShadowState state,
                           const NeeSetupInput &input) {
  INTEGRATOR_STATE_WRITE(state, shadow_path, sample) = input.sample;
  INTEGRATOR_STATE_WRITE(state, shadow_path, rng_pixel) = input.rng_hash;
  INTEGRATOR_STATE_WRITE(state, shadow_path, rng_offset) = input.rng_offset;
  INTEGRATOR_STATE_WRITE(state, shadow_path, bounce) = 7;
  INTEGRATOR_STATE_WRITE(state, shadow_path, transparent_bounce) = 3;
  INTEGRATOR_STATE_WRITE(state, shadow_path, diffuse_bounce) = 2;
  INTEGRATOR_STATE_WRITE(state, shadow_path, glossy_bounce) = 4;
  INTEGRATOR_STATE_WRITE(state, shadow_path, transmission_bounce) = 1;
  INTEGRATOR_STATE_WRITE(state, shadow_path, volume_bounds_bounce) = 0;
  INTEGRATOR_STATE_WRITE(state, shadow_path, throughput) = one_spectrum();
  INTEGRATOR_STATE_ARRAY_WRITE(state, shadow_volume_stack, 0, shader) =
      SHADER_NONE;
  Ray ray{};
  ray.P = make_float3(input.ray_P.x, input.ray_P.y, input.ray_P.z);
  ray.D = make_float3(input.ray_D.x, input.ray_D.y, input.ray_D.z);
  ray.tmax = 100.0f;
  ray.time = input.time;
  ray.dP = input.dP;
  ray.dD = input.dD;
  integrator_state_write_shadow_ray(state, &ray);
}
__global__ void probe(const NativeShadowInput *inputs, float4 *out,
                      uint4 *meta) {
  const auto i = threadIdx.x;
  if (i >= count)
    return;
  const IntegratorShadowState state{int(i)};
  const auto &input = inputs[i];
  init_state(state, input.geometry);
  Intersection isect{};
  isect.object = isect.prim = i;
  isect.type = PRIMITIVE_TRIANGLE;
  isect.t = input.geometry.distance;
  isect.u = input.u;
  isect.v = input.v;
  integrator_state_write_shadow_isect(state, &isect, 0);
  Ray ray{};
  integrator_state_read_shadow_ray(state, &ray);
  ShaderDataTinyStorage storage{};
  auto *sd = AS_SHADER_DATA(&storage);
  shader_setup_from_ray(nullptr, sd, &ray, &isect);
  sd->lcg_state = integrator_state_lcg_init(state, 0xb4bc3953u);
  out[i * 8u] = make_float4(sd->P, sd->time);
  out[i * 8u + 1u] = make_float4(sd->N, sd->ray_length);
  out[i * 8u + 2u] = make_float4(sd->Ng, sd->dP);
  out[i * 8u + 3u] = make_float4(sd->wi, sd->dI);
  out[i * 8u + 4u] = make_float4(sd->dPdu, sd->u);
  out[i * 8u + 5u] = make_float4(sd->dPdv, sd->v);
  out[i * 8u + 6u] = make_float4(sd->du.dx, sd->du.dy, sd->dv.dx, sd->dv.dy);
  meta[i * 2u] = make_uint4(sd->shader, sd->flag, sd->object_flag, sd->type);
  meta[i * 2u + 1u] = make_uint4(sd->object, sd->prim, sd->lcg_state, 0u);
  ShaderEvalResult result = SHADER_EVAL_EMPTY;
  const auto transparency =
      integrate_transparent_surface_shadow(nullptr, state, 0, result);
  out[i * 8u + 7u] = make_float4(
      transparency,
      float(INTEGRATOR_STATE(state, shadow_path, volume_bounds_bounce)));
}
__global__ void probe_batch(const NativeShadowInput *inputs,
                            const NativeShadowBatchInput *batches, float4 *out,
                            uint4 *meta) {
  const auto i = threadIdx.x;
  if (i >= batch_count)
    return;
  const IntegratorShadowState state{int(i)};
  init_state(state, inputs[0].geometry);
  const auto &input = batches[i];
  INTEGRATOR_STATE_WRITE(state, shadow_path, throughput) =
      make_float3(input.throughput.x, input.throughput.y, input.throughput.z);
  INTEGRATOR_STATE_WRITE(state, shadow_path, transparent_bounce) =
      input.transparent_bounce;
  INTEGRATOR_STATE_WRITE(state, shadow_path, rng_offset) = input.rng_offset;
  INTEGRATOR_STATE_WRITE(state, shadow_path, volume_bounds_bounce) =
      input.volume_bounds_bounce;
  // Cycles' intersection stage orders hits before transparent integration.
  unsigned order[4]{0u, 1u, 2u, 3u};
  for (auto a = 0u; a < input.count; ++a) {
    for (auto b = a + 1u; b < input.count; ++b) {
      if (input.distances[order[b]] < input.distances[order[a]]) {
        const auto tmp = order[a];
        order[a] = order[b];
        order[b] = tmp;
      }
    }
    const auto j = order[a];
    Intersection isect{};
    isect.object = isect.prim = input.objects[j];
    isect.type = PRIMITIVE_TRIANGLE;
    isect.t = input.distances[j];
    isect.u = inputs[isect.object].u;
    isect.v = inputs[isect.object].v;
    integrator_state_write_shadow_isect(state, &isect, a);
  }
  const auto packed = shadow_num_hits_pack(input.count, 0u, false);
  const auto result = integrate_transparent_shadow(nullptr, state, packed);
  out[i] = make_float4(INTEGRATOR_STATE(state, shadow_path, throughput),
                       INTEGRATOR_STATE(state, shadow_ray, tmin));
  const auto continued = result == TRANSPARENT_SHADOW_EVAL_CONTINUE;
  meta[i * 2u] =
      make_uint4(continued && shadow_intersections_has_remaining(packed),
                 continued && !shadow_intersections_has_remaining(packed),
                 INTEGRATOR_STATE(state, shadow_path, transparent_bounce),
                 INTEGRATOR_STATE(state, shadow_path, rng_offset));
  meta[i * 2u + 1u] =
      make_uint4(INTEGRATOR_STATE(state, shadow_path, volume_bounds_bounce),
                 result, 0u, 0u);
}
} // namespace

int main() {
  std::vector<NativeShadowInput> inputs(native_shadow_inputs.begin(),
                                        native_shadow_inputs.end());
  std::vector<KernelObject> objects(count);
  std::vector<unsigned> object_flags(count), tri_shaders(count);
  std::vector<KernelShader> shaders(count);
  using HostIndex = psycles::compiler::cycles_svm::packed_uint3;
  std::vector<HostIndex> indices(count, {0u, 1u, 2u});
  std::vector<packed_float3> vertices(count * 3u);
  for (auto i = 0u; i < count; ++i) {
    const auto &input = inputs[i].geometry;
    auto &object = objects[i];
    std::memcpy(&object.tfm, &input.tfm, sizeof(input.tfm));
    std::memcpy(&object.itfm, &input.itfm, sizeof(input.itfm));
    object.position_offset = int(i * 3u);
    object.normal_offset =
        (input.object_flags & SD_OBJECT_HAS_CORNER_NORMALS) ? 0 : int(i * 3u);
    object.primitive_type = PRIMITIVE_TRIANGLE;
    object_flags[i] = input.object_flags;
    tri_shaders[i] = input.shader;
    shaders[i].flags = inputs[i].shader_flags;
    for (auto v = 0u; v < 3u; ++v) {
      const auto p = input.vertices[v];
      vertices[i * 3u + v] = make_float3(p.x, p.y, p.z);
    }
  }
  KernelParamsHIP params{};
  params.objects = upload(objects);
  params.object_flag = upload(object_flags);
  params.tri_vindex = reinterpret_cast<const packed_uint3 *>(upload(indices));
  params.tri_verts = upload(vertices);
  params.tri_shader = upload(tri_shaders);
  packed_normal *normals{};
  check(hipMalloc(&normals, count * 3u * sizeof(packed_normal)));
  params.attributes_normal = normals;
  params.shaders = upload(shaders);
  params.svm_nodes = upload(make_native_shadow_image());
  params.data.kernel_features = KERNEL_FEATURE_PATH_TRACING;
  params.data.integrator.caustics_reflective = true;
  params.data.integrator.caustics_refractive = true;
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
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
  const auto *device_inputs = upload(inputs);
  const auto *device_batches = upload(std::vector<NativeShadowBatchInput>(
      native_shadow_batches.begin(), native_shadow_batches.end()));
  pack_normals<<<1u, 128u>>>(device_inputs, normals);
  float4 *out{};
  uint4 *meta{};
  check(hipMalloc(&out, count * 8u * sizeof(float4)));
  check(hipMalloc(&meta, count * 2u * sizeof(uint4)));
  probe<<<1u, 32u>>>(device_inputs, out, meta);
  check(hipGetLastError());
  std::vector<float4> values(count * 8u);
  std::vector<uint4> integers(count * 2u);
  check(hipMemcpy(values.data(), out, values.size() * sizeof(float4),
                  hipMemcpyDeviceToHost));
  check(hipMemcpy(integers.data(), meta, integers.size() * sizeof(uint4),
                  hipMemcpyDeviceToHost));
  for (auto i = 0u; i < count; ++i) {
    std::printf("S %u", i);
    for (auto j = 0u; j < 8u; ++j) {
      const auto v = values[i * 8u + j];
      std::printf(" %.9g %.9g %.9g %.9g", v.x, v.y, v.z, v.w);
    }
    for (auto j = 0u; j < 2u; ++j) {
      const auto v = integers[i * 2u + j];
      std::printf(" %u %u %u %u", v.x, v.y, v.z, v.w);
    }
    std::printf("\n");
  }
  probe_batch<<<1u, 32u>>>(device_inputs, device_batches, out, meta);
  check(hipGetLastError());
  check(hipMemcpy(values.data(), out, batch_count * sizeof(float4),
                  hipMemcpyDeviceToHost));
  check(hipMemcpy(integers.data(), meta, batch_count * 2u * sizeof(uint4),
                  hipMemcpyDeviceToHost));
  for (auto i = 0u; i < batch_count; ++i) {
    const auto v = values[i];
    std::printf("B %u %.9g %.9g %.9g %.9g", i, v.x, v.y, v.z, v.w);
    for (auto j = 0u; j < 2u; ++j) {
      const auto u = integers[i * 2u + j];
      std::printf(" %u %u %u %u", u.x, u.y, u.z, u.w);
    }
    std::printf("\n");
  }
  check(hipDeviceReset());
}
