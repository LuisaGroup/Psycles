// External Cycles 5.2.1 shader_setup and light geometry, executed on HIP.
#include "../tests/cycles_svm_nee_setup_fixture.h"

// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/geom/shader_data.h"
#include "kernel/light/light.h"
#include "kernel/light/triangle.h"
#include "kernel/sample/lcg.h"
// clang-format on

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
using psycles::test_support::nee_setup_cases;
using psycles::test_support::NeeSetupInput;
using psycles::test_support::NeeSetupKind;
constexpr auto count = static_cast<unsigned>(nee_setup_cases.size());
static_assert(psycles::test_support::nee_shader_smooth_normal ==
              SHADER_SMOOTH_NORMAL);

void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(error));
    std::exit(1);
  }
}

template <typename T> T *upload(const std::vector<T> &values) {
  T *result{};
  check(hipMalloc(&result, values.size() * sizeof(T)));
  check(hipMemcpy(result, values.data(), values.size() * sizeof(T),
                  hipMemcpyHostToDevice));
  return result;
}

__global__ void pack_normals(const NeeSetupInput *inputs, packed_normal *out) {
  const auto i = threadIdx.x;
  if (i < count * 3u) {
    const auto n = inputs[i / 3u].normals[i % 3u];
    out[i] = packed_normal{make_float3(n.x, n.y, n.z)};
  }
}

__global__ void probe(const NeeSetupInput *inputs, float4 *out, uint4 *meta) {
  const auto i = threadIdx.x;
  if (i >= count) {
    return;
  }
  const auto &input = inputs[i];
  Ray ray{};
  ray.P = make_float3(input.ray_P.x, input.ray_P.y, input.ray_P.z);
  ray.D = make_float3(input.ray_D.x, input.ray_D.y, input.ray_D.z);
  ray.tmax = input.distance;
  ray.dP = input.dP;
  ray.dD = input.dD;
  ray.time = input.time;
  ShaderDataTinyStorage storage{};
  auto *sd = AS_SHADER_DATA(&storage);
  const auto background = input.kind == NeeSetupKind::background;
  if (background) {
    shader_setup_from_background(nullptr, sd, ray.P, ray.D,
                                 min(ray.dD, kernel_data.background.map_dD),
                                 ray.time);
  } else if (input.kind == NeeSetupKind::triangle) {
    Intersection isect{};
    isect.object = isect.prim = i;
    isect.type = PRIMITIVE_TRIANGLE;
    isect.t = ray.tmax;
    const auto uv = triangle_light_uv(nullptr, i, i, ray.time, ray.P, ray.D);
    isect.u = uv.x;
    isect.v = uv.y;
    shader_setup_from_ray(nullptr, sd, &ray, &isect);
  } else {
    const auto *light = &kernel_data_fetch(lights, i);
    const auto P = ray.tmax == FLT_MAX ? -ray.D : ray.P + ray.tmax * ray.D;
    float3 normal = zero_float3();
    float2 uv = zero_float2();
    light_normal_uv_from_position(nullptr, light, P, ray.D, normal, uv);
    shader_setup_from_sample(nullptr, sd, P, normal, -ray.D, 0u, i, i, uv.x,
                             uv.y, ray.tmax, ray.time, false, true);
  }
  sd->lcg_state = lcg_state_init(input.rng_hash, input.rng_offset, input.sample,
                                 0xb4bc3953u);
  out[i * 8u] = make_float4(sd->P, sd->time);
  out[i * 8u + 1u] = make_float4(sd->N, sd->ray_length);
  out[i * 8u + 2u] = make_float4(sd->Ng, sd->dP);
  out[i * 8u + 3u] = make_float4(sd->wi, sd->dI);
  out[i * 8u + 4u] = make_float4(sd->dPdu, sd->u);
  out[i * 8u + 5u] = make_float4(sd->dPdv, sd->v);
  out[i * 8u + 6u] = make_float4(sd->du.dx, sd->du.dy, sd->dv.dx, sd->dv.dy);
  // ray_P is only defined/read for background setup in Cycles.
  out[i * 8u + 7u] =
      make_float4(background ? sd->ray_P : zero_float3(), float(background));
  meta[i * 2u] = make_uint4(sd->shader, sd->flag, sd->object_flag, sd->type);
  meta[i * 2u + 1u] = make_uint4(sd->object, sd->prim, sd->lcg_state, 0u);
}
} // namespace

int main() {
  std::vector<NeeSetupInput> inputs(nee_setup_cases.begin(),
                                    nee_setup_cases.end());
  std::vector<KernelObject> objects(count);
  std::vector<unsigned> object_flags(count), tri_shaders(count);
  using HostIndex = psycles::compiler::cycles_svm::packed_uint3;
  static_assert(sizeof(HostIndex) == sizeof(packed_uint3));
  std::vector<HostIndex> indices(count);
  std::vector<packed_float3> vertices(count * 3u);
  std::vector<KernelLight> lights(count);
  for (auto i = 0u; i < count; ++i) {
    const auto &input = inputs[i];
    auto &object = objects[i];
    static_assert(sizeof(object.tfm) == sizeof(input.tfm));
    std::memcpy(&object.tfm, &input.tfm, sizeof(input.tfm));
    std::memcpy(&object.itfm, &input.itfm, sizeof(input.itfm));
    object.position_offset = int(i * 3u);
    object.normal_offset =
        (input.object_flags & SD_OBJECT_HAS_CORNER_NORMALS) ? 0 : int(i * 3u);
    object.primitive_type = input.kind == NeeSetupKind::triangle
                                ? PRIMITIVE_TRIANGLE
                                : PRIMITIVE_LAMP;
    object_flags[i] = input.object_flags;
    tri_shaders[i] = input.shader;
    indices[i].x = 0u;
    indices[i].y = 1u;
    indices[i].z = 2u;
    for (auto v = 0u; v < 3u; ++v) {
      const auto p = input.vertices[v];
      vertices[i * 3u + v] = make_float3(p.x, p.y, p.z);
    }
    auto &light = lights[i];
    light.object_id = i;
    light.shader_id = 0;
    light.co = make_float3(input.tfm.x.w, input.tfm.y.w, input.tfm.z.w);
    light.type = input.kind == NeeSetupKind::spot   ? LIGHT_SPOT
                 : input.kind == NeeSetupKind::area ? LIGHT_AREA
                 : input.kind == NeeSetupKind::sun  ? LIGHT_SUN
                                                    : LIGHT_POINT;
    if (input.kind == NeeSetupKind::area) {
      light.area.axis_u = make_float3(1.0f, 0.0f, 0.0f);
      light.area.axis_v = make_float3(0.0f, 1.0f, 0.0f);
      light.area.dir = make_float3(0.0f, 0.0f, -1.0f);
      light.area.len_u = input.size_u;
      light.area.len_v = input.size_v;
    } else if (input.kind == NeeSetupKind::sun) {
      light.co = make_float3(0.0f, 0.0f, -1.0f);
      light.sun.half_inv_sin_half_angle =
          input.angle == 0.0f ? 0.0f : 0.5f / std::sin(0.25f * input.angle);
    } else {
      light.spot.is_sphere = input.sphere && input.radius != 0.0f;
      light.spot.radius = input.radius;
      light.spot.half_cot_half_spot_angle = 0.5f / std::tan(0.5f * input.angle);
    }
  }
  KernelParamsHIP params{};
  params.objects = upload(objects);
  params.object_flag = upload(object_flags);
  params.tri_vindex = reinterpret_cast<const packed_uint3 *>(upload(indices));
  params.tri_verts = upload(vertices);
  params.tri_shader = upload(tri_shaders);
  packed_normal *device_normals{};
  check(hipMalloc(&device_normals, count * 3u * sizeof(packed_normal)));
  params.attributes_normal = device_normals;
  params.lights = upload(lights);
  params.shaders = upload(std::vector<KernelShader>(1u));
  params.data.background.surface_shader = 0;
  params.data.background.map_dD = std::min(M_PI_F / 256.0f, M_2PI_F / 512.0f);
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
  auto *device_inputs = upload(inputs);
  pack_normals<<<1u, 64u>>>(device_inputs, device_normals);
  check(hipGetLastError());
  float4 *output{};
  uint4 *meta{};
  check(hipMalloc(&output, count * 8u * sizeof(float4)));
  check(hipMalloc(&meta, count * 2u * sizeof(uint4)));
  probe<<<1u, 64u>>>(device_inputs, output, meta);
  check(hipGetLastError());
  std::vector<float4> values(count * 8u);
  std::vector<uint4> metadata(count * 2u);
  check(hipMemcpy(values.data(), output, values.size() * sizeof(float4),
                  hipMemcpyDeviceToHost));
  check(hipMemcpy(metadata.data(), meta, metadata.size() * sizeof(uint4),
                  hipMemcpyDeviceToHost));
  for (auto i = 0u; i < count; ++i) {
    std::printf("%u", i);
    for (auto j = 0u; j < 8u; ++j) {
      const auto v = values[i * 8u + j];
      std::printf(" %.9g %.9g %.9g %.9g", v.x, v.y, v.z, v.w);
    }
    for (auto j = 0u; j < 2u; ++j) {
      const auto v = metadata[i * 2u + j];
      std::printf(" %u %u %u %u", v.x, v.y, v.z, v.w);
    }
    std::printf("\n");
  }
  check(hipDeviceReset());
}
