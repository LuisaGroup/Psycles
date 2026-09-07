// Original Cycles 5.2.1 GPU geometry, not a translated reference renderer.
// hipcc --offload-arch=gfx1201 -DHIPCC -std=c++20 -O3 -ffast-math \
//   -I /path/to/blender/intern/cycles -I include \
//   tools/cycles_svm_curve_setup_oracle.hip -o /tmp/curve-setup-oracle
#include "../tests/cycles_svm_curve_setup_fixture.h"

// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/geom/primitive.h"
#include "kernel/geom/shader_data.h"
// clang-format on

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
using namespace psycles::test_support;
constexpr unsigned count = curve_setup_inputs.size();
void check(hipError_t e) {
  if (e != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(e)); std::exit(1); }
}
template <typename T> T *upload(const std::vector<T> &data) {
  T *p{};
  check(hipMalloc(&p, data.size() * sizeof(T)));
  check(hipMemcpy(p, data.data(), data.size() * sizeof(T), hipMemcpyHostToDevice));
  return p;
}
__global__ void probe(const CurveSetupInput *inputs, float4 *output, uint4 *meta) {
  const auto i = threadIdx.x;
  if (i >= count) { return; }
  const auto &c = inputs[i];
  Ray ray{};
  ray.P = make_float3(c.ray_P.x, c.ray_P.y, c.ray_P.z);
  ray.D = make_float3(c.ray_D.x, c.ray_D.y, c.ray_D.z);
  ray.tmax = 100.0f;
  ray.dP = c.dP; ray.dD = c.dD; ray.time = c.time;
  float3 P = ray.P, D = ray.D;
  if (!(c.object_flags & SD_OBJECT_TRANSFORM_APPLIED)) {
    const auto itfm = kernel_data_fetch(objects, count - 1u - i).itfm;
    P = transform_point(&itfm, P);
    D = transform_direction(&itfm, D);
  }
  const auto k = 2u + c.segment;
  const unsigned indices[]{max(k - 1u, 2u), k, k + 1u, min(k + 2u, 5u)};
  float4 keys[4];
  for (auto j = 0u; j < 4u; ++j) {
    const auto key = c.keys[indices[j]];
    keys[j] = make_float4(key.x, key.y, key.z, key.w);
  }
  Intersection hit{};
  const auto valid = ribbon_intersect(P, D, 0.0f, 100.0f, 1u << c.subdivision_level, keys, &hit);
  if (!valid) { __builtin_trap(); }
  hit.object = count - 1u - i;
  hit.prim = curve_setup_primitive_base + i;
  hit.type = PRIMITIVE_PACK_SEGMENT(PRIMITIVE_CURVE_RIBBON, c.segment);
  ShaderDataTinyStorage storage{};
  auto *sd = AS_SHADER_DATA(&storage);
  shader_setup_from_ray(nullptr, sd, &ray, &hit);
  output[i * 7u] = make_float4(sd->P, sd->time);
  output[i * 7u + 1u] = make_float4(sd->N, sd->ray_length);
  output[i * 7u + 2u] = make_float4(sd->Ng, sd->dP);
  output[i * 7u + 3u] = make_float4(sd->wi, sd->dI);
  output[i * 7u + 4u] = make_float4(sd->dPdu, sd->u);
  output[i * 7u + 5u] = make_float4(sd->dPdv, sd->v);
  output[i * 7u + 6u] = make_float4(sd->du.dx, sd->du.dy, sd->dv.dx, sd->dv.dy);
  meta[i * 2u] = make_uint4(sd->shader, sd->flag, sd->object_flag, sd->type);
  meta[i * 2u + 1u] = make_uint4(sd->object, sd->prim, valid, 0u);
}
} // namespace

int main() {
  using namespace psycles::test_support;
  std::vector<KernelObject> objects(count);
  std::vector<unsigned> flags(count);
  std::vector<KernelCurve> curves(curve_setup_primitive_base + count);
  std::vector<float4> keys(count * 6u);
  std::vector<KernelShader> shaders(count);
  for (auto i = 0u; i < count; ++i) {
    const auto &c = curve_setup_inputs[i];
    auto &o = objects[count - 1u - i];
    std::memcpy(&o.tfm, &c.tfm, sizeof(c.tfm));
    std::memcpy(&o.itfm, &c.itfm, sizeof(c.itfm));
    o.position_offset = i * 6u;
    o.primitive_type = PRIMITIVE_CURVE_RIBBON;
    flags[count - 1u - i] = c.object_flags;
    curves[curve_setup_primitive_base + i] = {int(i | SHADER_CAST_SHADOW), 2, 4, PRIMITIVE_CURVE_RIBBON};
    shaders[i].flags = c.shader_flags;
    for (auto j = 0u; j < 6u; ++j) {
      const auto k = c.keys[j];
      keys[i * 6u + j] = make_float4(k.x, k.y, k.z, k.w);
    }
  }
  KernelParamsHIP params{};
  params.objects = upload(objects); params.object_flag = upload(flags);
  params.curves = upload(curves); params.curve_keys = upload(keys);
  params.shaders = upload(shaders);
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
  auto *output = upload(std::vector<float4>(count * 7u));
  auto *meta = upload(std::vector<uint4>(count * 2u));
  probe<<<1u, 32u>>>(upload(std::vector<CurveSetupInput>(curve_setup_inputs.begin(), curve_setup_inputs.end())), output, meta);
  check(hipGetLastError());
  std::vector<float4> values(count * 7u);
  std::vector<uint4> integers(count * 2u);
  check(hipMemcpy(values.data(), output, values.size() * sizeof(float4), hipMemcpyDeviceToHost));
  check(hipMemcpy(integers.data(), meta, integers.size() * sizeof(uint4), hipMemcpyDeviceToHost));
  for (auto i = 0u; i < count; ++i) {
    std::printf("C %u", i);
    for (auto j = 0u; j < 7u; ++j) {
      const auto v = values[i * 7u + j];
      std::printf(" %.9g %.9g %.9g %.9g", v.x, v.y, v.z, v.w);
    }
    for (auto j = 0u; j < 2u; ++j) {
      const auto v = integers[i * 2u + j];
      std::printf(" %u %u %u %u", v.x, v.y, v.z, v.w);
    }
    std::printf("\n");
  }
  check(hipDeviceReset());
}
