// Original Cycles 5.2.1 GPU shader setup, SVM and importance-map bake.
// hipcc --offload-arch=gfx1201 -DHIPCC -std=c++20 -O3 -ffast-math \
//   -I /path/to/blender/intern/cycles -I include \
//   tools/cycles_svm_background_oracle.hip -o /tmp/background-oracle
#include "../tests/cycles_svm_background_fixture.h"

// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/geom/primitive.h"
#include "kernel/geom/shader_data.h"
#include "kernel/integrator/surface_shader.h"
#include "kernel/light/background.h"
#include "kernel/bake/bake.h"
// clang-format on

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
using namespace psycles::test_support;
constexpr unsigned count = background_inputs.size();
void check(hipError_t e) {
  if (e != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(e)); std::exit(1); }
}
template <typename T> T *upload(const std::vector<T> &data) {
  T *p{};
  check(hipMalloc(&p, data.size() * sizeof(T)));
  check(hipMemcpy(p, data.data(), data.size() * sizeof(T), hipMemcpyHostToDevice));
  return p;
}

__global__ void probe(const BackgroundInput *inputs, unsigned mode,
                      float4 *out, uint4 *meta) {
  const unsigned i = threadIdx.x;
  if (i >= count) { return; }
  const auto &c = inputs[i];
  auto P = make_float3(c.origin[0], c.origin[1], c.origin[2]);
  auto D = make_float3(c.direction[0], c.direction[1], c.direction[2]);
  auto dD = c.differential;
  auto time = c.time;
  auto visibility = PathRayVisibility(mode == 0u ? c.visibility : 0u);
  auto flag = (mode == 0u ? c.flag : 0u) | PATH_RAY_EMISSION;
  if (mode == 2u) {
    P = zero_float3();
    D = equirectangular_to_direction(c.u, c.v);
    time = 0.5f;
    dD = 0.5f * (len(equirectangular_to_direction(c.u + 1.0f / background_width, c.v) - D) +
                 len(equirectangular_to_direction(c.u, c.v + 1.0f / background_height) - D));
    flag |= PATH_RAY_IMPORTANCE_BAKE;
  } else if (!(visibility & PATH_RAY_VISIBILITY_CAMERA)) {
    dD = background_light_clamp_dD(nullptr, dD);
  }
  ShaderDataTinyStorage storage{};
  auto *sd = AS_SHADER_DATA(&storage);
  shader_setup_from_background(nullptr, sd, P, D, dD, time);
  out[i * 9u] = make_float4(sd->P, sd->time);
  out[i * 9u + 1u] = make_float4(sd->ray_P, sd->ray_length);
  out[i * 9u + 2u] = make_float4(sd->N, sd->dP);
  out[i * 9u + 3u] = make_float4(sd->Ng, sd->dI);
  out[i * 9u + 4u] = make_float4(sd->wi, sd->u);
  out[i * 9u + 5u] = make_float4(sd->dPdu, sd->v);
  out[i * 9u + 6u] = make_float4(sd->dPdv, 0.0f);
  out[i * 9u + 7u] = make_float4(sd->du.dx, sd->du.dy, sd->dv.dx, sd->dv.dy);
  meta[i * 2u] = make_uint4(sd->shader, sd->flag, sd->object_flag, sd->type);
  meta[i * 2u + 1u] = make_uint4(sd->object, sd->prim, visibility, flag);
  if (mode == 2u) {
    surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE_LIGHT &
        ~(KERNEL_FEATURE_NODE_RAYTRACE | KERNEL_FEATURE_NODE_LIGHT_PATH)>(
        nullptr, ConstIntegratorBakeState{}, sd, nullptr, visibility, flag);
  } else if (mode == 0u) {
    surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE_BACKGROUND>(
        nullptr, int(i), sd, nullptr, visibility, flag);
  } else {
    surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE_LIGHT>(
        nullptr, int(i), sd, nullptr, visibility, flag);
  }
  out[i * 9u + 8u] = make_float4(surface_shader_background(sd), float(sd->flag));
}

__global__ void bake(const KernelShaderEvalInput *input, float *out, unsigned *miss) {
  const auto i = threadIdx.x;
  if (i < count) { kernel_background_evaluate(nullptr, input, out, miss, i); }
}
__global__ void window_bake(const KernelShaderEvalInput *input, float *out, unsigned *miss) {
  const auto i = threadIdx.x;
  if (i < 8u) { kernel_background_evaluate(nullptr, input, out, miss, i); }
}
} // namespace

int main() {
  using namespace psycles::test_support;
  KernelParamsHIP params{};
  params.svm_nodes = upload(background_words());
  std::vector<KernelShader> shaders(1u);
  shaders[0].flags = background_shader_flags;
  params.shaders = upload(shaders);
  params.data.background.surface_shader = background_shader;
  params.data.background.map_dD = M_PI_F / background_height;
  params.integrator_state.path.bounce = upload(std::vector<uint16_t>(count, 3));
  params.integrator_state.path.rng_pixel = upload(std::vector<unsigned>(count, 19u));
  params.integrator_state.path.rng_offset = upload(std::vector<uint16_t>(count, 24u));
  params.integrator_state.path.sample = upload(std::vector<unsigned>(count, 7u));
  const auto input = upload(std::vector<BackgroundInput>(background_inputs.begin(), background_inputs.end()));
  auto *output = upload(std::vector<float4>(count * 9u));
  auto *metadata = upload(std::vector<uint4>(count * 2u));
  std::vector<float4> values(count * 9u);
  std::vector<uint4> meta(count * 2u);
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
  for (unsigned mode = 0u; mode < 3u; ++mode) {
    probe<<<1, 32>>>(input, mode, output, metadata);
    check(hipGetLastError());
    check(hipMemcpy(values.data(), output, values.size() * sizeof(float4), hipMemcpyDeviceToHost));
    check(hipMemcpy(meta.data(), metadata, meta.size() * sizeof(uint4), hipMemcpyDeviceToHost));
    for (unsigned i = 0; i < count; ++i) {
      std::printf("B %u %u", mode, i);
      for (unsigned j = 0; j < 9; ++j) {
        const auto v = values[i * 9 + j];
        std::printf(" %.9g %.9g %.9g %.9g", v.x, v.y, v.z, v.w);
      }
      for (unsigned j = 0; j < 2; ++j) {
        const auto v = meta[i * 2 + j];
        std::printf(" %u %u %u %u", v.x, v.y, v.z, v.w);
      }
      std::printf("\n");
    }
  }
  std::vector<KernelShaderEvalInput> bake_inputs(count);
  for (unsigned i = 0; i < count; ++i) {
    bake_inputs[i].u = background_inputs[i].u;
    bake_inputs[i].v = background_inputs[i].v;
    bake_inputs[i].object = background_width;
    bake_inputs[i].prim = background_height;
  }
  auto *baked = upload(std::vector<float>(3 * count));
  bake<<<1, 32>>>(upload(bake_inputs), baked, upload(std::vector<unsigned>(1u)));
  check(hipGetLastError());
  std::vector<float> colors(3 * count);
  check(hipMemcpy(colors.data(), baked, colors.size() * sizeof(float), hipMemcpyDeviceToHost));
  for (unsigned i = 0; i < count; ++i) {
    std::printf("I %u %.9g %.9g %.9g\n", i, colors[3 * i], colors[3 * i + 1], colors[3 * i + 2]);
  }
  params.svm_nodes = upload(background_words(true));
  params.data.cam.type = CAMERA_ORTHOGRAPHIC;
  std::vector<KernelShaderEvalInput> window_inputs(8u);
  for (unsigned i = 0; i < 8u; ++i) {
    window_inputs[i].u = (float(i % 4u) + 0.5f) / 4.0f;
    window_inputs[i].v = (float(i / 4u) + 0.5f) / 2.0f;
    window_inputs[i].object = 4; window_inputs[i].prim = 2;
  }
  auto *window_input = upload(window_inputs);
  auto *window_output = upload(std::vector<float>(24u));
  std::vector<float> window_values(24u);
  for (unsigned variant = 0; variant < 2u; ++variant) {
    auto &tfm = params.data.cam.worldtondc;
    tfm = ProjectionTransform{};
    tfm.x.x = variant == 0u ? 0.25f : 0.5f; tfm.x.w = 0.5f;
    tfm.y.y = variant == 0u ? 0.5f : 0.25f; tfm.y.w = 0.5f;
    tfm.z.z = tfm.w.w = 1.0f;
    check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
    window_bake<<<1, 32>>>(window_input, window_output, upload(std::vector<unsigned>(1u)));
    check(hipGetLastError());
    check(hipMemcpy(window_values.data(), window_output, 24u * sizeof(float), hipMemcpyDeviceToHost));
    for (unsigned i = 0; i < 8u; ++i) {
      std::printf("W %u %u %.9g %.9g %.9g\n", variant, i,
                  window_values[3 * i], window_values[3 * i + 1], window_values[3 * i + 2]);
    }
  }
  check(hipDeviceReset());
}
