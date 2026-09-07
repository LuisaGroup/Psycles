// Original Cycles volume stack, shader setup, SVM and phase-copy execution.
#include "../tests/cycles_svm_volume_stack_fixture.h"

// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/geom/primitive.h"
#include "kernel/geom/shader_data.h"
#include "kernel/integrator/state_util.h"
#include "kernel/integrator/volume_shader.h"
// clang-format on

#include <cstdio>
#include <cstdlib>

namespace f = psycles::test_support::volume_stack_fixture;
void check(hipError_t e) {
  if (e != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(e)); std::exit(1); }
}
template<typename T> auto upload(const std::vector<T> &data) {
  T *p{}; check(hipMalloc(&p, data.size() * sizeof(T)));
  check(hipMemcpy(p, data.data(), data.size() * sizeof(T), hipMemcpyHostToDevice));
  return p;
}
__device__ void put3(float *out, float3 v) { out[0] = v.x; out[1] = v.y; out[2] = v.z; }
__device__ void closure(float *out, const ShaderVolumeClosure *c) {
  out[0] = float(c->type); put3(out + 1, c->weight); out[4] = c->sample_weight;
  if (c->type == CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID) {
    out[5] = reinterpret_cast<const HenyeyGreensteinVolume *>(c)->g;
  } else if (c->type == CLOSURE_VOLUME_DRAINE_ID) {
    const auto *v = reinterpret_cast<const DraineVolume *>(c);
    out[5] = v->g; out[6] = v->alpha;
  } else if (c->type == CLOSURE_VOLUME_FOURNIER_FORAND_ID) {
    const auto *v = reinterpret_cast<const FournierForandVolume *>(c);
    put3(out + 5, make_float3(v->c1, v->c2, v->c3));
  }
}
__global__ void probe(f::Output *outputs) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= f::count) { return; }
  const auto mode = i % f::state_count;
  Ray ray{};
  ray.P = make_float3(1.0f, -2.0f, 3.0f);
  ray.D = make_float3(0.0f, 0.0f, 1.0f);
  ray.tmin = 0.25f; ray.time = 0.375f;
  ShaderData sd{};
  shader_setup_from_volume(&sd, &ray, OBJECT_NONE);
  auto &out = outputs[i];
  put3(out.setup, sd.P); put3(out.setup + 3, sd.ray_P);
  put3(out.setup + 6, sd.N); put3(out.setup + 9, sd.Ng);
  put3(out.setup + 12, sd.wi); put3(out.setup + 15, sd.dPdu);
  put3(out.setup + 18, sd.dPdv);
  out.setup[21] = sd.time; out.setup[22] = sd.ray_length;
  out.setup[23] = sd.dP; out.setup[24] = sd.dI;
  out.setup[25] = sd.u; out.setup[26] = sd.v;
  out.setup[27] = sd.du.dx; out.setup[28] = sd.du.dy;
  out.setup[29] = sd.dv.dx; out.setup[30] = sd.dv.dy;
  sd.flag = SD_CACHE_MISS | SD_EXTINCTION | SD_EMISSION;
  sd.closure_transparent_extinction = make_float3(-9.0f);
  sd.closure_emission_background = make_float3(-7.0f);
  const auto visibility = PathRayVisibility(f::visibility(mode));
  const auto flag = f::flag(mode);
  if (mode == 2u) { volume_shader_eval<true>(nullptr, int(i), &sd, visibility, flag); }
  else { volume_shader_eval<false>(nullptr, int(i), &sd, visibility, flag); }
  ShaderVolumePhases phases{};
  volume_shader_copy_phases(&phases, &sd);
  out.meta[0] = sd.flag; out.meta[1] = sd.object_flag;
  out.meta[2] = sd.type; out.meta[3] = sd.object; out.meta[4] = sd.shader;
  out.meta[5] = sd.num_closure; out.meta[6] = sd.num_closure_left;
  out.meta[7] = phases.num_closure;
  if (sd.flag & SD_EXTINCTION) { put3(out.coefficients, sd.closure_transparent_extinction); }
  if (sd.flag & SD_EMISSION) { put3(out.coefficients + 6, sd.closure_emission_background); }
  float3 scatter = zero_float3();
  for (int j = 0; j < sd.num_closure; ++j) {
    if (CLOSURE_IS_VOLUME(sd.closure[j].type)) { scatter += sd.closure[j].weight; }
    if (j < 16) { closure(out.closures[j], reinterpret_cast<const ShaderVolumeClosure *>(&sd.closure[j])); }
  }
  put3(out.coefficients + 3, scatter);
  for (int j = 0; j < phases.num_closure; ++j) { closure(out.phases[j], &phases.closure[j]); }
}

int main() {
  KernelParamsHIP params{};
  params.svm_nodes = upload(f::words());
  std::vector<KernelShader> shaders(f::shader_count);
  for (unsigned s = 0; s < f::shader_count; ++s) { shaders[s].flags = f::shader_flags(s); }
  params.shaders = upload(shaders);
  std::vector<KernelObject> objects(4u);
  std::vector<unsigned> flags(4u);
  for (unsigned o = 0u; o < 4u; ++o) {
    auto &v = objects[o];
    v.volume_density = f::densities[o];
    v.visibility = o == 2u ? PATH_RAY_VISIBILITY_CAMERA : PATH_RAY_VISIBILITY_ALL;
    v.tfm = {make_float4(2, 0, 0, 0.5f), make_float4(0, -4, 0, -0.25f),
             make_float4(0, 0, 0.5f, 1)};
    v.itfm = {make_float4(0.5f, 0, 0, -0.25f), make_float4(0, -0.25f, 0, -0.0625f),
              make_float4(0, 0, 2, -2)};
    flags[o] = o % 2u ? SD_OBJECT_NEGATIVE_SCALE : 0u;
  }
  params.objects = upload(objects); params.object_flag = upload(flags);
  params.data.kernel_features = KERNEL_FEATURE_VOLUME;
  params.integrator_state.path.bounce = upload(std::vector<uint16_t>(f::count, 3u));
  params.integrator_state.shadow_path.bounce = params.integrator_state.path.bounce;
  for (unsigned e = 0u; e < 4u; ++e) {
    std::vector<int> object(f::count, OBJECT_NONE), shader(f::count, SHADER_NONE);
    for (unsigned i = 0; i < f::count; ++i) {
      const auto row = i / f::state_count;
      if (e < 3u && f::stacks[row][e] >= 0) {
        object[i] = int(f::object(row, e)); shader[i] = f::stacks[row][e];
      }
    }
    params.integrator_state.volume_stack[e].object = upload(object);
    params.integrator_state.volume_stack[e].shader = upload(shader);
    params.integrator_state.shadow_volume_stack[e].object = params.integrator_state.volume_stack[e].object;
    params.integrator_state.shadow_volume_stack[e].shader = params.integrator_state.volume_stack[e].shader;
  }
  for (unsigned capacity : f::capacities) {
    params.data.max_closures = capacity;
    check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
    auto *output = upload(std::vector<f::Output>(f::count));
    probe<<<(f::count + 31u) / 32u, 32u>>>(output);
    check(hipGetLastError());
    std::vector<f::Output> values(f::count);
    check(hipMemcpy(values.data(), output, values.size() * sizeof(f::Output), hipMemcpyDeviceToHost));
    for (unsigned i = 0u; i < f::count; ++i) {
      const auto &v = values[i];
      std::printf("R %u %u", capacity, i);
      for (auto x : v.meta) { std::printf(" %u", x); }
      for (auto x : v.setup) { std::printf(" %.9g", x); }
      for (auto x : v.coefficients) { std::printf(" %.9g", x); }
      for (const auto &c : v.closures) { for (auto x : c) { std::printf(" %.9g", x); } }
      for (const auto &c : v.phases) { for (auto x : c) { std::printf(" %.9g", x); } }
      std::printf("\n");
    }
    check(hipFree(output));
  }
  check(hipDeviceReset());
}
