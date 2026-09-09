// Observe unmodified Cycles 5.2.1 bsdf_sample/bsdf_eval on the original GPU
// object-frequency boundary. No host BSDF implementation or expected values.
// Capture identity and raw output: tests/data/cycles_shading_terminator.{json,txt}.
// hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC -std=c++20 \
//   -O3 -ffast-math -I /path/to/blender/intern/cycles \
//   tools/cycles_shading_terminator_oracle.hip -o /tmp/shading-terminator-oracle
#include "../tests/cycles_shading_terminator_fixture.h"

// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/bvh/bvh.h"
#include "kernel/integrator/state_util.h"
#include "kernel/svm/svm.h"
#include "kernel/integrator/surface_shader.h"
// clang-format on

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
namespace f = psycles::test_support::shading_terminator;
static_assert(SD_USE_BUMP_MAP_CORRECTION == f::bump_map_correction);
static_assert(PRIMITIVE_TRIANGLE == 1u);
static_assert(LABEL_NONE == 0 && LABEL_TRANSMIT == 1 && LABEL_REFLECT == 2);
constexpr auto count = unsigned(f::inputs.size());

void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(error));
    std::exit(1);
  }
}

__device__ float3 vector(f::Vector value) {
  return make_float3(value[0], value[1], value[2]);
}

__device__ uint4 state(const ShaderData &sd) {
  return make_uint4(sd.flag, sd.num_closure, sd.num_closure_left,
                    unsigned(sd.closure[0].type));
}

__global__ void probe(const f::Input *inputs, float4 *values, uint4 *metadata) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const auto input = inputs[i];
  ShaderData sd{};
  sd.N = normalize(vector(input.shader_normal));
  sd.Ng = normalize(vector(input.geometric_normal));
  sd.wi = normalize(vector(input.incoming));
  sd.type = PRIMITIVE_TRIANGLE;
  sd.object = int(i); // Every input has a real, initialized object-table row.
  sd.prim = 0;
  sd.flag = input.flags;
  sd.num_closure_left = 1;
  const float3 closure_normal = normalize(vector(input.closure_normal));
  if (input.kind == f::Kind::translucent) {
    bsdf_translucent_setup(&sd, closure_normal, one_spectrum());
  } else {
    bsdf_diffuse_setup(&sd, closure_normal, one_spectrum());
    if (input.kind == f::Kind::none) {
      // A defined, allocated closure slot whose type is no longer a BSDF.
      // This directly exercises the original dispatcher's LABEL_NONE branch.
      sd.closure[0].type = CLOSURE_NONE_ID;
    }
  }
  const auto *closure = &sd.closure[0];
  auto out = values + i * f::float_rows;
  auto meta = metadata + i * f::integer_rows;
  meta[0] = state(sd);
  out[f::shader_normal_frequency] = make_float4(sd.N, input.frequency);
  out[f::geometric_normal] = make_float4(sd.Ng, 0.0f);
  out[f::closure_normal_sample_weight] = make_float4(closure->N, closure->sample_weight);

  Spectrum sampled = zero_spectrum();
  float3 sampled_direction = zero_float3();
  float sampled_pdf = 0.0f;
  float2 roughness = zero_float2();
  float eta = 0.0f;
  const int label = bsdf_sample(nullptr, &sd, closure, vector(input.random),
                                &sampled, &sampled_direction, &sampled_pdf,
                                &roughness, &eta);
  const float3 evaluation_direction = normalize(vector(input.evaluation_direction));
  float evaluation_pdf = 0.0f;
  const Spectrum evaluated = bsdf_eval(nullptr, &sd, closure,
                                        evaluation_direction, &evaluation_pdf);
  out[f::sampled_value_pdf] = make_float4(sampled, sampled_pdf);
  out[f::sampled_direction_eta] = make_float4(sampled_direction, eta);
  out[f::sampled_roughness] = make_float4(roughness.x, roughness.y, 0.0f, 0.0f);
  out[f::evaluated_value_pdf] = make_float4(evaluated, evaluation_pdf);
  out[f::evaluation_direction] = make_float4(evaluation_direction, 0.0f);
  out[f::bump_terms] = make_float4(
      bump_shadowing_term(&sd, closure, sampled_direction, false),
      bump_shadowing_term(&sd, closure, evaluation_direction, true), 0.0f, 0.0f);
  meta[1] = state(sd);
  meta[2] = make_uint4(unsigned(label), unsigned(sd.object), unsigned(sd.type),
                       unsigned(input.kind));
}
} // namespace

int main() {
  std::vector<KernelObject> object_rows(count);
  for (unsigned i = 0; i < count; ++i) {
    object_rows[i].shadow_terminator_shading_offset = f::inputs[i].frequency;
  }
  KernelObject *objects{};
  f::Input *inputs{};
  float4 *values{};
  uint4 *metadata{};
  check(hipMalloc(&objects, object_rows.size() * sizeof(KernelObject)));
  check(hipMalloc(&inputs, sizeof(f::inputs)));
  check(hipMalloc(&values, count * f::float_rows * sizeof(float4)));
  check(hipMalloc(&metadata, count * f::integer_rows * sizeof(uint4)));
  check(hipMemcpy(objects, object_rows.data(), object_rows.size() * sizeof(KernelObject),
                  hipMemcpyHostToDevice));
  check(hipMemcpy(inputs, f::inputs.data(), sizeof(f::inputs), hipMemcpyHostToDevice));
  KernelParamsHIP params{};
  params.objects = objects;
  params.data.max_closures = 1;
  params.data.background.transparent_roughness_squared_threshold = -1.0f;
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
  probe<<<1, 64>>>(inputs, values, metadata);
  check(hipGetLastError());
  std::vector<float4> floats(count * f::float_rows);
  std::vector<uint4> integers(count * f::integer_rows);
  check(hipMemcpy(floats.data(), values, floats.size() * sizeof(float4), hipMemcpyDeviceToHost));
  check(hipMemcpy(integers.data(), metadata, integers.size() * sizeof(uint4), hipMemcpyDeviceToHost));
  std::printf("cycles-shading-terminator %u %u %u %u\n", f::format_version,
               count, unsigned(f::float_rows), f::integer_rows);
  for (unsigned i = 0; i < count; ++i) {
    std::printf("%u %s", i, f::names[i]);
    for (unsigned row = 0; row < f::float_rows; ++row) {
      const auto value = floats[i * f::float_rows + row];
      std::printf(" %.9g %.9g %.9g %.9g", value.x, value.y, value.z, value.w);
    }
    for (unsigned row = 0; row < f::integer_rows; ++row) {
      const auto value = integers[i * f::integer_rows + row];
      std::printf(" %u %u %u %u", value.x, value.y, value.z, value.w);
    }
    std::puts("");
  }
  check(hipFree(objects));
  check(hipFree(inputs));
  check(hipFree(values));
  check(hipFree(metadata));
}
