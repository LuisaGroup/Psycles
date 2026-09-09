// Observe the unmodified original Cycles 5.2.1 GPU holdout functions.
#include "../tests/cycles_holdout_state_fixture.h"

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

namespace f = psycles::test_support::holdout_state;
static_assert(SD_BACKFACING == 1 && SD_TRANSPARENT == (1u << 9));
static_assert(SD_MIS_FRONT == (1u << 16) && SD_HAS_ONLY_VOLUME == (1u << 19));
static_assert(SD_OBJECT_HOLDOUT_MASK == 1 && SVM_STACK_INVALID == 255);
static_assert((SD_CLOSURE_FLAGS | SD_TRANSPARENT | SD_BACKFACING) == 0x1fff);
static_assert(sizeof(SVMNodeClosureHoldout) == sizeof(unsigned));
__device__ f::Input prepared[f::inputs.size()];

void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(error));
    std::exit(1);
  }
}
__device__ float3 rgb(f::RGB value) { return make_float3(value[0], value[1], value[2]); }
__device__ ClosureType type(f::Kind kind) {
  switch (kind) {
    case f::Kind::holdout: return CLOSURE_HOLDOUT_ID;
    case f::Kind::diffuse: return CLOSURE_BSDF_DIFFUSE_ID;
    case f::Kind::transparent: return CLOSURE_BSDF_TRANSPARENT_ID;
    case f::Kind::bssrdf: return CLOSURE_BSSRDF_BURLEY_ID;
    case f::Kind::volume: return CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID;
    case f::Kind::portal: return CLOSURE_BSDF_RAY_PORTAL_ID;
    case f::Kind::none: return CLOSURE_NONE_ID;
    default: return NBUILTIN_CLOSURES;
  }
}
__device__ void snapshot(const ShaderData &sd, Spectrum holdout,
                         float4 *floats, uint4 *integers) {
  uint4 types{};
  for (unsigned c = 0; c < f::capacity; ++c) {
    float4 value{};
    float4 normal{};
    if (c < unsigned(sd.num_closure)) {
      const auto &sc = sd.closure[c];
      // Every slot was initialized by the fixture. closure_alloc must leave
      // these now-defined sample_weight/N sentinels untouched.
      value = make_float4(sc.weight, sc.sample_weight);
      normal = make_float4(sc.N, 0.0f);
      (&types.x)[c] = unsigned(sc.type);
    }
    floats[c] = value;
    floats[c + 4] = normal;
  }
  floats[8] = make_float4(sd.closure_transparent_extinction, 0.0f);
  floats[9] = make_float4(sd.closure_emission_background, 0.0f);
  floats[10] = make_float4(holdout, 0.0f);
  integers[0] = make_uint4(sd.flag, sd.num_closure, sd.num_closure_left, sd.object_flag);
  integers[1] = types;
}
__global__ void probe(float4 *floats, uint4 *integers) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= f::inputs.size()) return;
  const auto in = prepared[i];
  ShaderData sd{};
  sd.flag = in.flags;
  sd.object_flag = in.object_flags;
  sd.num_closure = int(in.count);
  sd.num_closure_left = int(in.left);
  sd.closure_transparent_extinction = rgb(in.extinction);
  sd.closure_emission_background = rgb(in.emission);
  for (unsigned c = 0; c < f::capacity; ++c) {
    sd.closure[c].type = type(in.closures[c].kind);
    sd.closure[c].weight = rgb(in.closures[c].weight);
    sd.closure[c].sample_weight = in.closures[c].sample_weight;
    sd.closure[c].N = rgb(f::normal_sentinel);
  }
  auto out = floats + i * f::float_rows;
  auto meta = integers + i * f::integer_rows;
  snapshot(sd, zero_spectrum(), out, meta);
  if (in.node_enabled) {
    float stack[SVM_STACK_SIZE];
    if (in.mix_offset != SVM_STACK_INVALID) stack[in.mix_offset] = in.mix;
    const SVMNodeClosureHoldout node{SVMStackOffset(in.mix_offset), {0, 0, 0}};
    svm_node_closure_holdout(&sd, stack, rgb(in.node_weight), node);
  }
  snapshot(sd, zero_spectrum(), out + f::snapshot_float_rows,
           meta + f::snapshot_integer_rows);
  const auto holdout = surface_shader_apply_holdout(&sd);
  snapshot(sd, holdout, out + 2 * f::snapshot_float_rows,
           meta + 2 * f::snapshot_integer_rows);
}

int main() {
  check(hipMemcpyToSymbol(HIP_SYMBOL(prepared), f::inputs.data(), sizeof(f::inputs)));
  std::vector<float4> floats(f::inputs.size() * f::float_rows);
  std::vector<uint4> integers(f::inputs.size() * f::integer_rows);
  float4 *output{};
  uint4 *metadata{};
  check(hipMalloc(&output, floats.size() * sizeof(float4)));
  check(hipMalloc(&metadata, integers.size() * sizeof(uint4)));
  probe<<<1, 64>>>(output, metadata);
  check(hipGetLastError());
  check(hipMemcpy(floats.data(), output, floats.size() * sizeof(float4), hipMemcpyDeviceToHost));
  check(hipMemcpy(integers.data(), metadata, integers.size() * sizeof(uint4), hipMemcpyDeviceToHost));
  for (unsigned i = 0; i < f::inputs.size(); ++i) {
    std::printf("%u", i);
    for (unsigned r = 0; r < f::float_rows; ++r) {
      const auto v = floats[i * f::float_rows + r];
      std::printf(" %.9g %.9g %.9g %.9g", v.x, v.y, v.z, v.w);
    }
    for (unsigned r = 0; r < f::integer_rows; ++r) {
      const auto v = integers[i * f::integer_rows + r];
      std::printf(" %u %u %u %u", v.x, v.y, v.z, v.w);
    }
    std::printf("\n");
  }
  check(hipFree(output));
  check(hipFree(metadata));
}
