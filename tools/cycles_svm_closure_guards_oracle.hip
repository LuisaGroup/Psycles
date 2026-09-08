// Original Cycles 5.2.1 HIP interpreter, with no copied closure implementation.
// hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC -std=c++20 -O3 \
//   -ffast-math -I /path/to/blender/intern/cycles this_file -o /tmp/oracle
// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/integrator/state_util.h"
#include "kernel/svm/svm.h"
#include "scene/shader.tables"
// clang-format on
#include "../tests/cycles_svm_closure_guards_test_data.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace fixture = psycles::test_support::closure_guards;
namespace {
struct Output { unsigned flag, count, left, type; float values[fixture::value_count]; };
void check(hipError_t error) {
  if (error != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(error)); std::exit(1); }
}
__global__ void probe(Output *output) {
  const auto i = threadIdx.x;
  if (i >= fixture::cases_per_flags) { return; }
  constexpr unsigned capacities[]{0u, 1u, 4u};
  ShaderData sd{};
  sd.shader = i / 6u;
  sd.object = OBJECT_NONE;
  sd.prim = PRIM_NONE;
  sd.type = PRIMITIVE_TRIANGLE;
  sd.N = sd.Ng = sd.wi = normalize(make_float3(0.2f, -0.3f, 1.0f));
  sd.num_closure_left = capacities[i % 3u];
  const auto visibility = (i / 3u) % 2u ? PATH_RAY_VISIBILITY_DIFFUSE : PATH_RAY_VISIBILITY_CAMERA;
  svm_eval_nodes<KERNEL_FEATURE_NODE_MASK_SURFACE, SHADER_TYPE_SURFACE>(
      nullptr, IntegratorState{0}, &sd, nullptr, visibility, PATH_RAY_FLAG_NONE);
  auto &out = output[i];
  out = {};
  out.flag = sd.flag;
  out.count = sd.num_closure;
  out.left = sd.num_closure_left;
  if (sd.num_closure != 0) {
    const auto &sc = sd.closure[0];
    const auto *bsdf = reinterpret_cast<const MicrofacetBsdf *>(&sc);
    out.type = sc.type;
    const auto put = [&](unsigned offset, float3 value) {
      out.values[offset] = value.x; out.values[offset + 1u] = value.y; out.values[offset + 2u] = value.z;
    };
    put(0u, sc.N); out.values[3] = bsdf->alpha_x;
    put(4u, sc.weight); out.values[7] = sc.sample_weight;
    put(8u, bsdf->T); out.values[11] = bsdf->alpha_y;
    out.values[12] = bsdf->ior;
    // Only GGX setup initializes this field and GGX sample/eval consumes it.
    // Beckmann storage is not an observable energy-compensation value.
    if (sc.type == CLOSURE_BSDF_MICROFACET_GGX_ID ||
        sc.type == CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID ||
        sc.type == CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID) {
      out.values[13] = bsdf->energy_scale;
    }
    out.values[14] = static_cast<unsigned>(bsdf->fresnel_type);
  }
}
} // namespace

int main() {
  std::vector<unsigned> words(16u);
  for (unsigned shader = 0u; shader < fixture::programs.size(); shader++) {
    const auto local = fixture::programs[shader];
    const auto start = static_cast<unsigned>(words.size());
    words[shader * 4u] = local[0];
    for (auto j = 1u; j < 4u; j++) { words[shader * 4u + j] = local[j] - 4u + start; }
    words.insert(words.end(), local.begin() + 4u, local.end());
  }
  unsigned *device_words{};
  Output *device_output{};
  check(hipMalloc(&device_words, words.size() * sizeof(unsigned)));
  check(hipMalloc(&device_output, fixture::cases_per_flags * sizeof(Output)));
  check(hipMemcpy(device_words, words.data(), words.size() * sizeof(unsigned), hipMemcpyHostToDevice));
  KernelParamsHIP params{};
  params.svm_nodes = device_words;
  params.data.film.is_rec709 = true;
  std::vector<float> tables;
  const auto append = [&](const auto &table) {
    const auto offset = static_cast<int>(tables.size());
    tables.insert(tables.end(), std::begin(table), std::end(table));
    return offset;
  };
  params.data.tables.ggx_E = append(table_ggx_E);
  params.data.tables.ggx_Eavg = append(table_ggx_Eavg);
  params.data.tables.ggx_glass_E = append(table_ggx_glass_E);
  params.data.tables.ggx_glass_Eavg = append(table_ggx_glass_Eavg);
  params.data.tables.ggx_glass_inv_E = append(table_ggx_glass_inv_E);
  params.data.tables.ggx_glass_inv_Eavg = append(table_ggx_glass_inv_Eavg);
  params.data.tables.sheen_ltc = append(table_sheen_ltc);
  params.data.tables.ggx_gen_schlick_ior_s = append(table_ggx_gen_schlick_ior_s);
  params.data.tables.ggx_gen_schlick_s = append(table_ggx_gen_schlick_s);
  float *device_tables{};
  check(hipMalloc(&device_tables, tables.size() * sizeof(float)));
  check(hipMemcpy(device_tables, tables.data(), tables.size() * sizeof(float), hipMemcpyHostToDevice));
  params.lookup_table = device_tables;
  for (unsigned flags = 0u; flags < 4u; flags++) {
    params.data.integrator.caustics_reflective = flags & 1u;
    params.data.integrator.caustics_refractive = flags & 2u;
    check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
    probe<<<1u, 32u>>>(device_output);
    check(hipGetLastError());
    std::array<Output, fixture::cases_per_flags> results;
    check(hipMemcpy(results.data(), device_output, sizeof(results), hipMemcpyDeviceToHost));
    for (unsigned i = 0u; i < results.size(); i++) {
      const auto &row = results[i];
      std::printf("%u %u %u %u %u", flags * fixture::cases_per_flags + i,
                  row.flag, row.count, row.left, row.type);
      for (float value : row.values) { std::printf(" %.9g", value); }
      std::printf("\n");
    }
  }
  check(hipFree(device_output));
  check(hipFree(device_tables));
  check(hipFree(device_words));
}
