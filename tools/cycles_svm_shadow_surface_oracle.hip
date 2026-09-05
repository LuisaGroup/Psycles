// Executes the external Cycles 5.2.1 closure implementation on HIP, with its
// own albedo tables and genuinely closure-free ShaderDataTinyStorage.
// hipcc --offload-arch=gfx1201 -DHIPCC -std=c++20 -O3 -ffast-math \
//   -I /path/to/blender/intern/cycles -I include \
//   tools/cycles_svm_shadow_surface_oracle.hip -o /tmp/shadow-surface-oracle
#include "../tests/cycles_svm_shadow_surface_fixture.h"

// Cycles' HIP compatibility macros must precede the dependent GPU headers.
// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/svm/closure.h"
#include "kernel/integrator/surface_shader.h"
#include "scene/shader.tables"
// clang-format on

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iterator>

namespace {

constexpr unsigned case_count =
    psycles::test_support::shadow_surface_case_count;

void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(error));
    std::exit(1);
  }
}

__global__ void probe(float4 *output) {
  const auto scenario = threadIdx.x;
  if (scenario >= case_count) {
    return;
  }
  ShaderDataTinyStorage storage{};
  auto *sd = AS_SHADER_DATA(&storage);
  sd->shader = scenario;
  sd->type = PRIMITIVE_TRIANGLE;
  sd->object = OBJECT_NONE;
  sd->prim = PRIM_NONE;
  sd->lcg_state = 0xabcdu;
  sd->N = sd->Ng = make_float3(0.0f, 0.0f, 1.0f);
  const auto cosine = kernel_data_fetch(attributes_float, scenario);
  sd->wi = make_float3(sqrtf(1.0f - cosine * cosine), 0.0f, cosine);
  sd->num_closure = sd->num_closure_left = 0;
  // Run the original PC loop too. The focused calls below expose the cursor
  // (svm_eval_nodes returns void); they must agree with the full interpreter.
  svm_eval_nodes<KERNEL_FEATURE_NODE_MASK_SURFACE_SHADOW, SHADER_TYPE_SURFACE>(
      nullptr, IntegratorState{0}, sd, nullptr, PATH_RAY_VISIBILITY_SHADOW,
      PATH_RAY_FLAG_NONE);
  const auto full_emission = sd->closure_emission_background;
  const auto full_extinction = sd->closure_transparent_extinction;
  const auto full_flag = sd->flag;
  sd->closure_emission_background = zero_spectrum();
  sd->closure_transparent_extinction = zero_spectrum();
  sd->flag = 0;
  float stack[SVM_STACK_SIZE];
  Spectrum closure_weight = one_spectrum();
  int offset = kernel_data_fetch(svm_nodes, scenario * 4u + 1u);
  // This cursor probe dispatches the four opcodes in the shared input. Each
  // shading operation is Cycles itself, not a host or GPU reference formula.
  for (;;) {
    const auto type = kernel_data_fetch(svm_nodes, offset++);
    if (type == NODE_END) {
      break;
    }
    if (type == NODE_VALUE_F) {
      const auto &node = svm_node_get<SVMNodeValueF>(nullptr, &offset);
      stack_store_float(stack, node.out_offset, node.value);
    } else if (type == NODE_CLOSURE_BSDF) {
      const auto &node = svm_node_get<SVMNodeClosureBsdf>(nullptr, &offset);
      offset = svm_node_closure_bsdf<KERNEL_FEATURE_NODE_MASK_SURFACE_SHADOW,
                                     SHADER_TYPE_SURFACE>(
          nullptr, sd, stack, closure_weight, node, PATH_RAY_VISIBILITY_SHADOW,
          PATH_RAY_FLAG_NONE, offset);
    } else if (type == NODE_CLOSURE_SET_WEIGHT) {
      const auto &node =
          svm_node_get<SVMNodeClosureSetWeight>(nullptr, &offset);
      closure_weight = node.rgb;
    } else {
      __builtin_trap();
    }
  }
  const auto emission_error =
      reduce_max(fabs(full_emission - sd->closure_emission_background));
  const auto extinction_error =
      reduce_max(fabs(full_extinction - sd->closure_transparent_extinction));
  const bool full_matches =
      emission_error <=
          5.0e-5f * fmaxf(reduce_max(fabs(full_emission)), 1.0e-8f) &&
      extinction_error <=
          5.0e-5f * fmaxf(reduce_max(fabs(full_extinction)), 1.0e-8f) &&
      full_flag == sd->flag;
  output[scenario * 3u] =
      make_float4(sd->closure_emission_background, float(sd->flag));
  output[scenario * 3u + 1u] =
      make_float4(sd->closure_transparent_extinction, float(offset));
  output[scenario * 3u + 2u] =
      make_float4(float(sd->num_closure), float(sd->num_closure_left),
                  float(full_matches), float(sd->lcg_state));
}

} // namespace

int main() {
  static_assert(
      sizeof(SVMNodePrincipledBsdfData) ==
      sizeof(psycles::compiler::cycles_svm::SVMNodePrincipledBsdfData));
#define CHECK_PAYLOAD_FIELD(name)                                              \
  static_assert(                                                               \
      offsetof(SVMNodePrincipledBsdfData, name) ==                             \
      offsetof(psycles::compiler::cycles_svm::SVMNodePrincipledBsdfData,       \
               name))
  CHECK_PAYLOAD_FIELD(alpha);
  CHECK_PAYLOAD_FIELD(sheen_weight);
  CHECK_PAYLOAD_FIELD(coat_weight);
  CHECK_PAYLOAD_FIELD(normal_offset);
  CHECK_PAYLOAD_FIELD(coat_normal_offset);
  CHECK_PAYLOAD_FIELD(emission_color);
  CHECK_PAYLOAD_FIELD(emission_strength);
  CHECK_PAYLOAD_FIELD(sheen_tint);
  CHECK_PAYLOAD_FIELD(sheen_roughness);
  CHECK_PAYLOAD_FIELD(coat_tint);
  CHECK_PAYLOAD_FIELD(coat_roughness);
  CHECK_PAYLOAD_FIELD(coat_ior);
#undef CHECK_PAYLOAD_FIELD
  static_assert(unsigned(NODE_CLOSURE_BSDF) ==
                unsigned(psycles::compiler::cycles_svm::NODE_CLOSURE_BSDF));
  static_assert(
      unsigned(CLOSURE_BSDF_PRINCIPLED_ID) ==
      unsigned(psycles::compiler::cycles_svm::CLOSURE_BSDF_PRINCIPLED_ID));
  const auto words = psycles::test_support::make_shadow_surface_image();
  KernelParamsHIP params{};
  unsigned *device_words{};
  check(hipMalloc(&device_words, words.size() * sizeof(unsigned)));
  check(hipMemcpy(device_words, words.data(), words.size() * sizeof(unsigned),
                  hipMemcpyHostToDevice));
  params.svm_nodes = device_words;
  params.data.integrator.caustics_reflective = true;
  params.data.integrator.caustics_refractive = true;
  std::vector<float> tables;
  const auto append_table = [&tables](const auto &table) {
    const auto offset = static_cast<int>(tables.size());
    tables.insert(tables.end(), std::begin(table), std::end(table));
    return offset;
  };
  params.data.tables.ggx_E = append_table(table_ggx_E);
  params.data.tables.ggx_Eavg = append_table(table_ggx_Eavg);
  params.data.tables.ggx_glass_E = append_table(table_ggx_glass_E);
  params.data.tables.ggx_glass_Eavg = append_table(table_ggx_glass_Eavg);
  params.data.tables.ggx_glass_inv_E = append_table(table_ggx_glass_inv_E);
  params.data.tables.ggx_glass_inv_Eavg =
      append_table(table_ggx_glass_inv_Eavg);
  params.data.tables.sheen_ltc = append_table(table_sheen_ltc);
  params.data.tables.ggx_gen_schlick_ior_s =
      append_table(table_ggx_gen_schlick_ior_s);
  params.data.tables.ggx_gen_schlick_s = append_table(table_ggx_gen_schlick_s);
  float *device_tables{};
  check(hipMalloc(&device_tables, tables.size() * sizeof(float)));
  check(hipMemcpy(device_tables, tables.data(), tables.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  params.lookup_table = device_tables;
  std::array<float, case_count> cosines{};
  for (auto i = 0u; i < case_count; ++i) {
    cosines[i] = psycles::test_support::shadow_surface_cosine(i);
  }
  float *device_cosines{};
  check(hipMalloc(&device_cosines, sizeof(cosines)));
  check(hipMemcpy(device_cosines, cosines.data(), sizeof(cosines),
                  hipMemcpyHostToDevice));
  params.attributes_float = device_cosines;
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
  float4 *output{};
  check(hipMalloc(&output, case_count * 3u * sizeof(float4)));
  probe<<<1u, 64u>>>(output);
  check(hipGetLastError());
  std::array<float4, case_count * 3u> host{};
  check(hipMemcpy(host.data(), output, sizeof(host), hipMemcpyDeviceToHost));
  for (auto i = 0u; i < case_count; ++i) {
    const auto emission = host[i * 3u];
    const auto extinction = host[i * 3u + 1u];
    const auto meta = host[i * 3u + 2u];
    if (meta.z != 1.0f) {
      std::fprintf(stderr,
                   "Full Cycles interpreter disagrees with focused cursor "
                   "probe in case %u\n",
                   i);
      return 1;
    }
    std::printf("%u %.9g %.9g %.9g %u %.9g %.9g %.9g %u %u %u %u\n", i,
                emission.x, emission.y, emission.z, unsigned(emission.w),
                extinction.x, extinction.y, extinction.z,
                unsigned(extinction.w), unsigned(meta.x), unsigned(meta.y),
                unsigned(meta.w));
  }
  check(hipFree(output));
  check(hipFree(device_cosines));
  check(hipFree(device_tables));
  check(hipFree(device_words));
}
