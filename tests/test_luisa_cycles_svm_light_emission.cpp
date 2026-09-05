#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_light_emission_fixture.h"
#include "luisa_cycles_svm_test_kernel_globals.h"
#include "path_tracer_bsdf_tables.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace svm = psycles::luisa_backend::cycles_svm;
using psycles::test_support::light_emission_cases;
constexpr auto case_count = static_cast<unsigned>(light_emission_cases.size());

class TableKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
  const BufferFloat &_table;

public:
  explicit TableKernelGlobals(const BufferFloat &table) : _table{table} {}
  [[nodiscard]] Float
  cycles_bsdf_data(Expr<std::uint32_t> i) const noexcept override {
    return _table.read(i);
  }
};

// Also exercise the production entry when the pending API migration exposes
// it. The stable diagnostic entry remains independently testable at the
// committed base; neither path implements reference shading behavior.
template <typename Globals, typename Words>
void evaluate(const Globals &kg, const Words &words,
              const std::array<bool, NODE_NUM> &used,
              const svm::TransformState &transforms, svm::ShaderData &sd,
              const svm::PathState &state, svm::EvaluationResult &result,
              bool diagnose) {
  constexpr auto mask = svm::kernel_feature_node_mask_surface_light;
  if (!diagnose) {
    if constexpr (requires {
                    eval_nodes_assume_valid(kg, words, SHADER_TYPE_SURFACE, 0u,
                                            mask, used, transforms, sd, state,
                                            1u);
                  }) {
      eval_nodes_assume_valid(kg, words, SHADER_TYPE_SURFACE, 0u, mask, used,
                              transforms, sd, state, 1u);
      return;
    }
  }
  eval_nodes(kg, words, SHADER_TYPE_SURFACE, 0u, mask, used, transforms, sd,
             state, result);
}

auto make_kernel(bool has_storage, bool diagnose) {
  return Kernel1D<Buffer<unsigned>, Buffer<float>, Buffer<float>,
                  Buffer<luisa::float4>, Buffer<luisa::uint4>>{
      [=](BufferUInt words, BufferFloat tables, BufferFloat cosines,
          BufferFloat4 output, Var<Buffer<luisa::uint4>> meta) noexcept {
        const auto i = dispatch_id().x;
        std::optional<svm::ClosurePool> pool;
        if (has_storage) {
          pool.emplace(4u);
          pool->set_left(0u);
        }
        const auto identity = make_float4x4(1.0f);
        const auto cosine = cosines.read(i);
        svm::ShaderData sd{
            make_float3(0.0f),
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(sqrt(1.0f - cosine * cosine), 0.0f, cosine),
            svm::primitive_triangle,
            i,
            0u,
            0u,
            0u,
            0.25f,
            0.25f,
            0u,
            0.5f,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            make_float3(1.0f, 0.0f, 0.0f),
            make_float3(0.0f, 1.0f, 0.0f),
            identity,
            identity,
            0xabcdu,
            pool ? &*pool : nullptr};
        const TableKernelGlobals kg{tables};
        const svm::PathState state{0u, svm::path_ray_emission};
        const svm::TransformState transforms{identity, identity, identity,
                                             identity};
        std::array<bool, NODE_NUM> used{};
        used[NODE_SHADER_JUMP] = true;
        used[NODE_VALUE_F] = true;
        used[NODE_CLOSURE_BSDF] = true;
        used[NODE_CLOSURE_SET_WEIGHT] = true;
        used[NODE_END] = true;
        svm::EvaluationResult result;
        evaluate(kg, words, used, transforms, sd, state, result, diagnose);
        output.write(i * 3u, make_float4(sd.closure_emission_background, 0.0f));
        output.write(i * 3u + 1u,
                     make_float4(sd.closure_transparent_extinction, 0.0f));
        output.write(i * 3u + 2u, make_float4(result.closure_weight, 0.0f));
        meta.write(i * 2u, make_uint4(sd.flag, result.status,
                                      result.final_offset, sd.lcg_state));
        meta.write(i * 2u + 1u,
                   make_uint4(pool ? pool->count() : UInt{0u},
                              pool ? pool->left() : UInt{0u}, 0u, 0u));
      }};
}

bool near(float actual, float expected) {
  // Relative scaling keeps the below-cutoff cases meaningful; 1 ULP is not
  // the compatibility boundary and no software floating-point path is used.
  return std::isfinite(actual) &&
         std::abs(actual - expected) <=
             5.0e-5f * std::max(std::abs(expected), 1.0e-8f);
}

bool run(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream(StreamTag::COMPUTE);
  const auto image = psycles::test_support::make_light_emission_image();
  const auto table =
      psycles::luisa_backend::detail::make_cycles_bsdf_table_values({});
  auto words = device.create_buffer<unsigned>(image.size());
  auto tables = device.create_buffer<float>(table.size());
  auto cosines = device.create_buffer<float>(case_count);
  std::array<float, case_count> cosine_values{};
  for (auto i = 0u; i < case_count; ++i) {
    cosine_values[i] = light_emission_cases[i].cosine;
  }
  auto output = device.create_buffer<luisa::float4>(case_count * 3u);
  auto meta = device.create_buffer<luisa::uint4>(case_count * 2u);
  stream << words.copy_from(image.data()) << tables.copy_from(table.data())
         << cosines.copy_from(cosine_values.data()) << synchronize();
  bool passed = true;
  for (const auto has_storage : {false, true}) {
    for (const auto diagnose : {true, false}) {
      // The diagnostic form runs first so the uncorrected null-pool case is
      // a reproducible status failure rather than a production unreachable.
      ShaderOption options;
      options.enable_cache =
          !(argc > 2 && std::string_view{argv[2]} == "--no-cache");
      auto shader = device.compile(make_kernel(has_storage, diagnose), options);
      std::array<luisa::float4, case_count * 3u> values{};
      std::array<luisa::uint4, case_count * 2u> statuses{};
      stream
          << shader(words, tables, cosines, output, meta).dispatch(case_count)
          << output.copy_to(values.data()) << meta.copy_to(statuses.data())
          << synchronize();
      std::ifstream oracle{PSYCLES_LIGHT_EMISSION_ORACLE};
      if (!oracle) {
        std::cerr << "Missing Cycles light-emission oracle\n";
        return false;
      }
      bool variant_passed = true;
      for (auto i = 0u; i < case_count; ++i) {
        unsigned scenario{}, flags{}, offset{}, count{}, left{};
        luisa::float3 emission{}, extinction{};
        if (!(oracle >> scenario >> emission.x >> emission.y >> emission.z >>
              flags >> extinction.x >> extinction.y >> extinction.z >> offset >>
              count >> left) ||
            scenario != i) {
          std::cerr << "Malformed Cycles oracle\n";
          return false;
        }
        const auto e = values[i * 3u];
        const auto t = values[i * 3u + 1u];
        const auto s = statuses[i * 2u];
        const auto p = statuses[i * 2u + 1u];
        const auto weight = values[i * 3u + 2u];
        const auto state_matches =
            !diagnose ||
            (s.y == static_cast<unsigned>(svm::EvaluationStatus::ended) &&
             s.z == offset && offset == image[i * 4u + 2u] + 1u &&
             weight.x == 0.125f && weight.y == 0.25f && weight.z == 0.5f);
        const auto matches = near(e.x, emission.x) && near(e.y, emission.y) &&
                             near(e.z, emission.z) && near(t.x, extinction.x) &&
                             near(t.y, extinction.y) &&
                             near(t.z, extinction.z) && s.x == flags &&
                             s.w == 0xabcdu && p.x == count && p.y == left &&
                             state_matches;
        if (!matches) {
          std::cerr << "storage=" << has_storage << " diagnostic=" << diagnose
                    << " case=" << i << " emission=(" << e.x << ',' << e.y
                    << ',' << e.z << ") expected=(" << emission.x << ','
                    << emission.y << ',' << emission.z
                    << ") flags/status/pc=" << s.x << '/' << s.y << '/' << s.z
                    << " expected flags/pc=" << flags << '/' << offset << '\n';
          variant_passed = false;
        }
      }
      passed &= variant_passed;
      if (!variant_passed && diagnose) {
        break;
      }
    }
  }
  if (passed) {
    std::cout << "Cycles closure-free light-emission tests passed\n";
  }
  return passed;
}

} // namespace

int main(int argc, char **argv) {
  return run(argc, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}
