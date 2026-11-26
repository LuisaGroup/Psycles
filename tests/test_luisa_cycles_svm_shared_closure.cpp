#include "cycles_svm_shared_closure_test_support.h"
#include "cycles_svm_test_ast_shape.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <psycles/compiler/cycles_svm_scene.h>
#include <psycles/luisa/cycles_svm_translator.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {
using namespace luisa::compute;
using namespace psycles::test_support::shared_closure;
namespace abi = psycles::compiler::cycles_svm;
namespace svm = psycles::luisa_backend::cycles_svm;

struct Oracle {
  std::array<unsigned, 4u> meta;
  std::array<float, 10u> values;
};

std::array<Oracle, 48u> read_oracle() {
  std::array<Oracle, 48u> result;
  std::ifstream input{PSYCLES_SHARED_CLOSURE_STATE_ORACLE};
  for (auto i = 0u; i < result.size(); ++i) {
    unsigned ordinal;
    require(bool(input >> ordinal) && ordinal == i,
            "invalid original closure state ordinal");
    for (auto &value : result[i].meta) {
      require(bool(input >> value), "truncated original closure state");
    }
    for (auto &value : result[i].values) {
      require(bool(input >> value), "truncated original closure values");
    }
  }
  return result;
}

bool within_one_ulp(float actual, float expected) {
  return actual == expected ||
         actual ==
             std::nextafter(expected, std::numeric_limits<float>::infinity()) ||
         actual ==
             std::nextafter(expected, -std::numeric_limits<float>::infinity());
}

void test_auto_direct_jit_weight_fallback() {
  if (std::getenv("PSYCLES_SVM_BACKEND") != nullptr ||
      std::getenv("PSYCLES_SVM_JIT_MAX_WEIGHT") != nullptr) {
    return;
  }
  auto make_heavy_shader = [](std::size_t voronoi_count) {
    abi::SvmShaderIr shader_ir;
    auto &surface =
        shader_ir.routines[static_cast<std::size_t>(abi::SHADER_TYPE_SURFACE)];
    surface.resize(voronoi_count);
    for (auto &op : surface) {
      op.instruction.opcode = abi::NODE_TEX_VORONOI;
    }
    return shader_ir;
  };
  abi::ShaderTableImage table;
  table.valid = true;
  table.ir.valid = true;
  table.words = {0u};
  table.shader_count = 3u;
  table.node_types_used[abi::NODE_TEX_VORONOI] = true;
  // Derive Voronoi counts from direct_jit_max_weight() (36 per Voronoi op).
  const std::size_t half_ops = svm::direct_jit_max_weight() / 72u + 1u;
  table.ir.shaders.push_back(make_heavy_shader(half_ops));
  table.ir.shaders.push_back(make_heavy_shader(half_ops));
  table.ir.shaders.push_back(make_heavy_shader(half_ops + 1u));

  const auto check_strategy = [&](std::span<const std::uint8_t> mask,
                                  svm::SvmBackendStrategy preferred) {
    svm::SvmBackendStrategy actual = svm::SvmBackendStrategy::auto_select;
    const Kernel1D<Buffer<unsigned>> ast_only = [&](BufferUInt words) {
      const auto identity = make_float4x4(1.0f);
      const auto normal = make_float3(0.0f, 0.0f, 1.0f);
      svm::ClosurePool pool{0u};
      svm::ShaderData sd{make_float3(0.0f), normal, normal, normal,
                         svm::primitive_triangle, 0u, 0u, 0u, 0u, 0.0f, 0.0f,
                         0u, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                         make_float3(1.0f, 0.0f, 0.0f),
                         make_float3(0.0f, 1.0f, 0.0f), identity, identity, 0u,
                         &pool};
      const svm::PathState path{svm::path_ray_visibility_camera, 0u};
      const svm::TransformState transforms{identity, identity, identity,
                                           identity};
      const psycles::test_support::DefaultCyclesSvmKernelGlobals globals;
      actual = svm::resolve_svm_backend_strategy(
          svm::SvmTranslationRequest{
              .kernel_globals = globals,
              .table = table,
              .words = words,
              .shader_type = abi::SHADER_TYPE_SURFACE,
              .kernel_features = 0u,
              .node_feature_mask = svm::kernel_feature_node_mask_surface,
              .node_types_used = table.node_types_used,
              .transform_state = transforms,
              .shader_data = sd,
              .path_state = path,
              .stack_size = 1u,
              .active_shader_mask = mask},
          preferred);
    };
    (void)ast_only;
    return actual;
  };

  constexpr std::array<std::uint8_t, 3u> identical_pair{1u, 1u, 0u};
  constexpr std::array<std::uint8_t, 3u> distinct_pair{1u, 0u, 1u};
  constexpr std::array<std::uint8_t, 3u> single_heavy{0u, 0u, 1u};
  require(check_strategy(identical_pair, svm::SvmBackendStrategy::auto_select) ==
              svm::SvmBackendStrategy::direct_jit,
          "auto_select failed to deduplicate identical heavy routines");
  require(check_strategy(single_heavy, svm::SvmBackendStrategy::auto_select) ==
              svm::SvmBackendStrategy::direct_jit,
          "auto_select rejected single routine under weight limit");
  require(check_strategy(distinct_pair, svm::SvmBackendStrategy::auto_select) ==
              svm::SvmBackendStrategy::bytecode_vm,
          "auto_select failed to fall back to bytecode_vm above unique weight limit");
  require(check_strategy(distinct_pair, svm::SvmBackendStrategy::direct_jit) ==
              svm::SvmBackendStrategy::direct_jit,
          "explicit direct_jit preference was overridden by auto weight limit");
}
} // namespace

int main(int argc, char **argv) {
  try {
    test_auto_direct_jit_weight_fallback();
    const std::string_view backend{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    std::array<psycles::compiler::ShaderCompilation, 4u> programs;
    std::array<abi::ShaderTableCompileUnit, 4u> units;
    const psycles::compiler::ShaderCompiler frontend{
        psycles::compiler::make_core_node_registry()};
    for (auto i = 0u; i < 4u; ++i) {
      programs[i] = frontend.compile(make_graph(i));
      require(programs[i].ok(), "shared-closure graph did not normalize");
      units[i] = {.shader_index = i, .shader = programs[i].program.get()};
    }
    const auto table = abi::compile_shader_table(units);
    require(table.table.valid, table.table.diagnostic);
    auto words = device.create_buffer<unsigned>(table.table.words.size());
    auto values = device.create_buffer<float>(16u * 10u);
    auto metadata = device.create_buffer<unsigned>(16u * 6u);
    stream << words.copy_from(table.table.words.data());
    const auto oracle = read_oracle();
    constexpr std::array capacities{0u, 1u, 64u};
    // Both specializations execute the unchanged original table. Compare
    // each one directly against Cycles GPU state, not against each other.
    for (auto run = 0u; run < 2u * capacities.size(); ++run) {
      const auto capacity_index = run % capacities.size();
      const auto specialized = run >= capacities.size();
      const auto capacity = capacities[capacity_index];
      const abi::ShaderEntryUsage whole{table.table.node_types_used,
                                         table.table.peak_stack_usage};
      const auto surface = specialized
                               ? table.table.usage_for(abi::SHADER_TYPE_SURFACE)
                               : whole;
      const auto volume = specialized
                              ? table.table.usage_for(abi::SHADER_TYPE_VOLUME)
                              : whole;
      Kernel1D<Buffer<unsigned>, Buffer<float>, Buffer<unsigned>> kernel =
          [=, features = table.kernel_features](
              BufferUInt words, BufferFloat values, BufferUInt metadata) {
            const UInt i = dispatch_x();
            const auto identity = make_float4x4(1.0f);
            const auto normal = make_float3(0.0f, 0.0f, 1.0f);
            svm::ClosurePool pool{capacity};
            svm::ShaderData sd{
                make_float3(0.0f),
                normal,
                normal,
                normal,
                select(svm::primitive_triangle, svm::primitive_volume, i >= 8u),
                i / 4u,
                select(0u, svm::shader_data_is_volume_shader_eval, i >= 8u),
                0u,
                svm::primitive_none,
                0.0f,
                0.0f,
                svm::object_none,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                make_float3(0.0f),
                make_float3(0.0f),
                identity,
                identity,
                0u,
                &pool};
            UInt visibility = svm::path_ray_visibility_camera;
            $if(i % 4u == 1u) {
              visibility = svm::path_ray_visibility_diffuse;
            };
            $if(i % 4u == 2u) { visibility = svm::path_ray_visibility_glossy; };
            $if(i % 4u == 3u) { visibility = svm::path_ray_visibility_shadow; };
            const svm::PathState path{visibility, 0u};
            const svm::TransformState transforms{identity, identity, identity,
                                                 identity};
            const psycles::test_support::DefaultCyclesSvmKernelGlobals globals;
            svm::EvaluationResult result;
            $if(i < 8u) {
              svm::eval_nodes(globals, words, abi::SHADER_TYPE_SURFACE,
                              features, svm::kernel_feature_node_mask_surface,
                              surface.node_types_used, transforms, sd, path,
                              result, std::max(1u, surface.peak_stack_usage));
            }
            $else {
              svm::eval_nodes(globals, words, abi::SHADER_TYPE_VOLUME, features,
                              svm::kernel_feature_node_mask_volume,
                              volume.node_types_used, transforms, sd, path,
                              result, std::max(1u, volume.peak_stack_usage));
            };
            const auto put = [&](unsigned offset, Expr<luisa::float3> value) {
              values.write(i * 10u + offset, value.x);
              values.write(i * 10u + offset + 1u, value.y);
              values.write(i * 10u + offset + 2u, value.z);
            };
            put(0u, sd.closure_transparent_extinction);
            put(3u, sd.closure_emission_background);
            put(6u, make_float3(0.0f));
            values.write(i * 10u + 9u, 0.0f);
            metadata.write(i * 6u, sd.flag);
            metadata.write(i * 6u + 1u, pool.count());
            metadata.write(i * 6u + 2u, pool.left());
            metadata.write(i * 6u + 3u, 0u);
            metadata.write(i * 6u + 4u, result.status);
            metadata.write(i * 6u + 5u, result.final_offset);
            if (capacity != 0u) {
              $if(pool.count() != 0u) {
                const auto closure = pool.common(0u);
                metadata.write(i * 6u + 3u, closure.type);
                put(6u, closure.weight);
                values.write(i * 10u + 9u, closure.sample_weight);
              };
            }
          };
      auto shader =
          device.compile(kernel, ShaderOption{.enable_cache = false,
                                              .enable_fast_math = true});
      std::array<float, 160u> actual_values;
      std::array<unsigned, 96u> actual_metadata;
      stream << shader(words, values, metadata).dispatch(16u)
             << values.copy_to(actual_values.data())
             << metadata.copy_to(actual_metadata.data()) << synchronize();
      for (auto i = 0u; i < 16u; ++i) {
        const auto ordinal = i * 3u + capacity_index;
        const auto &expected = oracle[ordinal];
        for (auto j = 0u; j < 4u; ++j) {
          require(actual_metadata[i * 6u + j] == expected.meta[j],
                  "original shared-closure state differs at case " +
                      std::to_string(ordinal) + " field " + std::to_string(j));
        }
        for (auto j = 0u; j < 10u; ++j) {
          require(
              within_one_ulp(actual_values[i * 10u + j], expected.values[j]),
              "original shared-closure value differs at case " +
                  std::to_string(ordinal) + " field " + std::to_string(j));
        }
        const auto pc = table.table.words[(i / 4u) * 4u + (i < 8u ? 2u : 3u)];
        require(actual_metadata[i * 6u + 4u] ==
                        static_cast<unsigned>(svm::EvaluationStatus::ended) &&
                    actual_metadata[i * 6u + 5u] == pc,
                "shared-closure PC/status differs from original full stream");
      }
    }

    // Verify both pluggable re-targetable SVM backend translators
    // (BytecodeVmTranslator and DirectJitTranslator) against the exact Cycles
    // GPU state oracle, and confirm DirectJitTranslator emits zero PC loops.
    for (const auto strategy :
         {svm::SvmBackendStrategy::bytecode_vm,
          svm::SvmBackendStrategy::direct_jit}) {
      for (const auto capacity : capacities) {
        const auto capacity_index =
            capacity == 0u ? 0u : (capacity == 1u ? 1u : 2u);
        const auto surface = table.table.usage_for(abi::SHADER_TYPE_SURFACE);
        const auto volume = table.table.usage_for(abi::SHADER_TYPE_VOLUME);
        Kernel1D<Buffer<unsigned>, Buffer<float>, Buffer<unsigned>> kernel =
            [&, capacity, strategy, surface, volume,
             features = table.kernel_features](
                BufferUInt words, BufferFloat values, BufferUInt metadata) {
              const UInt i = dispatch_x();
              const auto identity = make_float4x4(1.0f);
              const auto normal = make_float3(0.0f, 0.0f, 1.0f);
              svm::ClosurePool pool{capacity};
              svm::ShaderData sd{
                  make_float3(0.0f),
                  normal,
                  normal,
                  normal,
                  select(svm::primitive_triangle, svm::primitive_volume,
                         i >= 8u),
                  i / 4u,
                  select(0u, svm::shader_data_is_volume_shader_eval, i >= 8u),
                  0u,
                  svm::primitive_none,
                  0.0f,
                  0.0f,
                  svm::object_none,
                  0.0f,
                  0.0f,
                  0.0f,
                  0.0f,
                  0.0f,
                  0.0f,
                  0.0f,
                  0.0f,
                  make_float3(0.0f),
                  make_float3(0.0f),
                  identity,
                  identity,
                  0u,
                  &pool};
              UInt visibility = svm::path_ray_visibility_camera;
              $if(i % 4u == 1u) {
                visibility = svm::path_ray_visibility_diffuse;
              };
              $if(i % 4u == 2u) {
                visibility = svm::path_ray_visibility_glossy;
              };
              $if(i % 4u == 3u) {
                visibility = svm::path_ray_visibility_shadow;
              };
              const svm::PathState path{visibility, 0u};
              const svm::TransformState transforms{identity, identity, identity,
                                                   identity};
              const psycles::test_support::DefaultCyclesSvmKernelGlobals
                  globals;
              $if(i < 8u) {
                svm::svm_translator_for(strategy).translate(
                    svm::SvmTranslationRequest{
                        .kernel_globals = globals,
                        .table = table.table,
                        .words = words,
                        .shader_type = abi::SHADER_TYPE_SURFACE,
                        .kernel_features = features,
                        .node_feature_mask = svm::kernel_feature_node_mask_surface,
                        .node_types_used = surface.node_types_used,
                        .transform_state = transforms,
                        .shader_data = sd,
                        .path_state = path,
                        .stack_size = std::max<std::size_t>(
                            1u, surface.peak_stack_usage),
                        .noise_usage = surface.noise_usage});
              }
              $else {
                svm::svm_translator_for(strategy).translate(
                    svm::SvmTranslationRequest{
                        .kernel_globals = globals,
                        .table = table.table,
                        .words = words,
                        .shader_type = abi::SHADER_TYPE_VOLUME,
                        .kernel_features = features,
                        .node_feature_mask = svm::kernel_feature_node_mask_volume,
                        .node_types_used = volume.node_types_used,
                        .transform_state = transforms,
                        .shader_data = sd,
                        .path_state = path,
                        .stack_size = std::max<std::size_t>(
                            1u, volume.peak_stack_usage),
                        .noise_usage = volume.noise_usage});
              };
              const auto put = [&](unsigned offset, Expr<luisa::float3> value) {
                values.write(i * 10u + offset, value.x);
                values.write(i * 10u + offset + 1u, value.y);
                values.write(i * 10u + offset + 2u, value.z);
              };
              put(0u, sd.closure_transparent_extinction);
              put(3u, sd.closure_emission_background);
              put(6u, make_float3(0.0f));
              values.write(i * 10u + 9u, 0.0f);
              metadata.write(i * 6u, sd.flag);
              metadata.write(i * 6u + 1u, pool.count());
              metadata.write(i * 6u + 2u, pool.left());
              metadata.write(i * 6u + 3u, 0u);
              if (capacity != 0u) {
                $if(pool.count() != 0u) {
                  const auto closure = pool.common(0u);
                  metadata.write(i * 6u + 3u, closure.type);
                  put(6u, closure.weight);
                  values.write(i * 10u + 9u, closure.sample_weight);
                };
              }
            };
        std::size_t total_pc_switch_returns = 0u;
        for (const auto &callable :
             kernel.function()->function().custom_callables()) {
          psycles::test_support::InterpreterShape shape;
          callable->body()->accept(shape);
          total_pc_switch_returns += shape.primary_switch_returns;
        }
        if (strategy == svm::SvmBackendStrategy::direct_jit) {
          require(total_pc_switch_returns == 0u,
                  "DirectJitTranslator must emit zero PC-loop opcode switches");
        } else {
          require(total_pc_switch_returns == 2u,
                  "BytecodeVmTranslator must emit one PC-loop opcode switch per domain");
        }
        auto shader = device.compile(
            kernel,
            ShaderOption{.enable_cache = false, .enable_fast_math = true});
        std::array<float, 160u> actual_values;
        std::array<unsigned, 96u> actual_metadata;
        stream << shader(words, values, metadata).dispatch(16u)
               << values.copy_to(actual_values.data())
               << metadata.copy_to(actual_metadata.data()) << synchronize();
        for (auto i = 0u; i < 16u; ++i) {
          const auto ordinal = i * 3u + capacity_index;
          const auto &expected = oracle[ordinal];
          for (auto j = 0u; j < 4u; ++j) {
            require(actual_metadata[i * 6u + j] == expected.meta[j],
                    "re-targetable translator metadata differs at case " +
                        std::to_string(ordinal) + " field " +
                        std::to_string(j));
          }
          for (auto j = 0u; j < 10u; ++j) {
            require(
                within_one_ulp(actual_values[i * 10u + j], expected.values[j]),
                "re-targetable translator value differs at case " +
                    std::to_string(ordinal) + " field " + std::to_string(j));
          }
        }
      }
    }
    std::cout << "Original Cycles shared-closure state passed on " << backend
              << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
