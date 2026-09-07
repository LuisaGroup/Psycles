#include "cycles_svm_shared_closure_test_support.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <psycles/compiler/cycles_svm_scene.h>

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
} // namespace

int main(int argc, char **argv) {
  try {
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
    for (auto capacity_index = 0u; capacity_index < capacities.size();
         ++capacity_index) {
      const auto capacity = capacities[capacity_index];
      Kernel1D<Buffer<unsigned>, Buffer<float>, Buffer<unsigned>> kernel =
          [=, used = table.table.node_types_used,
           features = table.kernel_features,
           stack_size = table.table.peak_stack_usage](
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
                              used, transforms, sd, path, result, stack_size);
            }
            $else {
              svm::eval_nodes(globals, words, abi::SHADER_TYPE_VOLUME, features,
                              svm::kernel_feature_node_mask_volume, used,
                              transforms, sd, path, result, stack_size);
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
    std::cout << "Original Cycles shared-closure state passed on " << backend
              << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
