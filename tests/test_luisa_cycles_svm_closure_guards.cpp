#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "cycles_svm_closure_guards_test_data.h"
#include "luisa_cycles_svm_test_kernel_globals.h"
#include "path_tracer_bsdf_tables.h"

namespace {
using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace svm = psycles::luisa_backend::cycles_svm;
namespace fixture = psycles::test_support::closure_guards;

struct Oracle {
  std::array<unsigned, 4u> metadata;
  std::array<float, fixture::value_count> values;
};

class Globals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
  Expr<Buffer<float>> _table;
  unsigned _flags;

 public:
  Globals(Expr<Buffer<float>> table, unsigned flags) noexcept
      : _table{table}, _flags{flags} {}
  Float cycles_bsdf_data(Expr<unsigned> offset) const noexcept override {
    return _table.read(offset);
  }
  Bool caustics_reflective() const noexcept override {
    return (_flags & 1u) != 0u;
  }
  Bool caustics_refractive() const noexcept override {
    return (_flags & 2u) != 0u;
  }
};

std::array<Oracle, 96u> read_oracle() {
  std::ifstream input{PSYCLES_CLOSURE_GUARDS_ORACLE};
  std::array<Oracle, 96u> result;
  for (unsigned i = 0; i < result.size(); i++) {
    unsigned ordinal;
    if (!(input >> ordinal) || ordinal != i) {
      throw std::runtime_error{"invalid Cycles oracle ordinal"};
    }
    for (auto& value : result[i].metadata) {
      if (!(input >> value)) {
        throw std::runtime_error{"truncated Cycles metadata"};
      }
    }
    for (auto& value : result[i].values) {
      if (!(input >> value) || !std::isfinite(value)) {
        throw std::runtime_error{"invalid Cycles values"};
      }
    }
  }
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string_view backend{argc > 1 ? argv[1] : "fallback"};
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  std::vector<unsigned> image(16u);
  std::array<unsigned, 4u> final_offsets;
  for (unsigned shader = 0u; shader < fixture::programs.size(); shader++) {
    const auto local = fixture::programs[shader];
    const auto start = static_cast<unsigned>(image.size());
    image[shader * 4u] = local[0];
    for (auto j = 1u; j < 4u; j++) {
      image[shader * 4u + j] = local[j] - 4u + start;
    }
    final_offsets[shader] = image[shader * 4u + 2u];
    image.insert(image.end(), local.begin() + 4u, local.end());
  }
  auto words = device.create_buffer<unsigned>(image.size());
  const auto table_values =
      psycles::luisa_backend::detail::make_cycles_bsdf_table_values({});
  auto table = device.create_buffer<float>(table_values.size());
  auto values = device.create_buffer<luisa::float4>(8u * 4u);
  auto metadata = device.create_buffer<luisa::uint4>(8u * 2u);
  stream << words.copy_from(image.data())
         << table.copy_from(table_values.data()) << synchronize();
  const auto oracle = read_oracle();
  auto checked = 0u;
  auto failures = 0u;
  for (unsigned flags = 0u; flags < 4u; flags++) {
    // The fourth specialization has genuinely no physical closure storage;
    // compare it to original Cycles num_closure_left=0, including final PC.
    for (unsigned storage = 0u; storage < 4u; storage++) {
      const auto capacity_index = storage == 3u ? 0u : storage;
      const auto capacity = fixture::capacities[capacity_index];
      Kernel1D kernel = [=](BufferUInt words, BufferFloat table,
                            BufferFloat4 values, BufferUInt4 metadata) {
        const UInt i = dispatch_x();
        const UInt shader_index = i / 2u;
        const auto identity = make_float4x4(1.0f);
        const auto normal = normalize(make_float3(0.2f, -0.3f, 1.0f));
        svm::ClosurePool pool{capacity};
        svm::ShaderData sd{make_float3(0.0f),
                           normal,
                           normal,
                           normal,
                           svm::primitive_triangle,
                           shader_index,
                           0u,
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
                           storage == 3u ? nullptr : &pool};
        const Globals kg{table, flags};
        const svm::PathState path{
            select(svm::path_ray_visibility_camera,
                   svm::path_ray_visibility_diffuse, (i & 1u) != 0u),
            0u};
        const svm::TransformState transforms{identity, identity, identity,
                                             identity};
        std::array<bool, NODE_NUM> used{};
        for (auto node : {NODE_END, NODE_SHADER_JUMP, NODE_GEOMETRY,
                          NODE_CLOSURE_SET_WEIGHT, NODE_CLOSURE_BSDF}) {
          used[node] = true;
        }
        svm::EvaluationResult result;
        svm::eval_nodes(kg, words, SHADER_TYPE_SURFACE, 0u,
                        svm::kernel_feature_node_mask_surface, used, transforms,
                        sd, path, result, 3u);
        for (unsigned j = 0; j < 4u; j++) {
          values.write(i * 4u + j, make_float4(0.0f));
        }
        UInt type = 0u;
        if (capacity != 0u) {
          $if(pool.count() != 0u) {
            const auto common = pool.common(0u);
            const auto param = pool.microfacet_param(0u);
            type = common.type;
            values.write(i * 4u, make_float4(common.N, param.alpha_x));
            values.write(i * 4u + 1u,
                         make_float4(common.weight, common.sample_weight));
            values.write(i * 4u + 2u, make_float4(param.T, param.alpha_y));
            // Same live-field observation as the original GPU harness:
            // energy_scale is initialized/consumed only by GGX.
            const Bool has_energy_scale =
                (common.type ==
                 static_cast<unsigned>(CLOSURE_BSDF_MICROFACET_GGX_ID)) |
                (common.type ==
                 static_cast<unsigned>(
                     CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID)) |
                (common.type ==
                 static_cast<unsigned>(CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID));
            values.write(
                i * 4u + 3u,
                make_float4(param.ior,
                            select(0.0f, param.energy_scale, has_energy_scale),
                            param.fresnel_type.cast<float>(), 0.0f));
          };
        }
        metadata.write(i * 2u,
                       make_uint4(sd.flag, pool.count(), pool.left(), type));
        metadata.write(i * 2u + 1u,
                       make_uint4(result.status, result.final_offset, 0u, 0u));
      };
      auto shader = device.compile(
          kernel,
          ShaderOption{.enable_cache = false, .enable_fast_math = true});
      std::array<luisa::float4, 32u> actual_values;
      std::array<luisa::uint4, 16u> actual_metadata;
      stream << shader(words, table, values, metadata).dispatch(8u)
             << values.copy_to(actual_values.data())
             << metadata.copy_to(actual_metadata.data()) << synchronize();
      for (unsigned i = 0u; i < 8u; i++) {
        const auto ordinal =
            flags * fixture::cases_per_flags + i * 3u + capacity_index;
        const auto& expected = oracle[ordinal];
        const auto& meta = actual_metadata[i * 2u];
        bool valid =
            meta.x == expected.metadata[0] && meta.y == expected.metadata[1] &&
            meta.z == expected.metadata[2] && meta.w == expected.metadata[3];
        valid &= actual_metadata[i * 2u + 1u].x ==
                 static_cast<unsigned>(svm::EvaluationStatus::ended);
        valid &= actual_metadata[i * 2u + 1u].y == final_offsets[i / 2u];
        for (unsigned j = 0u; j < fixture::value_count; j++) {
          const auto& value = actual_values[i * 4u + j / 4u];
          const std::array lanes{value.x, value.y, value.z, value.w};
          const auto actual = lanes[j % 4u];
          valid &= std::isfinite(actual) &&
                   std::abs(actual - expected.values[j]) <=
                       2.0e-5f * std::max(1.0f, std::abs(expected.values[j]));
        }
        checked++;
        if (!valid) {
          failures++;
          std::cerr << "Original closure guard mismatch: flags=" << flags
                    << " storage=" << storage << " case=" << i
                    << " meta=" << meta.x << ',' << meta.y << ',' << meta.z
                    << ',' << meta.w << '\n';
        }
      }
    }
  }
  std::cout
      << "Cycles closure guards on " << backend << ": " << checked - failures
      << '/' << checked
      << " cases passed (96 original GPU states plus 32 no-storage controls)\n";
  return failures != 0u;
}
