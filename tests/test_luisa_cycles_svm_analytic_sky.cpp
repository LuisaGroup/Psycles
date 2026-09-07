#include "cycles_svm_analytic_sky_fixture.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using namespace luisa::compute;
using namespace psycles::test_support;
namespace abi = psycles::compiler::cycles_svm;
namespace svm = psycles::luisa_backend::cycles_svm;

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error{message};
  }
}

class SkyGlobals final : public DefaultCyclesSvmKernelGlobals {
  [[nodiscard]] Float3 row(unsigned i) const noexcept {
    const auto v = analytic_sky_xyz_to_rgb[i];
    return make_float3(v.x, v.y, v.z);
  }

public:
  Float3 film_xyz_to_r() const noexcept override { return row(0u); }
  Float3 film_xyz_to_g() const noexcept override { return row(1u); }
  Float3 film_xyz_to_b() const noexcept override { return row(2u); }
};

void run(Device &device) {
  constexpr auto count = analytic_sky_directions.size();
  auto stream = device.create_stream();
  auto words = device.create_buffer<unsigned>(52u);
  auto input = device.create_buffer<luisa::float4>(count);
  auto output = device.create_buffer<luisa::float4>(count);
  auto metadata = device.create_buffer<luisa::uint4>(count);
  std::array<luisa::float4, count> directions{};
  for (auto i = 0u; i < count; ++i) {
    const auto d = analytic_sky_directions[i];
    directions[i] = {d.x, d.y, d.z, 0.0f};
  }
  stream << input.copy_from(directions.data());
  struct Expected {
    luisa::float3 emission;
    unsigned flag;
    luisa::float3 lower;
    luisa::float3 upper;
  };
  std::array<std::array<std::array<Expected, 3>, count>, 2> expected{};
  std::ifstream file{PSYCLES_ANALYTIC_SKY_ORACLE};
  std::ifstream rounding{PSYCLES_ANALYTIC_SKY_ROUNDING};
  for (auto m = 0u; m < 2u; ++m) {
    for (auto i = 0u; i < count; ++i) {
      for (auto d = 0u; d < 3u; ++d) {
        unsigned model{}, index{}, domain{};
        auto &e = expected[m][i][d];
        file >> model >> index >> domain >> e.emission.x >> e.emission.y >>
            e.emission.z >> e.flag;
        require(bool(file) && model == m && index == i && domain == d,
                "invalid Cycles sky oracle");
        rounding >> model >> index >> domain >> e.lower.x >> e.lower.y >> e.lower.z
                 >> e.upper.x >> e.upper.y >> e.upper.z;
        require(bool(rounding) && model == m && index == i && domain == d,
                "invalid Cycles sky rounding envelope");
        for (auto c = 0u; c < 3u; ++c) {
          require(std::isfinite(e.lower[c]) && std::isfinite(e.upper[c]) &&
                      e.lower[c] <= e.emission[c] && e.upper[c] >= e.emission[c],
                  "sky rounding envelope excludes the unmodified Cycles result");
        }
      }
    }
  }
  constexpr std::array paths{PSYCLES_HOSEK_WORDS, PSYCLES_PREETHAM_WORDS};
  constexpr std::array domains{abi::SHADER_TYPE_SURFACE,
                               abi::SHADER_TYPE_VOLUME,
                               abi::SHADER_TYPE_DISPLACEMENT};
  constexpr std::array masks{
      svm::kernel_feature_node_mask_surface,
      svm::kernel_feature_node_emission | svm::kernel_feature_node_volume |
          svm::kernel_feature_node_voronoi_extra |
          svm::kernel_feature_node_light_path | svm::kernel_feature_node_portal,
      svm::kernel_feature_node_voronoi_extra | svm::kernel_feature_node_bump |
          svm::kernel_feature_node_bump_state |
          svm::kernel_feature_node_portal};
  constexpr std::array end_offsets{50u, 51u, 52u};
  std::array<bool, abi::NODE_NUM> used{};
  for (const auto node :
       {abi::NODE_SHADER_JUMP, abi::NODE_END, abi::NODE_GEOMETRY,
        abi::NODE_TEX_SKY, abi::NODE_EMISSION_WEIGHT,
        abi::NODE_CLOSURE_BACKGROUND}) {
    used[node] = true;
  }
  for (auto domain = 0u; domain < domains.size(); ++domain) {
    Kernel1D<Buffer<unsigned>, Buffer<luisa::float4>, Buffer<luisa::float4>,
             Buffer<luisa::uint4>>
        kernel = [=](BufferUInt source, BufferFloat4 inputs, BufferFloat4 out,
                     BufferVar<luisa::uint4> meta) {
          const auto i = dispatch_x();
          const auto direction = inputs.read(i).xyz();
          const auto identity = make_float4x4(1.0f);
          svm::ShaderData sd{direction,
                             make_float3(0, 0, 1),
                             make_float3(0, 0, 1),
                             -direction,
                             0u,
                             0u,
                             0u,
                             0u,
                             ~0u,
                             0.0f,
                             0.0f,
                             svm::object_none,
                             0.0f,
                             1.0f,
                             0.0f,
                             0.0f,
                             0.0f,
                             0.0f,
                             0.0f,
                             0.0f,
                             make_float3(0.0f),
                             make_float3(0.0f),
                             identity,
                             identity};
          sd.closure_emission_background = make_float3(-1.0f, -2.0f, -3.0f);
          const svm::TransformState transforms{identity, identity, identity,
                                               identity};
          const svm::PathState path{svm::path_ray_visibility_camera, 0u};
          const SkyGlobals kg;
          svm::EvaluationResult status;
          svm::eval_nodes(kg, source, domains[domain], 0u, masks[domain], used,
                          transforms, sd, path, status);
          out.write(i, make_float4(sd.closure_emission_background, 1.0f));
          meta.write(
              i, make_uint4(sd.flag, status.status, status.final_offset, 0u));
        };
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
    for (auto model = 0u; model < paths.size(); ++model) {
      const auto image = read_analytic_sky_words(paths[model]);
      std::array<luisa::float4, count> values{};
      std::array<luisa::uint4, count> states{};
      stream << words.copy_from(image.data())
             << shader(words, input, output, metadata).dispatch(count)
             << output.copy_to(values.data()) << metadata.copy_to(states.data())
             << synchronize();
      for (auto i = 0u; i < count; ++i) {
        const auto e = expected[model][i][domain];
        require(states[i].x == e.flag &&
                    states[i].y == unsigned(svm::EvaluationStatus::ended) &&
                    states[i].z == end_offsets[domain],
                "analytic sky flag/status/PC differs from Cycles");
        for (auto c = 0u; c < 3u; ++c) {
          // acos is ill-conditioned at +/-1. The original Cycles HIP
          // interpreter propagates adjacent-float cosine inputs to this
          // envelope; this is not a backend- or case-specific relaxed bound.
          const auto tolerance = 3.0e-5f * std::max(1.0f, std::abs(e.emission[c]));
          if (!std::isfinite(values[i][c]) ||
              values[i][c] < e.lower[c] - tolerance ||
              values[i][c] > e.upper[c] + tolerance) {
            std::cerr << "model=" << model << " case=" << i
                      << " domain=" << domain << " channel=" << c
                      << " actual=" << values[i][c]
                      << " Cycles=" << e.emission[c] << '\n';
            throw std::runtime_error{
                "analytic sky RGB differs from original Cycles HIP"};
          }
        }
      }
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    Context context{argv[0]};
    auto device = context.create_device(argc > 1 ? argv[1] : "hip");
    run(device);
    std::cout << "96 original Cycles HIP analytic sky/domain cases passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
