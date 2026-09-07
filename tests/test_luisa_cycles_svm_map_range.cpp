#include "cycles_svm_map_range_fixture.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <algorithm>
#include <cmath>
#include <iostream>

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

void run(Device &device) {
  const auto images = read_map_range_images(PSYCLES_MAP_RANGE_WORDS);
  constexpr auto count = map_range_points.size();
  auto stream = device.create_stream();
  auto words = device.create_buffer<unsigned>(256u);
  auto input = device.create_buffer<luisa::float4>(count);
  auto output = device.create_buffer<luisa::float4>(count);
  auto metadata = device.create_buffer<luisa::uint4>(count);
  std::array<luisa::float4, count> points{};
  for (auto i = 0u; i < count; ++i) {
    const auto p = map_range_points[i];
    points[i] = {p.x, p.y, p.z, 0.0f};
  }
  stream << input.copy_from(points.data());
  struct Expected {
    std::array<luisa::float3, 3> emission;
    unsigned flag;
  };
  std::vector<std::array<std::array<Expected, 3>, count>> expected(
      images.size());
  std::ifstream file{PSYCLES_MAP_RANGE_ORACLE};
  std::ifstream rounding{PSYCLES_MAP_RANGE_ROUNDING};
  for (auto m = 0u; m < images.size(); ++m) {
    for (auto i = 0u; i < count; ++i) {
      for (auto d = 0u; d < 3u; ++d) {
        unsigned model{}, index{}, domain{};
        auto &e = expected[m][i][d];
        file >> model >> index >> domain >> e.emission[0].x >> e.emission[0].y >>
            e.emission[0].z >> e.flag;
        require(bool(file) && model == m && index == i && domain == d,
                "invalid Cycles Map Range oracle");
        rounding >> model >> index >> domain >> e.emission[1].x >> e.emission[1].y >>
            e.emission[1].z >> e.emission[2].x >> e.emission[2].y >> e.emission[2].z;
        require(bool(rounding) && model == m && index == i && domain == d,
                "invalid Cycles Map Range rounding oracle");
      }
    }
  }
  std::string trailing;
  require(!(file >> trailing) && !(rounding >> trailing), "trailing Map Range oracle data");
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
  std::array<bool, abi::NODE_NUM> used{};
  for (const auto node :
       {abi::NODE_SHADER_JUMP, abi::NODE_END, abi::NODE_GEOMETRY,
        abi::NODE_SEPARATE_VECTOR, abi::NODE_CONVERT, abi::NODE_MAP_RANGE,
        abi::NODE_VECTOR_MAP_RANGE, abi::NODE_CLAMP, abi::NODE_EMISSION_WEIGHT,
        abi::NODE_CLOSURE_EMISSION}) {
    used[node] = true;
  }
  for (auto domain = 0u; domain < domains.size(); ++domain) {
    Kernel1D<Buffer<unsigned>, Buffer<luisa::float4>, Buffer<luisa::float4>,
             Buffer<luisa::uint4>>
        kernel = [=](BufferUInt source, BufferFloat4 inputs, BufferFloat4 out,
                     BufferVar<luisa::uint4> meta) {
          const auto i = dispatch_x();
          const auto point = inputs.read(i).xyz();
          const auto identity = make_float4x4(1.0f);
          svm::ShaderData sd{point,
                             make_float3(0, 0, 1),
                             make_float3(0, 0, 1),
                             make_float3(0, 0, 1),
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
          const DefaultCyclesSvmKernelGlobals kg;
          svm::EvaluationResult status;
          svm::eval_nodes(kg, source, domains[domain], 0u, masks[domain], used,
                          transforms, sd, path, status);
          out.write(i, make_float4(sd.closure_emission_background, 1.0f));
          meta.write(
              i, make_uint4(sd.flag, status.status, status.final_offset, 0u));
        };
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
    for (auto model = 0u; model < images.size(); ++model) {
      const auto &image = images[model];
      std::array<luisa::float4, count> values{};
      std::array<luisa::uint4, count> states{};
      const auto end = domain == 0u ? image.words[2]
                       : domain == 1u
                           ? image.words[3]
                           : static_cast<unsigned>(image.words.size());
      stream << words.view(0u, image.words.size()).copy_from(image.words.data())
             << shader(words, input, output, metadata).dispatch(count)
             << output.copy_to(values.data()) << metadata.copy_to(states.data())
             << synchronize();
      for (auto i = 0u; i < count; ++i) {
        const auto e = expected[model][i][domain];
        if (states[i].x != e.flag ||
            states[i].y != unsigned(svm::EvaluationStatus::ended) ||
            states[i].z != end) {
          std::cerr << image.name << " case=" << i << " domain=" << domain
                    << " flag=" << states[i].x << " expected=" << e.flag
                    << " status=" << states[i].y << " PC=" << states[i].z
                    << " expected PC=" << end << '\n';
          throw std::runtime_error{
              "Map Range flag/status/PC differs from Cycles"};
        }
        for (auto c = 0u; c < 3u; ++c) {
          // A one-ULP difference before floor can select an adjacent step.
          // Accept only the discrete outputs from three original Cycles HIP
          // runs (central, floor input -1/+1 ULP), not an enlarged RGB interval.
          const auto matches = std::any_of(e.emission.begin(), e.emission.end(),
              [&](const auto &candidate) {
                return std::abs(values[i][c] - candidate[c]) <=
                    3.0e-5f * std::max(1.0f, std::abs(candidate[c]));
              });
          if (!std::isfinite(values[i][c]) || !matches) {
            std::cerr << image.name << " case=" << i << " domain=" << domain
                      << " channel=" << c << " actual=" << values[i][c]
                      << " Cycles=" << e.emission[0][c]
                      << " adjacent=" << e.emission[1][c] << ',' << e.emission[2][c] << '\n';
            throw std::runtime_error{
                "Map Range RGB differs from original Cycles HIP"};
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
    std::cout << "1152 original Cycles HIP Map Range/domain cases passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
