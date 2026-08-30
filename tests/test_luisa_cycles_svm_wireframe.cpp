#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_internal.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace device_svm = psycles::luisa_backend::cycles_svm;

constexpr auto probe_count = 14u;

class ProbeKernelGlobals final : public device_svm::KernelGlobals {
public:
  [[nodiscard]] device_svm::TriangleVertices
  triangle_vertices(Expr<std::uint32_t>,
                    Expr<std::uint32_t> prim) const noexcept override {
    Float3 v0 = make_float3(0.0f, 0.0f, 0.0f);
    Float3 v1 = make_float3(1.0f, 0.0f, 0.0f);
    Float3 v2 = make_float3(0.0f, 1.0f, 0.0f);
    $if(prim == 1u) {
      v1 = v0;
      v2 = v0;
    }
    $elif(prim == 2u) {
      const auto displacement = make_float3(4.0f, 4.0f, 0.0f);
      v0 += displacement;
      v1 += displacement;
      v2 += displacement;
    };
    return {.v0 = v0, .v1 = v1, .v2 = v2};
  }

  [[nodiscard]] device_svm::TriangleVertices
  motion_triangle_vertices(Expr<std::uint32_t>, Expr<std::uint32_t>,
                           Expr<float>) const noexcept override {
    return {.v0 = make_float3(0.0f, 0.0f, 0.0f),
            .v1 = make_float3(1.0f, 0.0f, 0.0f),
            .v2 = make_float3(0.0f, 1.0f, 0.0f)};
  }

  [[nodiscard]] Float3 film_rgb_to_y() const noexcept override {
    return make_float3(0.2126f, 0.7152f, 0.0722f);
  }
};

[[nodiscard]] std::array<bool, NODE_NUM> immediate_node_types() {
  auto types = std::array<bool, NODE_NUM>{};
  for (const auto type : {NODE_END, NODE_SHADER_JUMP, NODE_CLOSURE_EMISSION,
                          NODE_EMISSION_WEIGHT, NODE_CONVERT, NODE_WIREFRAME}) {
    types[type] = true;
  }
  return types;
}

[[nodiscard]] std::array<bool, NODE_NUM> bump_node_types() {
  auto types = immediate_node_types();
  types[NODE_GEOMETRY] = true;
  types[NODE_SET_BUMP] = true;
  return types;
}

[[nodiscard]] auto make_probe_kernel(std::array<bool, NODE_NUM> node_types,
                                     bool bump_feature_enabled) {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<luisa::uint4>>{
      [node_types, bump_feature_enabled](BufferUInt words,
                                         BufferFloat4 floating_output,
                                         BufferUInt4 integer_output) noexcept {
        const UInt index = dispatch_x();
        Float3 position = make_float3(0.01f, 0.2f, 0.0f);
        UInt primitive_type = device_svm::primitive_triangle;
        UInt primitive_id = 0u;
        UInt object_flags = device_svm::shader_data_object_transform_applied;

        $if(index == 1u) { position = make_float3(0.25f, 0.25f, 0.0f); };
        $if(index == 2u) { position = make_float3(0.045f, 0.2f, 0.0f); };
        $if(index == 3u) { primitive_id = 1u; };
        $if(index == 4u) { primitive_id = device_svm::primitive_none; };
        $if(index == 5u) {
          primitive_type = device_svm::primitive_curve_thick;
        };
        $if(index == 6u) {
          primitive_type =
              device_svm::primitive_triangle | device_svm::primitive_motion;
          primitive_id = 2u;
        };
        $if(index == 7u) { primitive_id = 2u; };
        $if(index == 8u) {
          position = make_float3(2.01f, 0.2f, 0.0f);
          object_flags = 0u;
        };
        $if(index == 9u) { position = make_float3(2.01f, 0.2f, 0.0f); };
        $if(index == 10u) { position = make_float3(0.3f, 0.3f, 0.0f); };
        $if(index == 11u) { position = make_float3(0.2f, 0.3f, 0.0f); };
        $if(index == 12u) { position = make_float3(0.2f, 0.1f, 0.0f); };
        $if(index == 13u) { position = make_float3(0.2f, 0.2f, 0.0f); };

        const auto identity = make_float4x4(1.0f);
        const auto object_to_world =
            make_float4x4(make_float4(1.0f, 0.0f, 0.0f, 0.0f),
                          make_float4(0.0f, 1.0f, 0.0f, 0.0f),
                          make_float4(0.0f, 0.0f, 1.0f, 0.0f),
                          make_float4(2.0f, 0.0f, 0.0f, 1.0f));
        const auto world_to_object =
            make_float4x4(make_float4(1.0f, 0.0f, 0.0f, 0.0f),
                          make_float4(0.0f, 1.0f, 0.0f, 0.0f),
                          make_float4(0.0f, 0.0f, 1.0f, 0.0f),
                          make_float4(-2.0f, 0.0f, 0.0f, 1.0f));
        const device_svm::TransformState transforms{
            identity, identity, object_to_world, world_to_object};
        const ProbeKernelGlobals kernel_globals;
        device_svm::ShaderData shader_data{position,
                                           make_float3(0.0f, 0.0f, 1.0f),
                                           make_float3(0.0f, 0.0f, 1.0f),
                                           make_float3(0.0f, 0.0f, -1.0f),
                                           primitive_type,
                                           0u,
                                           0u,
                                           object_flags,
                                           primitive_id,
                                           0.2f,
                                           0.3f,
                                           0u,
                                           0.5f,
                                           4.0f,
                                           0.2f,
                                           identity,
                                           identity};
        const device_svm::PathState path_state{
            device_svm::path_ray_visibility_camera, 0u};
        device_svm::EvaluationResult result;
        const auto node_features =
            device_svm::kernel_feature_node_emission |
            (bump_feature_enabled ? device_svm::kernel_feature_node_bump : 0u);
        device_svm::eval_nodes(kernel_globals, words, SHADER_TYPE_SURFACE,
                               device_svm::kernel_feature_hair |
                                   device_svm::kernel_feature_object_motion,
                               node_features, node_types, transforms,
                               shader_data, path_state, result);
        floating_output.write(
            index, make_float4(shader_data.closure_emission_background,
                               result.closure_weight.x));
        integer_output.write(index,
                             make_uint4(result.status, result.final_offset,
                                        shader_data.flag, 0u));
      }};
}

void run(Device &device, Stream &stream, std::span<const std::uint32_t> words,
         std::array<bool, NODE_NUM> node_types, bool bump_feature_enabled,
         std::array<luisa::float4, probe_count> &floating,
         std::array<luisa::uint4, probe_count> &integer) {
  const auto kernel = make_probe_kernel(node_types, bump_feature_enabled);
  auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto floating_buffer = device.create_buffer<luisa::float4>(floating.size());
  auto integer_buffer = device.create_buffer<luisa::uint4>(integer.size());
  stream << word_buffer.copy_from(words)
         << shader(word_buffer, floating_buffer, integer_buffer)
                .dispatch(probe_count)
         << floating_buffer.copy_to(luisa::span{floating})
         << integer_buffer.copy_to(luisa::span{integer}) << synchronize();
}

[[nodiscard]] bool test_object_none_normal_transforms(Device &device,
                                                      Stream &stream) {
  const auto kernel = Kernel1D<Buffer<luisa::float4>>{
      [](BufferFloat4 output) noexcept {
        const auto identity = make_float4x4(1.0f);
        const auto object_to_world = make_float4x4(
            make_float4(2.0f, 0.0f, 0.0f, 0.0f),
            make_float4(0.0f, 3.0f, 0.0f, 0.0f),
            make_float4(0.0f, 0.0f, 4.0f, 0.0f),
            make_float4(0.0f, 0.0f, 0.0f, 1.0f));
        const auto world_to_object = make_float4x4(
            make_float4(0.5f, 0.0f, 0.0f, 0.0f),
            make_float4(0.0f, 1.0f / 3.0f, 0.0f, 0.0f),
            make_float4(0.0f, 0.0f, 0.25f, 0.0f),
            make_float4(0.0f, 0.0f, 0.0f, 1.0f));
        const device_svm::TransformState transforms{
            identity, identity, object_to_world, world_to_object};
        device_svm::ShaderData shader_data{
            make_float3(0.0f),
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            device_svm::primitive_triangle,
            0u,
            0u,
            0u,
            0u,
            0.0f,
            0.0f,
            device_svm::object_none,
            0.0f,
            0.0f,
            0.0f,
            identity,
            identity};
        Float3 inverse = make_float3(0.6f, 0.8f, 0.0f);
        Float3 forward = inverse;
        device_svm::detail::object_inverse_normal_transform(
            inverse, transforms, shader_data, true);
        device_svm::detail::object_normal_transform(forward, transforms,
                                                    shader_data, true);
        output.write(0u, make_float4(inverse, 0.0f));
        output.write(1u, make_float4(forward, 0.0f));
      }};
  auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
  auto buffer = device.create_buffer<luisa::float4>(2u);
  std::array<luisa::float4, 2u> output{};
  stream << shader(buffer).dispatch(1u)
         << buffer.copy_to(luisa::span{output}) << synchronize();
  for (const auto &value : output) {
    if (value.x != 0.6f || value.y != 0.8f || value.z != 0.0f) {
      std::cerr << "Cycles OBJECT_NONE normal-transform guard mismatch: ("
                << value.x << ", " << value.y << ", " << value.z << ")\n";
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 3.0e-5f) noexcept {
  return std::abs(actual - expected) <= tolerance;
}

[[nodiscard]] bool
require_factor(const std::array<luisa::float4, probe_count> &floating,
               std::size_t index, float expected, std::string_view label) {
  const auto &value = floating[index];
  if (value.x == expected && value.y == expected && value.z == expected) {
    return true;
  }
  std::cerr << label << " factor mismatch: (" << value.x << ", " << value.y
            << ", " << value.z << ") != " << expected << '\n';
  return false;
}

} // namespace

int main(int argc, char **argv) {
  static constexpr std::array world{
      0x00000001u, 0x00000004u, 0x00000013u, 0x00000014u, 0x0000005au,
      0x3db851ecu, 0x00000000u, 0x00000000u, 0x0000000du, 0x00000000u,
      0x00000100u, 0x00000007u, 0x7fc00001u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u,
  };
  static constexpr std::array pixel{
      0x00000001u, 0x00000004u, 0x00000013u, 0x00000014u, 0x0000005au,
      0x40200000u, 0x00000000u, 0x00000001u, 0x0000000du, 0x00000000u,
      0x00000100u, 0x00000007u, 0x7fc00001u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u,
  };
  static constexpr std::array bump{
      0x00000001u, 0x00000004u, 0x00000021u, 0x00000022u, 0x0000005au,
      0x3e051eb8u, 0x3ebd70a4u, 0x00000000u, 0x0000000bu, 0x01000001u,
      0x00000000u, 0x0000005au, 0x3e051eb8u, 0x3ebd70a4u, 0x00040100u,
      0x0000005au, 0x3e051eb8u, 0x3ebd70a4u, 0x00050200u, 0x00000021u,
      0x3e4ccccdu, 0x3f4ccccdu, 0x3ebd70a4u, 0x00000101u, 0xff060504u,
      0x00000007u, 0x7fc00006u, 0x00000000u, 0x00000000u, 0x3f800000u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u, 0x00000000u,
  };

  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  if (!test_object_none_normal_transforms(device, stream)) {
    return EXIT_FAILURE;
  }
  std::array<luisa::float4, probe_count> floating{};
  std::array<luisa::uint4, probe_count> integer{};
  const auto ended =
      static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended);

  run(device, stream, world, immediate_node_types(), true, floating, integer);
  static constexpr std::array world_expected{1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                             0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
  for (auto index = std::size_t{}; index < world_expected.size(); ++index) {
    if (!require_factor(floating, index, world_expected[index],
                        "Cycles world Wireframe") ||
        integer[index].x != ended || integer[index].y != 19u) {
      return EXIT_FAILURE;
    }
  }

  floating = {};
  integer = {};
  run(device, stream, pixel, immediate_node_types(), true, floating, integer);
  if (!require_factor(floating, 10u, 0.0f, "Cycles pixel Wireframe outside") ||
      !require_factor(floating, 11u, 1.0f, "Cycles pixel Wireframe inside") ||
      integer[10].x != ended || integer[10].y != 19u ||
      integer[11].x != ended || integer[11].y != 19u) {
    return EXIT_FAILURE;
  }

  floating = {};
  integer = {};
  run(device, stream, bump, bump_node_types(), true, floating, integer);
  static constexpr auto expected_bump =
      luisa::float3{0.596504688f, -0.596504688f, 0.536995649f};
  const auto &perturbed = floating[12u];
  const auto &unchanged = floating[13u];
  if (!near(perturbed.x, expected_bump.x) ||
      !near(perturbed.y, expected_bump.y) ||
      !near(perturbed.z, expected_bump.z) || !near(unchanged.x, 0.0f) ||
      !near(unchanged.y, 0.0f) || !near(unchanged.z, 1.0f) ||
      integer[12].x != ended || integer[12].y != 33u ||
      integer[13].x != ended || integer[13].y != 33u) {
    std::cerr << "Cycles Wireframe Bump mismatch on " << backend << ": ("
              << perturbed.x << ", " << perturbed.y << ", " << perturbed.z
              << ")\n";
    return EXIT_FAILURE;
  }

  floating = {};
  integer = {};
  run(device, stream, bump, bump_node_types(), false, floating, integer);
  if (!near(floating[12u].x, 0.0f) || !near(floating[12u].y, 0.0f) ||
      !near(floating[12u].z, 0.0f) || integer[12].x != ended ||
      integer[12].y != 33u) {
    std::cerr << "Cycles disabled Bump feature branch mismatch on " << backend
              << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
