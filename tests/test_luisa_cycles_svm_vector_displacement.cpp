#include <psycles/luisa/cycles_svm.h>

#include "luisa_cycles_svm_test_kernel_globals.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace device_svm = psycles::luisa_backend::cycles_svm;
using psycles::test_support::DefaultCyclesSvmKernelGlobals;

constexpr auto scenario_count = 8u;
constexpr auto shader_count = 5u;
constexpr auto jump_words = 4u;
constexpr auto tail_words = 13u;
constexpr auto output_offset = 20u;

[[nodiscard]] bool is_near(float actual, float expected,
                        float tolerance = 5.0e-5f) noexcept {
  return std::abs(actual - expected) <= tolerance;
}

[[nodiscard]] bool require_float4(luisa::float4 actual,
                                  luisa::float4 expected,
                                  std::string_view label) {
  if (is_near(actual.x, expected.x) && is_near(actual.y, expected.y) &&
      is_near(actual.z, expected.z) && is_near(actual.w, expected.w)) {
    return true;
  }
  std::cerr << label << " mismatch: (" << actual.x << ", " << actual.y
            << ", " << actual.z << ", " << actual.w << ") != ("
            << expected.x << ", " << expected.y << ", " << expected.z
            << ", " << expected.w << ")\n";
  return false;
}

class VectorDisplacementKernelGlobals final
    : public DefaultCyclesSvmKernelGlobals {
public:
  [[nodiscard]] Var<AttributeMap>
  attribute_map(Expr<std::uint32_t> offset) const noexcept override {
    Var<AttributeMap> entry;
    entry.id = static_cast<luisa::ulong>(ATTR_STD_NONE);
    entry.offset = 0;
    entry.element = static_cast<std::uint16_t>(ATTR_ELEMENT_NONE);
    entry.type = static_cast<std::uint8_t>(NODE_ATTR_FLOAT);
    entry.pad = static_cast<std::uint8_t>(0u);
    $if(offset == 0u) {
      entry.id =
          static_cast<luisa::ulong>(ATTR_STD_UV_TANGENT_UNDISPLACED);
      entry.element = static_cast<std::uint16_t>(ATTR_ELEMENT_OBJECT);
      entry.type = static_cast<std::uint8_t>(NODE_ATTR_FLOAT3);
    }
    $elif(offset == static_cast<std::uint32_t>(ATTR_PRIM_TYPES)) {
      entry.id =
          static_cast<luisa::ulong>(ATTR_STD_UV_TANGENT_SIGN_UNDISPLACED);
      entry.element = static_cast<std::uint16_t>(ATTR_ELEMENT_OBJECT);
      entry.type = static_cast<std::uint8_t>(NODE_ATTR_FLOAT);
    };
    return entry;
  }

  [[nodiscard]] Float
  attribute_float(Expr<std::int32_t>) const noexcept override {
    return -1.0f;
  }

  [[nodiscard]] Var<packed_float3>
  attribute_float3(Expr<std::int32_t>) const noexcept override {
    Var<packed_float3> result;
    result.x = 0.0f;
    result.y = 1.0f;
    result.z = 0.0f;
    return result;
  }
};

[[nodiscard]] Float4x4 static_object_to_world(Bool negative) noexcept {
  const Float scale_x = select(2.0f, -2.0f, negative);
  return make_float4x4(make_float4(scale_x, 0.0f, 0.0f, 0.0f),
                       make_float4(0.0f, 3.0f, 0.0f, 0.0f),
                       make_float4(0.0f, 0.0f, 4.0f, 0.0f),
                       make_float4(10.0f, 20.0f, 30.0f, 1.0f));
}

[[nodiscard]] Float4x4 static_world_to_object(Bool negative) noexcept {
  const Float scale_x = select(0.5f, -0.5f, negative);
  return make_float4x4(
      make_float4(scale_x, 0.0f, 0.0f, 0.0f),
      make_float4(0.0f, 1.0f / 3.0f, 0.0f, 0.0f),
      make_float4(0.0f, 0.0f, 0.25f, 0.0f),
      make_float4(select(-5.0f, 5.0f, negative), -20.0f / 3.0f, -7.5f,
                  1.0f));
}

[[nodiscard]] Float4x4 motion_object_to_world() noexcept {
  return make_float4x4(make_float4(5.0f, 0.0f, 0.0f, 0.0f),
                       make_float4(0.0f, 6.0f, 0.0f, 0.0f),
                       make_float4(0.0f, 0.0f, 7.0f, 0.0f),
                       make_float4(100.0f, 200.0f, 300.0f, 1.0f));
}

[[nodiscard]] Float4x4 motion_world_to_object() noexcept {
  return make_float4x4(make_float4(0.2f, 0.0f, 0.0f, 0.0f),
                       make_float4(0.0f, 1.0f / 6.0f, 0.0f, 0.0f),
                       make_float4(0.0f, 0.0f, 1.0f / 7.0f, 0.0f),
                       make_float4(-20.0f, -100.0f / 3.0f, -300.0f / 7.0f,
                                   1.0f));
}

[[nodiscard]] device_svm::ShaderData
make_shader_data(UInt shader_index, Bool backfacing, Bool motion,
                 Bool parallel_fallback) noexcept {
  UInt flags = 0u;
  $if(backfacing) { flags = device_svm::shader_data_backfacing; };
  UInt object_flags = 0u;
  $if(motion) { object_flags = device_svm::shader_data_object_motion; };
  const Float3 normal =
      select(make_float3(0.0f, 0.0f, 1.0f),
             make_float3(0.0f, 0.0f, -1.0f), backfacing);
  const Float3 dPdu =
      select(make_float3(3.0f, 0.0f, 0.0f),
             make_float3(0.0f, 0.0f, 3.0f), parallel_fallback);
  return device_svm::ShaderData{
      make_float3(1.0f, 2.0f, 3.0f),
      normal,
      normal,
      make_float3(0.0f, 0.0f, -1.0f),
      device_svm::primitive_triangle,
      shader_index,
      flags,
      object_flags,
      0u,
      0.25f,
      0.5f,
      0u,
      0.5f,
      1.0f,
      0.75f,
      0.0f,
      0.1f,
      -0.2f,
      0.3f,
      0.4f,
      dPdu,
      make_float3(0.0f, 3.0f, 0.0f),
      motion_object_to_world(),
      motion_world_to_object()};
}

void append_tail(std::vector<std::uint32_t> &words, NodeNormalMapSpace space,
                 std::uint32_t attr, std::uint32_t attr_sign) {
  words.emplace_back(static_cast<std::uint32_t>(NODE_VECTOR_DISPLACEMENT));
  words.emplace_back(static_cast<std::uint32_t>(space));
  words.emplace_back(std::bit_cast<std::uint32_t>(0.7f));
  words.emplace_back(std::bit_cast<std::uint32_t>(-0.1f));
  words.emplace_back(std::bit_cast<std::uint32_t>(0.4f));
  words.emplace_back(std::bit_cast<std::uint32_t>(0.2f));
  words.emplace_back(std::bit_cast<std::uint32_t>(0.5f));
  words.emplace_back(attr);
  words.emplace_back(attr_sign);
  words.emplace_back(output_offset);
  words.emplace_back(static_cast<std::uint32_t>(NODE_SET_DISPLACEMENT));
  words.emplace_back(output_offset);
  words.emplace_back(static_cast<std::uint32_t>(NODE_END));
}

[[nodiscard]] std::vector<std::uint32_t> make_words() {
  std::vector<std::uint32_t> words(shader_count * jump_words, 0u);
  for (auto shader = 0u; shader < shader_count; ++shader) {
    const auto tail = static_cast<std::uint32_t>(
        shader_count * jump_words + shader * tail_words);
    words[shader * jump_words + 0u] =
        static_cast<std::uint32_t>(NODE_SHADER_JUMP);
    words[shader * jump_words + 1u] = tail;
    words[shader * jump_words + 2u] = tail;
    words[shader * jump_words + 3u] = tail;
    switch (shader) {
    case 0u:
      append_tail(words, NODE_NORMAL_MAP_TANGENT,
                  ATTR_STD_UV_TANGENT_UNDISPLACED,
                  ATTR_STD_UV_TANGENT_SIGN_UNDISPLACED);
      break;
    case 1u:
      append_tail(words, NODE_NORMAL_MAP_TANGENT, 77u, 78u);
      break;
    case 2u:
      append_tail(words, NODE_NORMAL_MAP_TANGENT,
                  ATTR_STD_UV_TANGENT_UNDISPLACED, 78u);
      break;
    case 3u:
      append_tail(words, NODE_NORMAL_MAP_OBJECT, 0u, 0u);
      break;
    case 4u:
      append_tail(words, NODE_NORMAL_MAP_WORLD, 0u, 0u);
      break;
    }
  }
  return words;
}

[[nodiscard]] std::array<bool, NODE_NUM> node_types() {
  std::array<bool, NODE_NUM> result{};
  result[NODE_SHADER_JUMP] = true;
  result[NODE_VECTOR_DISPLACEMENT] = true;
  result[NODE_SET_DISPLACEMENT] = true;
  result[NODE_END] = true;
  return result;
}

[[nodiscard]] bool test_eval_nodes(Device &device, Stream &stream) {
  const auto words = make_words();
  const auto used = node_types();
  const auto kernel =
      Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
               Buffer<luisa::uint4>>{
          [used](BufferUInt bytecode, BufferFloat4 floating,
                 BufferUInt4 integer) noexcept {
            const UInt scenario = dispatch_x();
            UInt shader_index = 0u;
            $if(scenario == 1u) { shader_index = 1u; }
            $elif(scenario == 2u) { shader_index = 2u; }
            $elif(scenario == 3u) { shader_index = 3u; }
            $elif(scenario == 4u) { shader_index = 4u; }
            $elif(scenario == 7u) { shader_index = 1u; };

            const Bool negative = scenario == 5u;
            const Bool backfacing = scenario == 5u;
            const Bool motion = scenario == 6u;
            const Bool parallel_fallback = scenario == 7u;
            const auto identity = make_float4x4(1.0f);
            const device_svm::TransformState transforms{
                identity, identity, static_object_to_world(negative),
                static_world_to_object(negative)};
            const VectorDisplacementKernelGlobals kernel_globals;
            const device_svm::PathState path_state{
                device_svm::path_ray_visibility_camera, 0u};

            auto active = make_shader_data(
                shader_index, backfacing, motion, parallel_fallback);
            device_svm::EvaluationResult active_result;
            device_svm::eval_nodes(
                kernel_globals, bytecode, SHADER_TYPE_DISPLACEMENT,
                device_svm::kernel_feature_object_motion,
                device_svm::kernel_feature_node_bump, used, transforms, active,
                path_state, active_result);

            auto disabled = make_shader_data(
                shader_index, backfacing, motion, parallel_fallback);
            device_svm::EvaluationResult disabled_result;
            device_svm::eval_nodes(
                kernel_globals, bytecode, SHADER_TYPE_DISPLACEMENT,
                device_svm::kernel_feature_object_motion, 0u, used, transforms,
                disabled, path_state, disabled_result);

            const UInt base = scenario * 2u;
            floating.write(
                base,
                make_float4(active.P - make_float3(1.0f, 2.0f, 3.0f), 0.0f));
            floating.write(
                base + 1u,
                make_float4(disabled.P - make_float3(1.0f, 2.0f, 3.0f),
                            0.0f));
            integer.write(
                scenario,
                make_uint4(active_result.status, active_result.final_offset,
                           disabled_result.status,
                           disabled_result.final_offset));
          }};
  auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
  auto bytecode = device.create_buffer<std::uint32_t>(words.size());
  auto floating = device.create_buffer<luisa::float4>(scenario_count * 2u);
  auto integer = device.create_buffer<luisa::uint4>(scenario_count);
  std::array<luisa::float4, scenario_count * 2u> actual_floating{};
  std::array<luisa::uint4, scenario_count> actual_integer{};
  stream << bytecode.copy_from(luisa::span{words})
         << shader(bytecode, floating, integer).dispatch(scenario_count)
         << floating.copy_to(luisa::span{actual_floating})
         << integer.copy_to(luisa::span{actual_integer}) << synchronize();

  static constexpr std::array expected{
      luisa::float4{0.2f, 0.75f, -0.6f, 0.0f},
      luisa::float4{0.5f, 0.3f, -0.6f, 0.0f},
      luisa::float4{-0.2f, 0.75f, -0.6f, 0.0f},
      luisa::float4{0.5f, -0.45f, 0.4f, 0.0f},
      luisa::float4{0.25f, -0.15f, 0.1f, 0.0f},
      luisa::float4{0.2f, 0.75f, 0.6f, 0.0f},
      luisa::float4{0.5f, 1.5f, -1.05f, 0.0f},
      luisa::float4{0.0f, 0.0f, 0.4f, 0.0f},
  };
  static constexpr std::array program_index{0u, 1u, 2u, 3u, 4u, 0u, 0u, 1u};
  for (auto scenario = std::size_t{}; scenario < scenario_count; ++scenario) {
    if (!require_float4(actual_floating[scenario * 2u], expected[scenario],
                        "active vector displacement") ||
        !require_float4(actual_floating[scenario * 2u + 1u],
                        {0.0f, 0.0f, 0.0f, 0.0f},
                        "feature-disabled vector displacement")) {
      std::cerr << "scenario " << scenario << " failed\n";
      return false;
    }
    const auto final_offset =
        shader_count * jump_words +
        (program_index[scenario] + 1u) * tail_words;
    if (actual_integer[scenario].x !=
            static_cast<std::uint32_t>(
                device_svm::EvaluationStatus::ended) ||
        actual_integer[scenario].y != final_offset ||
        actual_integer[scenario].z !=
            static_cast<std::uint32_t>(
                device_svm::EvaluationStatus::ended) ||
        actual_integer[scenario].w != final_offset) {
      std::cerr << "scenario " << scenario
                << " PC/status mismatch: (" << actual_integer[scenario].x
                << ", " << actual_integer[scenario].y << ", "
                << actual_integer[scenario].z << ", "
                << actual_integer[scenario].w << ")\n";
      return false;
    }
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  if (!test_eval_nodes(device, stream)) {
    return EXIT_FAILURE;
  }
  std::cout << "Luisa Cycles SVM vector-displacement tests passed on "
            << backend << '\n';
  return EXIT_SUCCESS;
}
