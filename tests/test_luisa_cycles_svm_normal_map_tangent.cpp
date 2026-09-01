#include <psycles/compiler/cycles_svm_bytecode.h>
#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_internal.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace device_svm = psycles::luisa_backend::cycles_svm;
namespace svm_detail = psycles::luisa_backend::cycles_svm::detail;

constexpr auto tangent_attribute_id = std::uint32_t{ATTR_STD_NUM + 1u};
constexpr auto tangent_sign_attribute_id = std::uint32_t{ATTR_STD_NUM + 2u};
constexpr auto uv_float2_attribute_id = std::uint32_t{ATTR_STD_NUM + 3u};
constexpr auto uv_float3_attribute_id = std::uint32_t{ATTR_STD_NUM + 4u};
constexpr auto missing_attribute_id = std::uint32_t{ATTR_STD_NUM + 31u};

constexpr auto tangent_map_offset = std::uint32_t{0u};
constexpr auto sign_map_offset = std::uint32_t{2u};
constexpr auto undisplaced_normal_map_offset = std::uint32_t{4u};
constexpr auto uv_float2_map_offset = std::uint32_t{6u};
constexpr auto uv_float3_map_offset = std::uint32_t{8u};
constexpr auto generated_map_offset = std::uint32_t{10u};
constexpr auto primary_map_end = std::uint32_t{12u};
constexpr auto no_original_map_begin = std::uint32_t{20u};
constexpr auto no_original_tangent_map_offset = std::uint32_t{20u};
constexpr auto no_original_sign_map_offset = std::uint32_t{22u};
constexpr auto no_original_map_end = std::uint32_t{24u};

constexpr auto tangent_data_offset = std::int32_t{0};
constexpr auto undisplaced_normal_data_offset = std::int32_t{1};
constexpr auto uv_float2_data_offset = std::int32_t{0};
constexpr auto uv_float3_data_offset = std::int32_t{3};
constexpr auto generated_data_offset = std::int32_t{6};

constexpr auto untouched = -73.0f;

constexpr auto normal_map_world_opengl = std::uint32_t{0u};
constexpr auto normal_map_world_directx = std::uint32_t{1u};
constexpr auto normal_map_blender_world = std::uint32_t{2u};
constexpr auto normal_map_world_half_strength = std::uint32_t{3u};
constexpr auto normal_map_world_negative_strength = std::uint32_t{4u};
constexpr auto normal_map_world_zero_vector = std::uint32_t{5u};
constexpr auto normal_map_tangent_displaced = std::uint32_t{6u};
constexpr auto normal_map_tangent_missing_attribute = std::uint32_t{7u};
constexpr auto normal_map_tangent_original_attribute = std::uint32_t{8u};
constexpr auto normal_map_tangent_original_triangle_fallback =
    std::uint32_t{9u};
constexpr auto normal_map_tangent_backfacing = std::uint32_t{10u};
constexpr auto normal_map_tangent_ineligible = std::uint32_t{11u};
constexpr auto normal_map_tangent_non_finite_fallback = std::uint32_t{12u};
constexpr auto normal_map_object_opengl = std::uint32_t{13u};
constexpr auto normal_map_blender_object = std::uint32_t{14u};
constexpr auto normal_map_case_count = std::uint32_t{15u};

constexpr auto tangent_uv_float2 = std::uint32_t{0u};
constexpr auto tangent_uv_float3 = std::uint32_t{1u};
constexpr auto tangent_uv_missing = std::uint32_t{2u};
constexpr auto tangent_radial_position_fallback = std::uint32_t{3u};
constexpr auto tangent_radial_generated_attribute = std::uint32_t{4u};
constexpr auto tangent_case_count = std::uint32_t{5u};

constexpr auto tangent_derivative_uv_float2 = std::uint32_t{0u};
constexpr auto tangent_derivative_uv_missing = std::uint32_t{1u};
constexpr auto tangent_derivative_case_count = std::uint32_t{2u};

constexpr auto normal_map_payload_words =
    std::uint32_t{sizeof(SVMNodeNormalMap) / sizeof(std::uint32_t)};
constexpr auto tangent_payload_words =
    std::uint32_t{sizeof(SVMNodeTangent) / sizeof(std::uint32_t)};
constexpr auto normal_map_record_words = normal_map_payload_words + 1u;
constexpr auto tangent_record_words = tangent_payload_words + 1u;

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 3.0e-5f) noexcept {
  return std::isfinite(actual) &&
         std::abs(actual - expected) <=
             tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] bool near(luisa::float3 actual, luisa::float3 expected,
                        float tolerance = 3.0e-5f) noexcept {
  return near(actual.x, expected.x, tolerance) &&
         near(actual.y, expected.y, tolerance) &&
         near(actual.z, expected.z, tolerance);
}

class NormalMapTangentKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
public:
  [[nodiscard]] UInt object_attribute_map_offset(
      Expr<std::uint32_t> object) const noexcept override {
    return select(0u, no_original_map_begin, object == 1u);
  }

  [[nodiscard]] Var<AttributeMap> attribute_map(
      Expr<std::uint32_t> offset) const noexcept override {
    Var<AttributeMap> entry;
    entry.id = static_cast<luisa::ulong>(ATTR_STD_NONE);
    entry.offset = 0;
    entry.element = static_cast<std::uint16_t>(ATTR_ELEMENT_NONE);
    entry.type = static_cast<std::uint8_t>(NODE_ATTR_FLOAT);
    entry.pad = static_cast<std::uint8_t>(0u);

    const auto assign = [&](std::uint64_t id, std::int32_t data_offset,
                            AttributeElement element,
                            NodeAttributeType type) noexcept {
      entry.id = static_cast<luisa::ulong>(id);
      entry.offset = data_offset;
      entry.element = static_cast<std::uint16_t>(element);
      entry.type = static_cast<std::uint8_t>(type);
    };

    $if(offset == tangent_map_offset) {
      assign(tangent_attribute_id, tangent_data_offset, ATTR_ELEMENT_OBJECT,
             NODE_ATTR_FLOAT3);
    }
    $elif(offset == sign_map_offset) {
      assign(tangent_sign_attribute_id, 0, ATTR_ELEMENT_OBJECT,
             NODE_ATTR_FLOAT);
    }
    $elif(offset == undisplaced_normal_map_offset) {
      assign(ATTR_STD_NORMAL_UNDISPLACED, undisplaced_normal_data_offset,
             ATTR_ELEMENT_OBJECT, NODE_ATTR_FLOAT3);
    }
    $elif(offset == uv_float2_map_offset) {
      assign(uv_float2_attribute_id, uv_float2_data_offset,
             ATTR_ELEMENT_VERTEX, NODE_ATTR_FLOAT2);
    }
    $elif(offset == uv_float3_map_offset) {
      assign(uv_float3_attribute_id, uv_float3_data_offset,
             ATTR_ELEMENT_VERTEX, NODE_ATTR_FLOAT3);
    }
    $elif(offset == generated_map_offset) {
      assign(ATTR_STD_GENERATED, generated_data_offset, ATTR_ELEMENT_VERTEX,
             NODE_ATTR_FLOAT3);
    }
    $elif(offset == no_original_tangent_map_offset) {
      assign(tangent_attribute_id, tangent_data_offset, ATTR_ELEMENT_OBJECT,
             NODE_ATTR_FLOAT3);
    }
    $elif(offset == no_original_sign_map_offset) {
      assign(tangent_sign_attribute_id, 0, ATTR_ELEMENT_OBJECT,
             NODE_ATTR_FLOAT);
    }
    $elif((offset == primary_map_end) | (offset == no_original_map_end)) {
      // The default ATTR_STD_NONE/ATTR_ELEMENT_NONE entry terminates Cycles'
      // primitive-specific attribute chain.
    };
    return entry;
  }

  [[nodiscard]] Float attribute_float(
      Expr<std::int32_t>) const noexcept override {
    return 1.0f;
  }

  [[nodiscard]] Float2 attribute_float2(
      Expr<std::int32_t> offset) const noexcept override {
    Float2 value = make_float2(1.0f, 0.0f);
    $if(offset == 1) { value = make_float2(1.0f, 1.0f); }
    $elif(offset == 2) { value = make_float2(2.0f, 0.0f); };
    return value;
  }

  [[nodiscard]] Var<packed_float3> attribute_float3(
      Expr<std::int32_t> offset) const noexcept override {
    Float3 value = make_float3(1.0f, 0.0f, 0.0f);
    $if(offset == undisplaced_normal_data_offset) {
      value = make_float3(0.0f, 1.0f, 0.0f);
    }
    $elif((offset >= uv_float3_data_offset) &
          (offset < uv_float3_data_offset + 3)) {
      value = make_float3(1.0f, 1.0f, 0.0f);
    }
    $elif((offset >= generated_data_offset) &
          (offset < generated_data_offset + 3)) {
      value = make_float3(0.75f, 0.75f, 0.5f);
    };
    Var<packed_float3> packed;
    packed.x = value.x;
    packed.y = value.y;
    packed.z = value.z;
    return packed;
  }

  [[nodiscard]] Var<packed_normal> attribute_normal(
      Expr<std::int32_t>) const noexcept override {
    Var<packed_normal> value;
    value.value = 0x80008000u;
    return value;
  }

  [[nodiscard]] UInt3 triangle_vertex_indices(
      Expr<std::uint32_t>) const noexcept override {
    return make_uint3(0u, 1u, 2u);
  }

  [[nodiscard]] Int object_normal_offset(
      Expr<std::uint32_t>) const noexcept override {
    return 32;
  }
};

[[nodiscard]] device_svm::TransformState identity_transform_state() noexcept {
  const auto identity = make_float4x4(1.0f);
  return {identity, identity, identity, identity};
}

[[nodiscard]] device_svm::TransformState normal_map_transform_state(
    Expr<std::uint32_t> index) noexcept {
  const auto identity = make_float4x4(1.0f);
  Float4x4 object_to_world = identity;
  Float4x4 world_to_object = identity;
  const Bool transformed = (index == normal_map_object_opengl) |
                           (index == normal_map_blender_object);
  $if(transformed) {
    object_to_world = make_float4x4(
        make_float4(2.0f, 0.0f, 0.0f, 0.0f),
        make_float4(0.0f, 0.5f, 0.0f, 0.0f),
        make_float4(0.0f, 0.0f, 1.0f, 0.0f),
        make_float4(0.0f, 0.0f, 0.0f, 1.0f));
    world_to_object = make_float4x4(
        make_float4(0.5f, 0.0f, 0.0f, 0.0f),
        make_float4(0.0f, 2.0f, 0.0f, 0.0f),
        make_float4(0.0f, 0.0f, 1.0f, 0.0f),
        make_float4(0.0f, 0.0f, 0.0f, 1.0f));
  };
  return {identity, identity, object_to_world, world_to_object};
}

[[nodiscard]] device_svm::ShaderData make_shader_data(
    Expr<std::uint32_t> object, Expr<std::uint32_t> shader,
    Expr<std::uint32_t> flag, Expr<luisa::float3> position,
    Expr<luisa::float3> normal, Expr<luisa::float3> geometric_normal) noexcept {
  const auto identity = make_float4x4(1.0f);
  return {position,
          normal,
          geometric_normal,
          make_float3(0.0f, 0.0f, -1.0f),
          device_svm::primitive_triangle,
          shader,
          flag,
          0u,
          0u,
          0.25f,
          0.25f,
          object,
          0.0f,
          1.0f,
          0.0f,
          0.0f,
          0.1f,
          0.2f,
          0.3f,
          0.4f,
          make_float3(1.0f, 0.0f, 0.0f),
          make_float3(0.0f, 1.0f, 0.0f),
          identity,
          identity};
}

[[nodiscard]] BytecodeBuilder normal_map_records() {
  BytecodeBuilder builder;
  const auto add = [&](NodeNormalMapSpace space, bool invert_green,
                       bool use_original_base, std::uint32_t attr,
                       std::uint32_t attr_sign, luisa::float3 color,
                       float strength) {
    static_cast<void>(builder.add_node(
        NODE_NORMAL_MAP,
        SVMNodeNormalMap{
            .space = space,
            .invert_green = invert_green ? 1 : 0,
            .use_original_base = use_original_base ? 1 : 0,
            .attr = static_cast<std::int32_t>(attr),
            .attr_sign = static_cast<std::int32_t>(attr_sign),
            .color = input_float3(color.x, color.y, color.z),
            .strength = input_float(strength),
            .normal_offset = SVMStackOffset{0u},
            ._pad = {0u, 0u, 0u}}));
  };

  add(NODE_NORMAL_MAP_WORLD, false, false, 0u, 0u,
      luisa::float3{1.0f, 0.5f, 0.5f}, 1.0f);
  add(NODE_NORMAL_MAP_WORLD, true, false, 0u, 0u,
      luisa::float3{0.5f, 1.0f, 0.5f}, 1.0f);
  add(NODE_NORMAL_MAP_BLENDER_WORLD, false, false, 0u, 0u,
      luisa::float3{0.5f, 1.0f, 1.0f}, 1.0f);
  add(NODE_NORMAL_MAP_WORLD, false, false, 0u, 0u,
      luisa::float3{1.0f, 0.5f, 0.5f}, 0.5f);
  add(NODE_NORMAL_MAP_WORLD, false, false, 0u, 0u,
      luisa::float3{1.0f, 0.5f, 0.5f}, -2.0f);
  add(NODE_NORMAL_MAP_WORLD, false, false, 0u, 0u,
      luisa::float3{0.5f, 0.5f, 0.5f}, 1.0f);
  add(NODE_NORMAL_MAP_TANGENT, false, false, tangent_attribute_id,
      tangent_sign_attribute_id, luisa::float3{1.0f, 1.0f, 1.0f}, 0.5f);
  add(NODE_NORMAL_MAP_TANGENT, false, false, missing_attribute_id,
      tangent_sign_attribute_id, luisa::float3{1.0f, 0.5f, 0.5f}, 1.0f);
  add(NODE_NORMAL_MAP_TANGENT, false, true, tangent_attribute_id,
      tangent_sign_attribute_id, luisa::float3{0.5f, 0.5f, 1.0f}, 0.5f);
  add(NODE_NORMAL_MAP_TANGENT, false, true, tangent_attribute_id,
      tangent_sign_attribute_id, luisa::float3{1.0f, 0.5f, 0.5f}, 0.5f);
  add(NODE_NORMAL_MAP_TANGENT, false, false, tangent_attribute_id,
      tangent_sign_attribute_id, luisa::float3{1.0f, 0.5f, 0.5f}, 1.0f);
  add(NODE_NORMAL_MAP_TANGENT, false, false, tangent_attribute_id,
      tangent_sign_attribute_id, luisa::float3{1.0f, 0.5f, 0.5f}, 1.0f);
  add(NODE_NORMAL_MAP_TANGENT, false, false, tangent_attribute_id,
      tangent_sign_attribute_id, luisa::float3{0.5f, 0.5f, 0.5f}, 1.0f);
  add(NODE_NORMAL_MAP_OBJECT, false, false, 0u, 0u,
      luisa::float3{1.0f, 1.0f, 0.5f}, 1.0f);
  add(NODE_NORMAL_MAP_BLENDER_OBJECT, false, false, 0u, 0u,
      luisa::float3{0.5f, 1.0f, 1.0f}, 1.0f);
  return builder;
}

[[nodiscard]] BytecodeBuilder tangent_records(bool derivatives) {
  BytecodeBuilder builder;
  const auto add = [&](NodeTangentDirectionType direction,
                       NodeTangentAxis axis, std::uint32_t attr) {
    static_cast<void>(builder.add_node(
        derivatives ? NODE_TANGENT_DERIVATIVE : NODE_TANGENT,
        SVMNodeTangent{.direction_type = direction,
                       .axis = axis,
                       .attr = static_cast<std::int32_t>(attr),
                       .tangent_offset = SVMStackOffset{0u},
                       ._pad = {0u, 0u, 0u}}));
  };
  if (derivatives) {
    add(NODE_TANGENT_UVMAP, NODE_TANGENT_AXIS_Z, uv_float2_attribute_id);
    add(NODE_TANGENT_UVMAP, NODE_TANGENT_AXIS_Z, missing_attribute_id);
  } else {
    add(NODE_TANGENT_UVMAP, NODE_TANGENT_AXIS_Z, uv_float2_attribute_id);
    add(NODE_TANGENT_UVMAP, NODE_TANGENT_AXIS_Z, uv_float3_attribute_id);
    add(NODE_TANGENT_UVMAP, NODE_TANGENT_AXIS_Z, missing_attribute_id);
    add(NODE_TANGENT_RADIAL, NODE_TANGENT_AXIS_Z, missing_attribute_id);
    add(NODE_TANGENT_RADIAL, NODE_TANGENT_AXIS_Z, ATTR_STD_GENERATED);
  }
  return builder;
}

[[nodiscard]] auto normal_map_direct_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferFloat4 output, BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        $if(index < normal_map_case_count) {
          svm_detail::Stack stack;
          svm_detail::stack_store_float3(stack, 0u, make_float3(untouched));
          svm_detail::stack_store_float(stack, 3u, untouched);
          UInt cursor_offset = index * normal_map_record_words + 1u;
          const UInt begin = cursor_offset;
          svm_detail::Cursor cursor{words, cursor_offset};
          const auto transform_state = normal_map_transform_state(index);
          const NormalMapTangentKernelGlobals kernel_globals;

          UInt object = 0u;
          $if(index == normal_map_tangent_original_triangle_fallback) {
            object = 1u;
          }
          $elif(index == normal_map_tangent_ineligible) {
            object = device_svm::object_none;
          };
          const Bool smooth =
              (index == normal_map_tangent_original_attribute) |
              (index == normal_map_tangent_original_triangle_fallback);
          const Bool backfacing = index == normal_map_tangent_backfacing;
          const auto normal = select(make_float3(0.0f, 0.0f, 1.0f),
                                     make_float3(0.0f, 0.0f, -1.0f),
                                     backfacing);
          const auto shader = select(0u, device_svm::shader_smooth_normal,
                                     smooth);
          const auto flag = select(0u, device_svm::shader_data_backfacing,
                                   backfacing);
          const auto shader_data = make_shader_data(
              object, shader, flag, make_float3(0.75f, 0.5f, 0.5f), normal,
              normal);
          svm_detail::node_normal_map(cursor, stack, kernel_globals,
                                      transform_state, shader_data, false);
          output.write(index,
                       make_float4(svm_detail::stack_load_float3(stack, 0u),
                                   svm_detail::stack_load_float(stack, 3u)));
          cursors.write(index, cursor_offset - begin);
        };
      }};
}

[[nodiscard]] auto tangent_plain_direct_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferFloat4 output, BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        $if(index < tangent_case_count) {
          svm_detail::Stack stack;
          svm_detail::stack_store_float(stack, 3u, untouched);
          UInt cursor_offset = index * tangent_record_words + 1u;
          const UInt begin = cursor_offset;
          svm_detail::Cursor cursor{words, cursor_offset};
          const NormalMapTangentKernelGlobals kernel_globals;
          const auto shader_data = make_shader_data(
              0u, 0u, 0u, make_float3(0.75f, 0.5f, 0.5f),
              make_float3(0.0f, 0.0f, 1.0f),
              make_float3(0.0f, 0.0f, 1.0f));
          svm_detail::node_tangent(cursor, stack, kernel_globals,
                                   identity_transform_state(), shader_data,
                                   false, false);
          output.write(index,
                       make_float4(svm_detail::stack_load_float3(stack, 0u),
                                   svm_detail::stack_load_float(stack, 3u)));
          cursors.write(index, cursor_offset - begin);
        };
      }};
}

[[nodiscard]] auto tangent_derivative_direct_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferFloat4 output, BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        $if(index < tangent_derivative_case_count) {
          svm_detail::Stack stack;
          svm_detail::stack_store_float(stack, 9u, untouched);
          UInt cursor_offset = index * tangent_record_words + 1u;
          const UInt begin = cursor_offset;
          svm_detail::Cursor cursor{words, cursor_offset};
          const NormalMapTangentKernelGlobals kernel_globals;
          const auto shader_data = make_shader_data(
              0u, 0u, 0u, make_float3(0.75f, 0.5f, 0.5f),
              make_float3(0.0f, 0.0f, 1.0f),
              make_float3(0.0f, 0.0f, 1.0f));
          svm_detail::node_tangent(cursor, stack, kernel_globals,
                                   identity_transform_state(), shader_data,
                                   true, false);
          const auto sentinel = svm_detail::stack_load_float(stack, 9u);
          output.write(index * 3u,
                       make_float4(svm_detail::stack_load_float3(stack, 0u),
                                   sentinel));
          output.write(index * 3u + 1u,
                       make_float4(svm_detail::stack_load_float3(stack, 3u),
                                   sentinel));
          output.write(index * 3u + 2u,
                       make_float4(svm_detail::stack_load_float3(stack, 6u),
                                   sentinel));
          cursors.write(index, cursor_offset - begin);
        };
      }};
}

[[nodiscard]] bool test_normal_map(Device &device, Stream &stream,
                                   std::string_view backend) {
  const auto records = normal_map_records();
  if (records.size() != normal_map_case_count * normal_map_record_words) {
    std::cerr << "Normal Map record stride regression\n";
    return false;
  }
  auto words = device.create_buffer<std::uint32_t>(records.size());
  auto output = device.create_buffer<luisa::float4>(normal_map_case_count);
  auto cursor = device.create_buffer<std::uint32_t>(normal_map_case_count);
  auto shader = device.compile(
      normal_map_direct_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::array<luisa::float4, normal_map_case_count> actual{};
  std::array<std::uint32_t, normal_map_case_count> cursors{};
  stream << words.copy_from(records.words().data())
         << shader(words, output, cursor).dispatch(normal_map_case_count)
         << output.copy_to(actual.data()) << cursor.copy_to(cursors.data())
         << synchronize();

  constexpr auto inverse_sqrt_two = 0.7071067811865475f;
  constexpr auto inverse_sqrt_six = 0.4082482904638630f;
  const std::array<luisa::float3, normal_map_case_count> expected{
      luisa::float3{1.0f, 0.0f, 0.0f},
      luisa::float3{0.0f, -1.0f, 0.0f},
      luisa::float3{0.0f, -inverse_sqrt_two, -inverse_sqrt_two},
      luisa::float3{inverse_sqrt_two, 0.0f, inverse_sqrt_two},
      luisa::float3{0.0f, 0.0f, 1.0f},
      luisa::float3{0.0f, 0.0f, 1.0f},
      luisa::float3{inverse_sqrt_six, inverse_sqrt_six,
                    2.0f * inverse_sqrt_six},
      luisa::float3{0.0f, 0.0f, 1.0f},
      luisa::float3{0.0f, inverse_sqrt_two, inverse_sqrt_two},
      luisa::float3{inverse_sqrt_two, 0.0f, inverse_sqrt_two},
      luisa::float3{-1.0f, 0.0f, 0.0f},
      luisa::float3{0.0f, 0.0f, 1.0f},
      luisa::float3{0.0f, 0.0f, 1.0f},
      luisa::float3{0.24253562503633297f, 0.9701425001453319f, 0.0f},
      luisa::float3{0.0f, -0.8944271909999159f,
                    -0.4472135954999579f}};

  auto valid = true;
  for (auto index = std::size_t{}; index < expected.size(); ++index) {
    valid &= near(actual[index].xyz(), expected[index]) &&
             actual[index].w == untouched &&
             cursors[index] == normal_map_payload_words;
  }
  if (!valid) {
    std::cerr << "Cycles Normal Map handler mismatch on " << backend << '\n';
    for (auto index = std::size_t{}; index < expected.size(); ++index) {
      std::cerr << index << ": (" << actual[index].x << ", "
                << actual[index].y << ", " << actual[index].z
                << "), cursor=" << cursors[index] << '\n';
    }
  }
  return valid;
}

[[nodiscard]] bool test_tangent_plain(Device &device, Stream &stream,
                                      std::string_view backend) {
  const auto records = tangent_records(false);
  auto words = device.create_buffer<std::uint32_t>(records.size());
  auto output = device.create_buffer<luisa::float4>(tangent_case_count);
  auto cursor = device.create_buffer<std::uint32_t>(tangent_case_count);
  auto shader = device.compile(
      tangent_plain_direct_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::array<luisa::float4, tangent_case_count> actual{};
  std::array<std::uint32_t, tangent_case_count> cursors{};
  stream << words.copy_from(records.words().data())
         << shader(words, output, cursor).dispatch(tangent_case_count)
         << output.copy_to(actual.data()) << cursor.copy_to(cursors.data())
         << synchronize();

  constexpr auto inverse_sqrt_two = 0.7071067811865475f;
  const std::array<luisa::float3, tangent_case_count> expected{
      luisa::float3{0.9805806756909202f, 0.1961161351381840f, 0.0f},
      luisa::float3{inverse_sqrt_two, inverse_sqrt_two, 0.0f},
      luisa::float3{0.0f, 0.0f, 0.0f},
      luisa::float3{0.0f, 1.0f, 0.0f},
      luisa::float3{-inverse_sqrt_two, inverse_sqrt_two, 0.0f}};
  auto valid = records.size() == tangent_case_count * tangent_record_words;
  for (auto index = std::size_t{}; index < expected.size(); ++index) {
    valid &= near(actual[index].xyz(), expected[index]) &&
             actual[index].w == untouched &&
             cursors[index] == tangent_payload_words;
  }
  if (!valid) {
    std::cerr << "Cycles Tangent handler mismatch on " << backend << '\n';
    for (auto index = std::size_t{}; index < expected.size(); ++index) {
      std::cerr << index << ": (" << actual[index].x << ", "
                << actual[index].y << ", " << actual[index].z
                << "), cursor=" << cursors[index] << '\n';
    }
  }
  return valid;
}

[[nodiscard]] bool test_tangent_derivative(Device &device, Stream &stream,
                                           std::string_view backend) {
  const auto records = tangent_records(true);
  auto words = device.create_buffer<std::uint32_t>(records.size());
  auto output = device.create_buffer<luisa::float4>(
      tangent_derivative_case_count * 3u);
  auto cursor =
      device.create_buffer<std::uint32_t>(tangent_derivative_case_count);
  auto shader = device.compile(
      tangent_derivative_direct_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::array<luisa::float4, tangent_derivative_case_count * 3u> actual{};
  std::array<std::uint32_t, tangent_derivative_case_count> cursors{};
  stream << words.copy_from(records.words().data())
         << shader(words, output, cursor).dispatch(tangent_derivative_case_count)
         << output.copy_to(actual.data()) << cursor.copy_to(cursors.data())
         << synchronize();

  const std::array<luisa::float3, tangent_derivative_case_count * 3u> expected{
      luisa::float3{0.9805806756909202f, 0.1961161351381840f, 0.0f},
      luisa::float3{-0.006034342, 0.030171711, 0.0f},
      luisa::float3{-0.018103026, 0.090515139, 0.0f},
      luisa::float3{0.0f, 0.0f, 0.0f},
      luisa::float3{0.0f, 0.0f, 0.0f},
      luisa::float3{0.0f, 0.0f, 0.0f}};
  auto valid = records.size() ==
               tangent_derivative_case_count * tangent_record_words;
  for (auto index = std::size_t{}; index < expected.size(); ++index) {
    valid &= near(actual[index].xyz(), expected[index], 8.0e-5f) &&
             actual[index].w == untouched;
  }
  for (const auto value : cursors) {
    valid &= value == tangent_payload_words;
  }
  if (!valid) {
    std::cerr << "Cycles Tangent derivative handler mismatch on " << backend
              << '\n';
    for (auto index = std::size_t{}; index < actual.size(); ++index) {
      std::cerr << index << ": (" << actual[index].x << ", "
                << actual[index].y << ", " << actual[index].z << ")\n";
    }
  }
  return valid;
}

[[nodiscard]] auto interpreter_kernel(
    std::array<bool, NODE_NUM> node_types_used) {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{
      [node_types_used](BufferUInt words, BufferFloat4 output,
                        BufferUInt status) noexcept {
        const NormalMapTangentKernelGlobals kernel_globals;
        auto shader_data = make_shader_data(
            0u, 0u, 0u, make_float3(0.75f, 0.5f, 0.5f),
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, 1.0f));
        const device_svm::PathState path_state{
            device_svm::path_ray_visibility_camera, 0u};
        device_svm::EvaluationResult result;
        device_svm::eval_nodes(
            kernel_globals, words, SHADER_TYPE_SURFACE, 0u,
            device_svm::kernel_feature_node_emission, node_types_used,
            identity_transform_state(), shader_data, path_state, result);
        output.write(0u,
                     make_float4(shader_data.closure_emission_background,
                                 1.0f));
        status.write(0u, result.status);
      }};
}

[[nodiscard]] bool test_interpreter_dispatch(Device &device, Stream &stream,
                                             std::string_view backend) {
  BytecodeBuilder builder;
  const auto jump = builder.add_node(
      NODE_SHADER_JUMP, SVMNodeShaderJump{.offset_surface = 0,
                                          .offset_volume = 0,
                                          .offset_displacement = 0});
  const auto surface = static_cast<std::uint32_t>(builder.size());
  static_cast<void>(builder.add_node(
      NODE_NORMAL_MAP,
      SVMNodeNormalMap{
          .space = NODE_NORMAL_MAP_WORLD,
          .invert_green = 0,
          .use_original_base = 0,
          .attr = 0,
          .attr_sign = 0,
          .color = input_float3(1.0f, 0.5f, 0.5f),
          .strength = input_float(1.0f),
          .normal_offset = SVMStackOffset{0u},
          ._pad = {0u, 0u, 0u}}));
  static_cast<void>(builder.add_node(
      NODE_TANGENT,
      SVMNodeTangent{.direction_type = NODE_TANGENT_RADIAL,
                     .axis = NODE_TANGENT_AXIS_Z,
                     .attr = static_cast<std::int32_t>(missing_attribute_id),
                     .tangent_offset = SVMStackOffset{3u},
                     ._pad = {0u, 0u, 0u}}));
  static_cast<void>(builder.add_node(
      NODE_TANGENT_DERIVATIVE,
      SVMNodeTangent{
          .direction_type = NODE_TANGENT_UVMAP,
          .axis = NODE_TANGENT_AXIS_Z,
          .attr = static_cast<std::int32_t>(uv_float2_attribute_id),
          .tangent_offset = SVMStackOffset{6u},
          ._pad = {0u, 0u, 0u}}));
  static_cast<void>(builder.add_node(
      NODE_EMISSION_WEIGHT,
      SVMNodeEmissionWeight{.color = input_float3(SVMStackOffset{6u}),
                            .strength = input_float(1.0f)}));
  static_cast<void>(builder.add_node(
      NODE_CLOSURE_EMISSION,
      SVMNodeClosureEmission{.mix_weight_offset = SVM_STACK_INVALID,
                             ._pad = {0u, 0u, 0u}}));
  static_cast<void>(builder.add_node(NODE_END));
  const auto volume = static_cast<std::uint32_t>(builder.size());
  static_cast<void>(builder.add_node(NODE_END));
  const auto displacement = static_cast<std::uint32_t>(builder.size());
  static_cast<void>(builder.add_node(NODE_END));
  builder.set_word(jump + 1u, surface);
  builder.set_word(jump + 2u, volume);
  builder.set_word(jump + 3u, displacement);

  auto words = device.create_buffer<std::uint32_t>(builder.size());
  auto output = device.create_buffer<luisa::float4>(1u);
  auto status_buffer = device.create_buffer<std::uint32_t>(1u);
  auto shader = device.compile(
      interpreter_kernel(builder.node_types_used()),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  luisa::float4 actual{};
  std::uint32_t status{};
  stream << words.copy_from(builder.words().data())
         << shader(words, output, status_buffer).dispatch(1u)
         << output.copy_to(&actual) << status_buffer.copy_to(&status)
         << synchronize();
  const auto expected =
      luisa::float3{0.9805806756909202f, 0.1961161351381840f, 0.0f};
  if (!near(actual.xyz(), expected) ||
      status !=
          static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended)) {
    std::cerr << "Cycles Normal Map/Tangent interpreter mismatch on "
              << backend << ": status=" << status << ", value=("
              << actual.x << ", " << actual.y << ", " << actual.z << ")\n";
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  return test_normal_map(device, stream, backend) &&
                 test_tangent_plain(device, stream, backend) &&
                 test_tangent_derivative(device, stream, backend) &&
                 test_interpreter_dispatch(device, stream, backend)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
