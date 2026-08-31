#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_internal.h"

#include <array>
#include <bit>
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
constexpr auto geometry_dual_case_count = 8u;
constexpr auto attribute_float_id = 100u;
constexpr auto attribute_float2_id = 101u;
constexpr auto attribute_float3_id = 102u;
constexpr auto attribute_float4_id = 103u;
constexpr auto attribute_rgba_id = 104u;
constexpr auto attribute_voxel_id = 105u;
constexpr auto attribute_missing_id = 999u;

[[nodiscard]] constexpr std::uint32_t pack_attribute_node(
    std::uint8_t output_offset, NodeAttributeOutputType output_type,
    NodeBumpOffset bump_offset = NODE_BUMP_OFFSET_CENTER,
    bool store_derivatives = false) noexcept {
  return static_cast<std::uint32_t>(output_offset) |
         (static_cast<std::uint32_t>(output_type) << 8u) |
         (static_cast<std::uint32_t>(bump_offset) << 16u) |
         (static_cast<std::uint32_t>(store_derivatives) << 24u);
}
[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 3.0e-5f) noexcept {
  return std::abs(actual - expected) <= tolerance;
}

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

  [[nodiscard]] Float3 primitive_tangent(
      const device_svm::ShaderData &) const noexcept override {
    return make_float3(0.25f, -0.5f, 0.75f);
  }

  [[nodiscard]] device_svm::Dual3 primitive_tangent_derivative(
      const device_svm::ShaderData &) const noexcept override {
    return {.val = make_float3(0.25f, -0.5f, 0.75f),
            .dx = make_float3(0.125f, 0.25f, -0.375f),
            .dy = make_float3(-0.5f, 0.625f, 0.75f)};
  }

  [[nodiscard]] UInt
  object_attribute_map_offset(Expr<std::uint32_t>) const noexcept override {
    return 0u;
  }

  [[nodiscard]] Var<AttributeMap>
  attribute_map(Expr<std::uint32_t> offset) const noexcept override {
    Var<AttributeMap> entry;
    entry.id = static_cast<luisa::ulong>(ATTR_STD_NONE);
    entry.offset = 0;
    entry.element = static_cast<std::uint16_t>(0u);
    entry.type = static_cast<std::uint8_t>(0u);
    entry.pad = static_cast<std::uint8_t>(0u);
    $if(offset == 0u) {
      entry.id = static_cast<luisa::ulong>(attribute_float_id);
      entry.element = static_cast<std::uint16_t>(ATTR_ELEMENT_VERTEX);
      entry.type = static_cast<std::uint8_t>(NODE_ATTR_FLOAT);
      entry.offset = 0;
    }
    $elif(offset == 2u) {
      entry.id = static_cast<luisa::ulong>(attribute_float2_id);
      entry.element = static_cast<std::uint16_t>(ATTR_ELEMENT_VERTEX);
      entry.type = static_cast<std::uint8_t>(NODE_ATTR_FLOAT2);
      entry.offset = 0;
    }
    $elif(offset == 4u) {
      entry.id = static_cast<luisa::ulong>(attribute_float3_id);
      entry.element = static_cast<std::uint16_t>(ATTR_ELEMENT_VERTEX);
      entry.type = static_cast<std::uint8_t>(NODE_ATTR_FLOAT3);
      entry.offset = 0;
    }
    $elif(offset == 6u) {
      entry.id = static_cast<luisa::ulong>(attribute_float4_id);
      entry.element = static_cast<std::uint16_t>(ATTR_ELEMENT_VERTEX);
      entry.type = static_cast<std::uint8_t>(NODE_ATTR_FLOAT4);
      entry.offset = 0;
    }
    $elif(offset == 8u) {
      entry.id = static_cast<luisa::ulong>(attribute_rgba_id);
      entry.element = static_cast<std::uint16_t>(ATTR_ELEMENT_VERTEX);
      entry.type = static_cast<std::uint8_t>(NODE_ATTR_RGBA);
      entry.offset = 0;
    }
    $elif(offset == 10u) {
      entry.id = static_cast<luisa::ulong>(ATTR_STD_POINTINESS);
      entry.element = static_cast<std::uint16_t>(ATTR_ELEMENT_VERTEX);
      entry.type = static_cast<std::uint8_t>(NODE_ATTR_FLOAT);
      entry.offset = 3;
    }
    $elif(offset == 12u) {
      entry.id = static_cast<luisa::ulong>(ATTR_STD_RANDOM_PER_ISLAND);
      entry.element = static_cast<std::uint16_t>(ATTR_ELEMENT_FACE);
      entry.type = static_cast<std::uint8_t>(NODE_ATTR_FLOAT);
      entry.offset = 6;
    }
    $elif(offset == 14u) {
      entry.id = static_cast<luisa::ulong>(attribute_voxel_id);
      entry.element = static_cast<std::uint16_t>(ATTR_ELEMENT_VOXEL);
      entry.type = static_cast<std::uint8_t>(NODE_ATTR_FLOAT4);
      entry.offset = 7;
    };
    return entry;
  }

  [[nodiscard]] Float
  attribute_float(Expr<std::int32_t> offset) const noexcept override {
    Float value = 0.0f;
    $if(offset == 0) { value = 0.0625f; }
    $elif(offset == 1) { value = 0.375f; }
    $elif(offset == 2) { value = 0.0625f; }
    $elif(offset == 3) { value = 0.05f; }
    $elif(offset == 4) { value = 1.05f; }
    $elif(offset == 5) { value = 0.05f; }
    $elif(offset == 6) { value = 0.75f; };
    return value;
  }

  [[nodiscard]] Float2
  attribute_float2(Expr<std::int32_t> offset) const noexcept override {
    Float2 value = make_float2(0.177f, 0.384f);
    $if(offset == 1) { value = make_float2(0.307f, 0.344f); }
    $elif(offset == 2) { value = make_float2(0.167f, 0.464f); };
    return value;
  }

  [[nodiscard]] Var<packed_float3>
  attribute_float3(Expr<std::int32_t> offset) const noexcept override {
    Float3 source = make_float3(0.174f, 0.387f, 0.74f);
    $if(offset == 1) { source = make_float3(0.334f, 0.317f, 1.04f); }
    $elif(offset == 2) { source = make_float3(0.154f, 0.477f, 0.74f); };
    Var<packed_float3> value;
    value.x = source.x;
    value.y = source.y;
    value.z = source.z;
    return value;
  }

  [[nodiscard]] Float4
  attribute_float4(Expr<std::int32_t> offset) const noexcept override {
    Float4 value = make_float4(0.174f, 0.387f, 0.74f, 0.565f);
    $if(offset == 1) { value = make_float4(0.334f, 0.317f, 1.04f, 0.515f); }
    $elif(offset == 2) { value = make_float4(0.154f, 0.477f, 0.74f, 0.715f); };
    return value;
  }

  [[nodiscard]] Var<uchar4>
  attribute_uchar4(Expr<std::int32_t>) const noexcept override {
    Var<uchar4> value;
    value.x = static_cast<std::uint8_t>(0u);
    value.y = static_cast<std::uint8_t>(0u);
    value.z = static_cast<std::uint8_t>(0u);
    value.w = static_cast<std::uint8_t>(0u);
    return value;
  }

  [[nodiscard]] Var<packed_normal>
  attribute_normal(Expr<std::int32_t>) const noexcept override {
    Var<packed_normal> value;
    value.value = 0u;
    return value;
  }

  [[nodiscard]] UInt3
  triangle_vertex_indices(Expr<std::uint32_t>) const noexcept override {
    return make_uint3(0u, 1u, 2u);
  }

  [[nodiscard]] Int
  object_normal_offset(Expr<std::uint32_t>) const noexcept override {
    return 0;
  }
  [[nodiscard]] UInt
  object_num_geom_steps(Expr<std::uint32_t>) const noexcept override {
    return 2u;
  }
  [[nodiscard]] Int
  object_num_vertices(Expr<std::uint32_t>) const noexcept override {
    return 3;
  }
  [[nodiscard]] Int
  object_num_primitives(Expr<std::uint32_t>) const noexcept override {
    return 1;
  }
  [[nodiscard]] Float3
  object_dupli_generated(Expr<std::uint32_t>) const noexcept override {
    return make_float3(0.0f);
  }
  [[nodiscard]] Float3
  object_dupli_uv(Expr<std::uint32_t>) const noexcept override {
    return make_float3(0.0f);
  }
  [[nodiscard]] UInt camera_type() const noexcept override {
    return device_svm::camera_perspective;
  }
  [[nodiscard]] Float camera_width() const noexcept override { return 1.0f; }
  [[nodiscard]] Float camera_height() const noexcept override { return 1.0f; }
  [[nodiscard]] Float3 camera_world_to_ndc(
      const device_svm::ShaderData &,
      Expr<luisa::float3> position) const noexcept override {
    return position;
  }

  [[nodiscard]] Var<KernelCurve>
  curve(Expr<std::uint32_t>) const noexcept override {
    Var<KernelCurve> value;
    value.shader_id = 0;
    value.first_key = 0;
    value.num_keys = 0;
    value.type = 0;
    return value;
  }

  [[nodiscard]] Bool film_is_rec709() const noexcept override { return true; }

  [[nodiscard]] Float3 film_rec709_to_r() const noexcept override {
    return make_float3(1.0f, 0.0f, 0.0f);
  }

  [[nodiscard]] Float3 film_rec709_to_g() const noexcept override {
    return make_float3(0.0f, 1.0f, 0.0f);
  }

  [[nodiscard]] Float3 film_rec709_to_b() const noexcept override {
    return make_float3(0.0f, 0.0f, 1.0f);
  }

  [[nodiscard]] Float3 object_inverse_position_transform_if_object(
      const device_svm::ShaderData &shader_data,
      Expr<luisa::float3> value) const noexcept override {
    Float3 result = value;
    $if(shader_data.object != device_svm::object_none) {
      result -= make_float3(1.0f, 2.0f, 3.0f);
    };
    return result;
  }

  [[nodiscard]] device_svm::Dual3
  object_inverse_position_transform_if_object_derivative(
      const device_svm::ShaderData &shader_data,
      const device_svm::Dual3 &value) const noexcept override {
    device_svm::Dual3 result{.val = value.val,
                             .dx = value.dx,
                             .dy = value.dy};
    $if(shader_data.object != device_svm::object_none) {
      result.val -= make_float3(1.0f, 2.0f, 3.0f);
    };
    return result;
  }

  [[nodiscard]] Float3 object_inverse_position_transform(
      const device_svm::ShaderData &,
      Expr<luisa::float3> value) const noexcept override {
    return value - make_float3(1.0f, 2.0f, 3.0f);
  }

  [[nodiscard]] Float4 kernel_image_interp_with_udim(
      device_svm::ShaderData &,
      Expr<std::int32_t>,
      const device_svm::Dual2 &) const noexcept override {
    return make_float4(1.0f, 0.0f, 1.0f, 1.0f);
  }

  [[nodiscard]] Float4 kernel_image_interp_3d(
      device_svm::ShaderData &shader_data,
      Expr<std::int32_t> image_texture_id,
      Expr<luisa::float3> position,
      Expr<std::int32_t> interpolation,
      Expr<bool> stochastic) const noexcept override {
    Float4 value = make_float4(0.12f, 0.24f, 0.48f, 0.6f);
    $if(image_texture_id == 8) {
      value = make_float4(position * 0.25f, 0.25f);
    }
    $elif(image_texture_id == 9) {
      value = make_float4(interpolation.cast<float>(), position.x,
                          position.y, 1.0f);
    }
    $elif(image_texture_id == 10) {
      value = make_float4(0.3f, 0.6f, 0.9f, 1.0e-7f);
    };
    $if(stochastic) {
      shader_data.lcg_state += 17u;
      $if(image_texture_id == 7) {
        value = make_float4(0.6f, 0.7f, 0.9f, 0.5f);
      };
    };
    return value;
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
  types[NODE_GEOMETRY_DERIVATIVE] = true;
  types[NODE_SEPARATE_VECTOR] = true;
  types[NODE_SET_BUMP] = true;
  return types;
}

[[nodiscard]] std::array<bool, NODE_NUM>
attribute_node_types(bool use_bump) {
  auto types = immediate_node_types();
  types[NODE_ATTR] = !use_bump;
  types[NODE_ATTR_DERIVATIVE] = use_bump;
  if (use_bump) {
    types[NODE_GEOMETRY] = true;
    types[NODE_SET_BUMP] = true;
  }
  return types;
}

[[nodiscard]] device_svm::ShaderData make_attribute_shader_data(
    Expr<std::uint32_t> primitive_type,
    Expr<std::uint32_t> object) noexcept {
  const auto identity = make_float4x4(1.0f);
  return {make_float3(1.5f, 2.25f, 3.75f),
          make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, -1.0f),
          primitive_type,
          0u,
          0u,
          device_svm::shader_data_object_transform_applied,
          0u,
          0.2f,
          0.3f,
          object,
          0.5f,
          4.0f,
          0.2f,
          0.0f,
          0.1f,
          -0.2f,
          0.3f,
          0.4f,
          make_float3(1.0f, 2.0f, 3.0f),
          make_float3(-1.0f, 0.5f, 2.0f),
          identity,
          identity};
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
                                           0.0f,
                                           0.2f,
                                           0.0f,
                                           0.0f,
                                           0.2f,
                                           make_float3(0.7071067811865475f,
                                                       -0.7071067811865475f,
                                                       0.0f),
                                           make_float3(0.7071067811865475f,
                                                       0.7071067811865475f,
                                                       0.0f),
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

[[nodiscard]] bool test_geometry_dual_lanes(Device &device, Stream &stream) {
  // Payloads are exact SVMNodeGeometry words: geom_type, bump_offset,
  // store_derivatives, out_offset, followed by bump_filter_width. The first
  // six cover every Geometry evaluator arm; the final two cover the Cycles
  // DX/DY first-order value shift while retaining the original dual lanes.
  static constexpr std::array<std::uint32_t,
                              geometry_dual_case_count * 2u>
      payloads{
          0x00010000u, 0x3e947ae1u, // P, CENTER
          0x00010001u, 0x3e947ae1u, // N, CENTER
          0x00010002u, 0x3e947ae1u, // T, CENTER
          0x00010003u, 0x3e947ae1u, // I, CENTER
          0x00010004u, 0x3e947ae1u, // Ng, CENTER
          0x00010005u, 0x3e947ae1u, // uv, CENTER
          0x00010100u, 0x3e947ae1u, // P, DX
          0x00010205u, 0x3e947ae1u, // uv, DY
      };

  const auto kernel =
      Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>>{
          [](BufferUInt words, BufferFloat4 output) noexcept {
            const UInt index = dispatch_x();
            const auto identity = make_float4x4(1.0f);
            device_svm::ShaderData shader_data{
                make_float3(1.0f, 2.0f, 3.0f),
                make_float3(0.1f, 0.2f, 0.3f),
                make_float3(0.0f, 0.0f, 1.0f),
                make_float3(0.0f, 0.0f, -1.0f),
                device_svm::primitive_triangle,
                0u,
                0u,
                device_svm::shader_data_object_transform_applied,
                0u,
                0.2f,
                0.3f,
                0u,
                0.5f,
                4.0f,
                0.2f,
                0.4f,
                0.1f,
                -0.2f,
                0.3f,
                0.4f,
                make_float3(1.0f, 2.0f, 3.0f),
                make_float3(-1.0f, 0.5f, 2.0f),
                identity,
                identity};
            ProbeKernelGlobals kernel_globals;
            device_svm::detail::Stack stack;
            UInt offset = index * 2u;
            device_svm::detail::Cursor cursor{words, offset};
            device_svm::detail::node_geometry(
                cursor, stack, kernel_globals, shader_data, true);
            output.write(index * 3u,
                         make_float4(device_svm::detail::stack_load_float3(
                                         stack, 0u),
                                     0.0f));
            output.write(index * 3u + 1u,
                         make_float4(device_svm::detail::stack_load_float3(
                                         stack, 3u),
                                     0.0f));
            output.write(index * 3u + 2u,
                         make_float4(device_svm::detail::stack_load_float3(
                                         stack, 6u),
                                     0.0f));
          }};
  auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
  auto word_buffer = device.create_buffer<std::uint32_t>(payloads.size());
  std::array<luisa::float4, geometry_dual_case_count * 3u> actual{};
  auto output_buffer = device.create_buffer<luisa::float4>(actual.size());
  stream << word_buffer.copy_from(luisa::span{payloads})
         << shader(word_buffer, output_buffer).dispatch(geometry_dual_case_count)
         << output_buffer.copy_to(luisa::span{actual}) << synchronize();

  constexpr auto s = 0.28284271247461901f;
  static constexpr std::array<luisa::float3,
                              geometry_dual_case_count * 3u>
      expected{
          luisa::float3{1.0f, 2.0f, 3.0f},
          luisa::float3{-0.2f, 0.35f, 0.9f},
          luisa::float3{-0.6f, -0.2f, 0.2f},
          luisa::float3{0.1f, 0.2f, 0.3f},
          luisa::float3{0.0f},
          luisa::float3{0.0f},
          luisa::float3{0.25f, -0.5f, 0.75f},
          luisa::float3{0.125f, 0.25f, -0.375f},
          luisa::float3{-0.5f, 0.625f, 0.75f},
          luisa::float3{0.0f, 0.0f, -1.0f},
          luisa::float3{-s, s, 0.0f},
          luisa::float3{s, s, 0.0f},
          luisa::float3{0.0f, 0.0f, 1.0f},
          luisa::float3{0.0f},
          luisa::float3{0.0f},
          luisa::float3{0.5f, 0.2f, 0.0f},
          luisa::float3{-0.4f, 0.1f, 0.0f},
          luisa::float3{-0.2f, -0.2f, 0.0f},
          luisa::float3{0.942f, 2.1015f, 3.261f},
          luisa::float3{-0.2f, 0.35f, 0.9f},
          luisa::float3{-0.6f, -0.2f, 0.2f},
          luisa::float3{0.442f, 0.142f, 0.0f},
          luisa::float3{-0.4f, 0.1f, 0.0f},
          luisa::float3{-0.2f, -0.2f, 0.0f},
      };
  for (auto i = std::size_t{}; i < actual.size(); ++i) {
    if (!near(actual[i].x, expected[i].x) ||
        !near(actual[i].y, expected[i].y) ||
        !near(actual[i].z, expected[i].z)) {
      std::cerr << "Cycles Geometry dual lane " << i << " mismatch: ("
                << actual[i].x << ", " << actual[i].y << ", "
                << actual[i].z << ") != (" << expected[i].x << ", "
                << expected[i].y << ", " << expected[i].z << ")\n";
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool test_attribute_surface_handler(Device &device,
                                                   Stream &stream) {
  constexpr auto case_count = 21u;
  static constexpr std::array<std::uint32_t, case_count * 3u> payloads{
      attribute_float_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT3),
      0u,
      attribute_float_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT),
      0u,
      attribute_float_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT_ALPHA),
      0u,
      attribute_float2_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT3),
      0u,
      attribute_float2_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT),
      0u,
      attribute_float2_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT_ALPHA),
      0u,
      attribute_float3_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT3),
      0u,
      attribute_float3_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT),
      0u,
      attribute_float3_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT_ALPHA),
      0u,
      attribute_float4_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT3),
      0u,
      attribute_float4_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT),
      0u,
      attribute_float4_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT_ALPHA),
      0u,
      attribute_rgba_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT3),
      0u,
      attribute_rgba_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT_ALPHA),
      0u,
      attribute_missing_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT3),
      0u,
      attribute_missing_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT),
      0u,
      attribute_missing_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT_ALPHA),
      0u,
      static_cast<std::uint32_t>(ATTR_STD_UV),
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT3),
      0u,
      static_cast<std::uint32_t>(ATTR_STD_GENERATED),
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT3),
      0u,
      attribute_float3_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT3),
      0u,
      static_cast<std::uint32_t>(ATTR_STD_GENERATED),
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT3),
      0u,
  };
  static constexpr std::array<luisa::float3, case_count> expected{
      luisa::float3{0.125f},
      luisa::float3{0.125f, 0.0f, 0.0f},
      luisa::float3{1.0f, 0.0f, 0.0f},
      luisa::float3{0.2f, 0.4f, 0.0f},
      luisa::float3{0.2f, 0.0f, 0.0f},
      luisa::float3{1.0f, 0.0f, 0.0f},
      luisa::float3{0.2f, 0.4f, 0.8f},
      luisa::float3{1.4f / 3.0f, 0.0f, 0.0f},
      luisa::float3{1.0f, 0.0f, 0.0f},
      luisa::float3{0.2f, 0.4f, 0.8f},
      luisa::float3{1.4f / 3.0f, 0.0f, 0.0f},
      luisa::float3{0.6f, 0.0f, 0.0f},
      luisa::float3{0.2f, 0.4f, 0.8f},
      luisa::float3{0.6f, 0.0f, 0.0f},
      luisa::float3{0.0f},
      luisa::float3{0.0f},
      luisa::float3{1.0f, 0.0f, 0.0f},
      luisa::float3{0.5f, 0.2f, 0.0f},
      luisa::float3{0.5f, 0.25f, 0.75f},
      luisa::float3{0.0f},
      luisa::float3{1.5f, 2.25f, 3.75f},
  };

  const auto kernel =
      Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>>{
          [](BufferUInt words, BufferFloat4 output) noexcept {
            const UInt index = dispatch_x();
            UInt primitive_type = device_svm::primitive_triangle;
            UInt object = 0u;
            $if(index == 17u) {
              primitive_type = device_svm::primitive_lamp;
            };
            $if((index == 19u) | (index == 20u)) {
              object = device_svm::object_none;
            };
            auto shader_data =
                make_attribute_shader_data(primitive_type, object);
            ProbeKernelGlobals kernel_globals;
            device_svm::detail::Stack stack;
            for (auto lane = 0u; lane < 9u; ++lane) {
              stack[lane] = 0.0f;
            }
            UInt offset = index * 3u;
            device_svm::detail::Cursor cursor{words, offset};
            device_svm::detail::node_attr_surface(
                cursor, stack, kernel_globals, shader_data);
            output.write(
                index,
                make_float4(device_svm::detail::stack_load_float3(
                                stack, 0u),
                            0.0f));
          }};
  auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
  auto word_buffer = device.create_buffer<std::uint32_t>(payloads.size());
  auto output_buffer = device.create_buffer<luisa::float4>(case_count);
  std::array<luisa::float4, case_count> actual{};
  stream << word_buffer.copy_from(luisa::span{payloads})
         << shader(word_buffer, output_buffer).dispatch(case_count)
         << output_buffer.copy_to(luisa::span{actual}) << synchronize();
  for (auto index = std::size_t{}; index < case_count; ++index) {
    if (!near(actual[index].x, expected[index].x) ||
        !near(actual[index].y, expected[index].y) ||
        !near(actual[index].z, expected[index].z)) {
      std::cerr << "Cycles ATTR surface case " << index << " mismatch: ("
                << actual[index].x << ", " << actual[index].y << ", "
                << actual[index].z << ") != (" << expected[index].x << ", "
                << expected[index].y << ", " << expected[index].z << ")\n";
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool test_attribute_derivative_handler(Device &device,
                                                      Stream &stream) {
  constexpr auto case_count = 15u;
  constexpr auto filter = std::bit_cast<std::uint32_t>(0.29f);
  static constexpr std::array<std::uint32_t, case_count * 3u> payloads{
      attribute_float_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT3,
                          NODE_BUMP_OFFSET_CENTER, true),
      filter,
      attribute_float_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT,
                          NODE_BUMP_OFFSET_CENTER, true),
      filter,
      attribute_float_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT_ALPHA,
                          NODE_BUMP_OFFSET_CENTER, true),
      filter,
      attribute_float2_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT3,
                          NODE_BUMP_OFFSET_CENTER, true),
      filter,
      attribute_float2_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT,
                          NODE_BUMP_OFFSET_CENTER, true),
      filter,
      attribute_float3_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT,
                          NODE_BUMP_OFFSET_CENTER, true),
      filter,
      attribute_float4_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT,
                          NODE_BUMP_OFFSET_CENTER, true),
      filter,
      attribute_float4_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT_ALPHA,
                          NODE_BUMP_OFFSET_CENTER, true),
      filter,
      attribute_rgba_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT3,
                          NODE_BUMP_OFFSET_CENTER, true),
      filter,
      attribute_missing_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT_ALPHA,
                          NODE_BUMP_OFFSET_CENTER, true),
      filter,
      static_cast<std::uint32_t>(ATTR_STD_UV),
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT3,
                          NODE_BUMP_OFFSET_CENTER, true),
      filter,
      static_cast<std::uint32_t>(ATTR_STD_GENERATED),
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT3,
                          NODE_BUMP_OFFSET_CENTER, true),
      filter,
      attribute_float_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT,
                          NODE_BUMP_OFFSET_DX, false),
      filter,
      attribute_float_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT,
                          NODE_BUMP_OFFSET_DY, false),
      filter,
      static_cast<std::uint32_t>(ATTR_STD_GENERATED),
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT3,
                          NODE_BUMP_OFFSET_CENTER, true),
      filter,
  };
  constexpr auto float3_average = 1.4f / 3.0f;
  constexpr auto dx_average = 0.02f;
  constexpr auto dy_average = -0.05f / 3.0f;
  static constexpr std::array<luisa::float3, case_count * 3u> expected{
      luisa::float3{0.125f},
      luisa::float3{0.03125f},
      luisa::float3{-0.0625f},
      luisa::float3{0.125f, 0.03125f, -0.0625f},
      luisa::float3{0.0f},
      luisa::float3{0.0f},
      luisa::float3{1.0f, 0.0f, 0.0f},
      luisa::float3{0.0f},
      luisa::float3{0.0f},
      luisa::float3{0.2f, 0.4f, 0.0f},
      luisa::float3{0.01f, 0.02f, 0.0f},
      luisa::float3{-0.03f, 0.04f, 0.0f},
      luisa::float3{0.2f, 0.01f, -0.03f},
      luisa::float3{0.0f},
      luisa::float3{0.0f},
      luisa::float3{float3_average, dx_average, dy_average},
      luisa::float3{0.0f},
      luisa::float3{0.0f},
      luisa::float3{float3_average, dx_average, dy_average},
      luisa::float3{0.0f},
      luisa::float3{0.0f},
      luisa::float3{0.6f, 0.04f, 0.07f},
      luisa::float3{0.0f},
      luisa::float3{0.0f},
      luisa::float3{0.2f, 0.4f, 0.8f},
      luisa::float3{0.01f, 0.02f, 0.03f},
      luisa::float3{-0.04f, 0.05f, -0.06f},
      luisa::float3{1.0f, 0.0f, 0.0f},
      luisa::float3{0.0f},
      luisa::float3{0.0f},
      luisa::float3{0.5f, 0.2f, 0.0f},
      luisa::float3{-0.4f, 0.1f, 0.0f},
      luisa::float3{-0.2f, -0.2f, 0.0f},
      luisa::float3{0.5f, 0.25f, 0.75f},
      luisa::float3{-0.2f, 0.35f, 0.9f},
      luisa::float3{-0.6f, -0.2f, 0.2f},
      luisa::float3{0.1340625f, 0.0f, 0.0f},
      luisa::float3{0.0f},
      luisa::float3{0.0f},
      luisa::float3{0.106875f, 0.0f, 0.0f},
      luisa::float3{0.0f},
      luisa::float3{0.0f},
      luisa::float3{1.5f, 2.25f, 3.75f},
      luisa::float3{-0.2f, 0.35f, 0.9f},
      luisa::float3{-0.6f, -0.2f, 0.2f},
  };

  const auto kernel =
      Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>>{
          [](BufferUInt words, BufferFloat4 output) noexcept {
            const UInt index = dispatch_x();
            UInt primitive_type = device_svm::primitive_triangle;
            $if(index == 10u) {
              primitive_type = device_svm::primitive_lamp;
            };
            UInt object = 0u;
            $if(index == 14u) { object = device_svm::object_none; };
            auto shader_data =
                make_attribute_shader_data(primitive_type, object);
            ProbeKernelGlobals kernel_globals;
            device_svm::detail::Stack stack;
            for (auto lane = 0u; lane < 9u; ++lane) {
              stack[lane] = 0.0f;
            }
            UInt offset = index * 3u;
            device_svm::detail::Cursor cursor{words, offset};
            device_svm::detail::node_attr_derivative(
                cursor, stack, kernel_globals, shader_data);
            output.write(
                index * 3u,
                make_float4(device_svm::detail::stack_load_float3(
                                stack, 0u),
                            0.0f));
            output.write(
                index * 3u + 1u,
                make_float4(device_svm::detail::stack_load_float3(
                                stack, 3u),
                            0.0f));
            output.write(
                index * 3u + 2u,
                make_float4(device_svm::detail::stack_load_float3(
                                stack, 6u),
                            0.0f));
          }};
  auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
  auto word_buffer = device.create_buffer<std::uint32_t>(payloads.size());
  auto output_buffer = device.create_buffer<luisa::float4>(expected.size());
  std::array<luisa::float4, expected.size()> actual{};
  stream << word_buffer.copy_from(luisa::span{payloads})
         << shader(word_buffer, output_buffer).dispatch(case_count)
         << output_buffer.copy_to(luisa::span{actual}) << synchronize();
  for (auto index = std::size_t{}; index < expected.size(); ++index) {
    if (!near(actual[index].x, expected[index].x) ||
        !near(actual[index].y, expected[index].y) ||
        !near(actual[index].z, expected[index].z)) {
      std::cerr << "Cycles ATTR derivative lane group " << index
                << " mismatch: (" << actual[index].x << ", "
                << actual[index].y << ", " << actual[index].z << ") != ("
                << expected[index].x << ", " << expected[index].y << ", "
                << expected[index].z << ")\n";
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool test_attribute_volume_handler(Device &device,
                                                  Stream &stream) {
  constexpr auto case_count = 4u;
  static constexpr std::array<std::uint32_t, case_count * 3u> payloads{
      attribute_voxel_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT),
      0u,
      attribute_voxel_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT3),
      0u,
      attribute_voxel_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT_ALPHA),
      0u,
      attribute_voxel_id,
      pack_attribute_node(0u, NODE_ATTR_OUTPUT_FLOAT3),
      1u,
  };
  static constexpr std::array<luisa::float3, case_count> expected{
      luisa::float3{1.4f / 3.0f, 0.0f, 0.0f},
      luisa::float3{0.2f, 0.4f, 0.8f},
      luisa::float3{0.6f, 0.0f, 0.0f},
      luisa::float3{1.2f, 1.4f, 1.8f},
  };
  const auto kernel =
      Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>>{
          [](BufferUInt words, BufferFloat4 output) noexcept {
            const UInt index = dispatch_x();
            auto shader_data = make_attribute_shader_data(
                device_svm::primitive_volume, 0u);
            ProbeKernelGlobals kernel_globals;
            device_svm::detail::Stack stack;
            for (auto lane = 0u; lane < 3u; ++lane) {
              stack[lane] = 0.0f;
            }
            UInt offset = index * 3u;
            device_svm::detail::Cursor cursor{words, offset};
            device_svm::detail::node_attr_volume(
                cursor, stack, kernel_globals, shader_data);
            output.write(
                index,
                make_float4(device_svm::detail::stack_load_float3(
                                stack, 0u),
                            0.0f));
          }};
  auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
  auto word_buffer = device.create_buffer<std::uint32_t>(payloads.size());
  auto output_buffer = device.create_buffer<luisa::float4>(case_count);
  std::array<luisa::float4, case_count> actual{};
  stream << word_buffer.copy_from(luisa::span{payloads})
         << shader(word_buffer, output_buffer).dispatch(case_count)
         << output_buffer.copy_to(luisa::span{actual}) << synchronize();
  for (auto index = std::size_t{}; index < case_count; ++index) {
    if (!near(actual[index].x, expected[index].x) ||
        !near(actual[index].y, expected[index].y) ||
        !near(actual[index].z, expected[index].z)) {
      std::cerr << "Cycles ATTR volume case " << index << " mismatch: ("
                << actual[index].x << ", " << actual[index].y << ", "
                << actual[index].z << ") != (" << expected[index].x << ", "
                << expected[index].y << ", " << expected[index].z << ")\n";
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool test_primitive_volume_attributes(Device &device,
                                                    Stream &stream) {
  constexpr auto case_count = 18u;
  const auto kernel =
      Kernel1D<Buffer<luisa::float4>, Buffer<luisa::uint4>>{
          [](BufferFloat4 output, BufferUInt4 state_output) noexcept {
            const UInt index = dispatch_x();
            UInt primitive_type = device_svm::primitive_volume;
            UInt shader_flags = 0u;
            UInt element = static_cast<std::uint32_t>(ATTR_ELEMENT_OBJECT);
            UInt type = static_cast<std::uint32_t>(NODE_ATTR_FLOAT);
            Int offset = 0;
            Bool stochastic = false;
            UInt conversion = 0u;

            $if(index == 1u) {
              element = static_cast<std::uint32_t>(ATTR_ELEMENT_MESH);
              type = static_cast<std::uint32_t>(NODE_ATTR_FLOAT2);
            }
            $elif(index == 2u) {
              type = static_cast<std::uint32_t>(NODE_ATTR_FLOAT3);
            }
            $elif(index == 3u) {
              element = static_cast<std::uint32_t>(ATTR_ELEMENT_MESH);
              type = static_cast<std::uint32_t>(NODE_ATTR_FLOAT4);
            }
            $elif(index == 4u) {
              type = static_cast<std::uint32_t>(NODE_ATTR_RGBA);
            }
            $elif(index == 5u) {
              type = static_cast<std::uint32_t>(NODE_ATTR_MATRIX);
            }
            $elif(index == 6u) {
              element = static_cast<std::uint32_t>(ATTR_ELEMENT_VOXEL);
              offset = 8;
            }
            $elif(index == 7u) {
              element = static_cast<std::uint32_t>(ATTR_ELEMENT_VOXEL);
              offset = 9;
              shader_flags = device_svm::shader_data_volume_cubic;
              stochastic = true;
            }
            $elif(index == 8u) {
              element = static_cast<std::uint32_t>(ATTR_ELEMENT_VOXEL);
              offset = 10;
            }
            $elif(index == 9u) {
              primitive_type = device_svm::primitive_triangle;
              element = static_cast<std::uint32_t>(ATTR_ELEMENT_VOXEL);
              offset = 8;
              conversion = 4u;
            }
            $elif(index == 10u) {
              element = static_cast<std::uint32_t>(ATTR_ELEMENT_VOXEL);
              offset = 7;
              conversion = 1u;
            }
            $elif(index == 11u) {
              element = static_cast<std::uint32_t>(ATTR_ELEMENT_VOXEL);
              offset = 7;
              conversion = 2u;
            }
            $elif(index == 12u) {
              element = static_cast<std::uint32_t>(ATTR_ELEMENT_VOXEL);
              offset = 7;
              conversion = 3u;
            }
            $elif(index == 13u) {
              element = static_cast<std::uint32_t>(ATTR_ELEMENT_VOXEL);
              offset = 7;
              conversion = 4u;
            }
            $elif(index == 14u) {
              element = static_cast<std::uint32_t>(ATTR_ELEMENT_OBJECT |
                                                   ATTR_ELEMENT_VOXEL);
            }
            $elif(index == 15u) {
              element = static_cast<std::uint32_t>(ATTR_ELEMENT_OBJECT |
                                                   ATTR_ELEMENT_VOXEL);
              type = 99u;
              offset = 8;
            }
            $elif(index == 16u) {
              element = static_cast<std::uint32_t>(ATTR_ELEMENT_NONE);
            }
            $elif(index == 17u) {
              element = static_cast<std::uint32_t>(ATTR_ELEMENT_VOXEL);
              offset = 9;
            };

            auto shader_data = make_attribute_shader_data(primitive_type, 0u);
            shader_data.flag = shader_flags;
            shader_data.lcg_state = 23u;
            const device_svm::AttributeDescriptor descriptor{
                .element = element, .type = type, .offset = offset};
            ProbeKernelGlobals kernel_globals;
            Float4 value = make_float4(0.0f);
            $switch(conversion) {
              $case(0u) {
                value = device_svm::volume_attribute_float4(
                    kernel_globals, shader_data, descriptor, stochastic);
              };
              $case(1u) {
                value.x = device_svm::primitive_volume_attribute_float(
                    kernel_globals, shader_data, descriptor, stochastic);
              };
              $case(2u) {
                const Float2 converted =
                    device_svm::primitive_volume_attribute_float2(
                        kernel_globals, shader_data, descriptor, stochastic);
                value = make_float4(converted.x, converted.y, 0.0f, 0.0f);
              };
              $case(3u) {
                const Float3 converted =
                    device_svm::primitive_volume_attribute_float3(
                        kernel_globals, shader_data, descriptor, stochastic);
                value = make_float4(converted, 0.0f);
              };
              $case(4u) {
                value = device_svm::primitive_volume_attribute_float4(
                    kernel_globals, shader_data, descriptor, stochastic);
              };
            };
            output.write(index, value);
            state_output.write(
                index,
                make_uint4(shader_data.lcg_state,
                           device_svm::primitive_is_volume_attribute(
                               shader_data)
                               .cast<std::uint32_t>(),
                           0u, 0u));
          }};
  auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
  auto output_buffer = device.create_buffer<luisa::float4>(case_count);
  auto state_buffer = device.create_buffer<luisa::uint4>(case_count);
  std::array<luisa::float4, case_count> actual{};
  std::array<luisa::uint4, case_count> states{};
  stream << shader(output_buffer, state_buffer).dispatch(case_count)
         << output_buffer.copy_to(luisa::span{actual})
         << state_buffer.copy_to(luisa::span{states}) << synchronize();

  static constexpr std::array<luisa::float4, case_count> expected{
      luisa::float4{0.0625f, 0.0625f, 0.0625f, 1.0f},
      luisa::float4{0.177f, 0.384f, 0.0f, 1.0f},
      luisa::float4{0.174f, 0.387f, 0.74f, 1.0f},
      luisa::float4{0.174f, 0.387f, 0.74f, 0.565f},
      luisa::float4{0.174f, 0.387f, 0.74f, 0.565f},
      luisa::float4{0.0f},
      luisa::float4{0.5f, 0.25f, 0.75f, 0.25f},
      luisa::float4{2.0f, 0.5f, 0.25f, 1.0f},
      luisa::float4{0.3f, 0.6f, 0.9f, 1.0e-7f},
      luisa::float4{0.0f},
      luisa::float4{1.4f / 3.0f, 0.0f, 0.0f, 0.0f},
      luisa::float4{0.2f, 0.4f, 0.0f, 0.0f},
      luisa::float4{0.2f, 0.4f, 0.8f, 0.0f},
      luisa::float4{0.2f, 0.4f, 0.8f, 0.6f},
      luisa::float4{0.0625f, 0.0625f, 0.0625f, 1.0f},
      luisa::float4{0.5f, 0.25f, 0.75f, 0.25f},
      luisa::float4{0.0f},
      luisa::float4{-1.0f, 0.5f, 0.25f, 1.0f},
  };
  for (auto index = std::size_t{}; index < case_count; ++index) {
    const auto &value = actual[index];
    const auto &reference = expected[index];
    const auto expected_lcg = index == 7u ? 40u : 23u;
    const auto expected_volume = index == 9u ? 0u : 1u;
    if (!near(value.x, reference.x) || !near(value.y, reference.y) ||
        !near(value.z, reference.z) || !near(value.w, reference.w) ||
        states[index].x != expected_lcg ||
        states[index].y != expected_volume) {
      std::cerr << "Cycles primitive volume attribute case " << index
                << " mismatch: (" << value.x << ", " << value.y << ", "
                << value.z << ", " << value.w << "), lcg="
                << states[index].x << ", is_volume=" << states[index].y
                << '\n';
      return false;
    }
  }
  return true;
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
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            make_float3(1.0f, 0.0f, 0.0f),
            make_float3(0.0f, 1.0f, 0.0f),
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
  // Shader-local streams dumped by unmodified Cycles 5.2.1 at
  // 9e2066aef7ef. They are consumed verbatim so this executes the same
  // PC/payload sequence as the external Blender oracle.
  static constexpr std::array geometry_position_bump{
      0x00000001u, 0x00000004u, 0x0000004bu, 0x0000004cu, 0x0000000cu,
      0x00000000u, 0x3e947ae1u, 0x0000000bu, 0x03000001u, 0x3e947ae1u,
      0x00000054u, 0x7fc00000u, 0x00000000u, 0x00000000u, 0x00000600u,
      0x00000054u, 0x7fc00000u, 0x00000000u, 0x00000000u, 0x0000ff01u,
      0x00000054u, 0x7fc00000u, 0x00000000u, 0x00000000u, 0x0000ff02u,
      0x0000000cu, 0x00000100u, 0x3e947ae1u, 0x00000054u, 0x7fc00000u,
      0x00000000u, 0x00000000u, 0x00000700u, 0x00000054u, 0x7fc00000u,
      0x00000000u, 0x00000000u, 0x0000ff01u, 0x00000054u, 0x7fc00000u,
      0x00000000u, 0x00000000u, 0x0000ff02u, 0x0000000cu, 0x00000200u,
      0x3e947ae1u, 0x00000054u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x00000800u, 0x00000054u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x0000ff01u, 0x00000054u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x0000ff02u, 0x00000021u, 0x3ed1eb85u, 0x3f3ae148u, 0x3e947ae1u,
      0x06000003u, 0xff000807u, 0x00000007u, 0x7fc00000u, 0x00000000u,
      0x00000000u, 0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u,
      0x00000000u, 0x00000000u,
  };
  static constexpr std::array geometry_parametric_bump{
      0x00000001u, 0x00000004u, 0x0000004bu, 0x0000004cu, 0x0000000bu,
      0x00000001u, 0x3e947ae1u, 0x0000000cu, 0x03000005u, 0x3e947ae1u,
      0x00000054u, 0x7fc00003u, 0x00000000u, 0x00000000u, 0x00000600u,
      0x00000054u, 0x7fc00003u, 0x00000000u, 0x00000000u, 0x0000ff01u,
      0x00000054u, 0x7fc00003u, 0x00000000u, 0x00000000u, 0x0000ff02u,
      0x0000000cu, 0x03000105u, 0x3e947ae1u, 0x00000054u, 0x7fc00003u,
      0x00000000u, 0x00000000u, 0x00000700u, 0x00000054u, 0x7fc00003u,
      0x00000000u, 0x00000000u, 0x0000ff01u, 0x00000054u, 0x7fc00003u,
      0x00000000u, 0x00000000u, 0x0000ff02u, 0x0000000cu, 0x03000205u,
      0x3e947ae1u, 0x00000054u, 0x7fc00003u, 0x00000000u, 0x00000000u,
      0x00000800u, 0x00000054u, 0x7fc00003u, 0x00000000u, 0x00000000u,
      0x0000ff01u, 0x00000054u, 0x7fc00003u, 0x00000000u, 0x00000000u,
      0x0000ff02u, 0x00000021u, 0x3ed1eb85u, 0x3f3ae148u, 0x3e947ae1u,
      0x06000000u, 0xff030807u, 0x00000007u, 0x7fc00003u, 0x00000000u,
      0x00000000u, 0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u,
      0x00000000u, 0x00000000u,
  };
  static constexpr std::array geometry_pointiness_surface{
      0x00000001u, 0x00000004u, 0x00000013u, 0x00000014u, 0x00000015u,
      0x00000020u, 0x00000100u, 0x00000000u, 0x0000000du, 0x00000000u,
      0x00000100u, 0x00000007u, 0x7fc00001u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u,
  };
  static constexpr std::array geometry_random_surface{
      0x00000001u, 0x00000004u, 0x00000013u, 0x00000014u, 0x00000015u,
      0x00000021u, 0x00000100u, 0x00000000u, 0x0000000du, 0x00000000u,
      0x00000100u, 0x00000007u, 0x7fc00001u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u,
  };
  static constexpr std::array geometry_pointiness_bump{
      0x00000001u, 0x00000004u, 0x00000021u, 0x00000022u, 0x0000000bu,
      0x00000001u, 0x3e947ae1u, 0x00000016u, 0x00000020u, 0x00000103u,
      0x3e947ae1u, 0x00000016u, 0x00000020u, 0x00010104u, 0x3e947ae1u,
      0x00000016u, 0x00000020u, 0x00020105u, 0x3e947ae1u, 0x00000021u,
      0x3ed1eb85u, 0x3f3ae148u, 0x3e947ae1u, 0x03000000u, 0xff060504u,
      0x00000007u, 0x7fc00006u, 0x00000000u, 0x00000000u, 0x3f800000u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u, 0x00000000u,
  };
  static constexpr std::array geometry_random_bump{
      0x00000001u, 0x00000004u, 0x00000021u, 0x00000022u, 0x0000000bu,
      0x00000001u, 0x3e947ae1u, 0x00000016u, 0x00000021u, 0x00000103u,
      0x3e947ae1u, 0x00000016u, 0x00000021u, 0x00010104u, 0x3e947ae1u,
      0x00000016u, 0x00000021u, 0x00020105u, 0x3e947ae1u, 0x00000021u,
      0x3ed1eb85u, 0x3f3ae148u, 0x3e947ae1u, 0x03000000u, 0xff060504u,
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
  if (!test_geometry_dual_lanes(device, stream)) {
    return EXIT_FAILURE;
  }
  if (!test_attribute_surface_handler(device, stream) ||
      !test_attribute_derivative_handler(device, stream) ||
      !test_attribute_volume_handler(device, stream) ||
      !test_primitive_volume_attributes(device, stream)) {
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

  floating = {};
  integer = {};
  run(device, stream, geometry_position_bump, bump_node_types(), true,
      floating, integer);
  static constexpr auto expected_position =
      luisa::float3{-0.281100065f, 0.0f, 0.959678471f};
  if (!near(floating[13u].x, expected_position.x) ||
      !near(floating[13u].y, expected_position.y) ||
      !near(floating[13u].z, expected_position.z) ||
      integer[13u].x != ended || integer[13u].y != 75u) {
    std::cerr << "Cycles Geometry Position Bump stream mismatch on " << backend
              << ": (" << floating[13u].x << ", " << floating[13u].y << ", "
              << floating[13u].z << "), status=" << integer[13u].x
              << ", pc=" << integer[13u].y << '\n';
    return EXIT_FAILURE;
  }

  floating = {};
  integer = {};
  run(device, stream, geometry_parametric_bump, bump_node_types(), true,
      floating, integer);
  static constexpr auto expected_parametric =
      luisa::float3{0.376315475f, 0.0f, 0.926491559f};
  if (!near(floating[13u].x, expected_parametric.x) ||
      !near(floating[13u].y, expected_parametric.y) ||
      !near(floating[13u].z, expected_parametric.z) ||
      integer[13u].x != ended || integer[13u].y != 75u) {
    std::cerr << "Cycles Geometry Parametric Bump stream mismatch on "
              << backend << ": (" << floating[13u].x << ", "
              << floating[13u].y << ", " << floating[13u].z
              << "), status=" << integer[13u].x
              << ", pc=" << integer[13u].y << '\n';
    return EXIT_FAILURE;
  }

  floating = {};
  integer = {};
  run(device, stream, geometry_pointiness_surface,
      attribute_node_types(false), false, floating, integer);
  if (!require_factor(floating, 0u, 0.25f,
                      "Cycles Geometry Pointiness surface") ||
      integer[0u].x != ended || integer[0u].y != 19u) {
    return EXIT_FAILURE;
  }

  floating = {};
  integer = {};
  run(device, stream, geometry_random_surface,
      attribute_node_types(false), false, floating, integer);
  if (!require_factor(floating, 0u, 0.75f,
                      "Cycles Geometry Random Per Island surface") ||
      integer[0u].x != ended || integer[0u].y != 19u) {
    return EXIT_FAILURE;
  }

  floating = {};
  integer = {};
  run(device, stream, geometry_pointiness_bump, attribute_node_types(true),
      true, floating, integer);
  // This is the Cycles triangle-attribute derivative obtained from the three
  // source values and ShaderData du/dv below. The former semantic callback
  // supplied unrelated hand-authored derivatives and is intentionally gone.
  static constexpr auto expected_pointiness_bump =
      luisa::float3{-0.198768035f, 0.198768035f, 0.959678471f};
  if (!near(floating[0u].x, expected_pointiness_bump.x) ||
      !near(floating[0u].y, expected_pointiness_bump.y) ||
      !near(floating[0u].z, expected_pointiness_bump.z) ||
      integer[0u].x != ended || integer[0u].y != 33u) {
    std::cerr << "Cycles Geometry Pointiness Bump stream mismatch on "
              << backend << ": (" << floating[0u].x << ", "
              << floating[0u].y << ", " << floating[0u].z
              << "), status=" << integer[0u].x
              << ", pc=" << integer[0u].y << '\n';
    return EXIT_FAILURE;
  }

  floating = {};
  integer = {};
  run(device, stream, geometry_random_bump,
      attribute_node_types(true), true, floating, integer);
  if (!near(floating[0u].x, 0.0f) ||
      !near(floating[0u].y, 0.0f) ||
      !near(floating[0u].z, 1.0f) || integer[0u].x != ended ||
      integer[0u].y != 33u) {
    std::cerr << "Cycles Geometry Random Per Island Bump stream mismatch on "
              << backend << ": (" << floating[0u].x << ", "
              << floating[0u].y << ", " << floating[0u].z
              << "), status=" << integer[0u].x
              << ", pc=" << integer[0u].y << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
