#pragma once

#include <psycles/luisa/cycles_svm.h>

#include <cstdint>

#include <luisa/luisa-compute.h>

namespace psycles::test_support {

namespace device_svm = luisa_backend::cycles_svm;
using namespace luisa::compute;

// Neutral host/JIT service projection for focused SVM node tests. Individual
// fixtures override only the Cycles KernelGlobals services whose observable
// state belongs to the node family under test.
class DefaultCyclesSvmKernelGlobals : public device_svm::KernelGlobals {
public:
  [[nodiscard]] device_svm::TriangleVertices triangle_vertices(
      Expr<std::uint32_t>, Expr<std::uint32_t>) const noexcept override {
    return {.v0 = make_float3(0.0f),
            .v1 = make_float3(1.0f, 0.0f, 0.0f),
            .v2 = make_float3(0.0f, 1.0f, 0.0f)};
  }

  [[nodiscard]] device_svm::TriangleVertices motion_triangle_vertices(
      Expr<std::uint32_t>, Expr<std::uint32_t>,
      Expr<float>) const noexcept override {
    return triangle_vertices(0u, 0u);
  }

  [[nodiscard]] Float3 film_rgb_to_y() const noexcept override {
    return make_float3(0.2126f, 0.7152f, 0.0722f);
  }

  [[nodiscard]] Float3 primitive_tangent(
      const device_svm::ShaderData &) const noexcept override {
    return make_float3(1.0f, 0.0f, 0.0f);
  }

  [[nodiscard]] device_svm::Dual3 primitive_tangent_derivative(
      const device_svm::ShaderData &) const noexcept override {
    return {.val = make_float3(1.0f, 0.0f, 0.0f),
            .dx = make_float3(0.0f),
            .dy = make_float3(0.0f)};
  }

  [[nodiscard]] UInt object_attribute_map_offset(
      Expr<std::uint32_t>) const noexcept override {
    return 0u;
  }

  [[nodiscard]] Var<compiler::cycles_svm::AttributeMap> attribute_map(
      Expr<std::uint32_t>) const noexcept override {
    using namespace compiler::cycles_svm;
    Var<AttributeMap> entry;
    entry.id = static_cast<luisa::ulong>(ATTR_STD_NONE);
    entry.offset = 0;
    entry.element = static_cast<std::uint16_t>(0u);
    entry.type = static_cast<std::uint8_t>(0u);
    entry.pad = static_cast<std::uint8_t>(0u);
    return entry;
  }

  [[nodiscard]] Float attribute_float(
      Expr<std::int32_t>) const noexcept override {
    return 0.0f;
  }

  [[nodiscard]] Float2 attribute_float2(
      Expr<std::int32_t>) const noexcept override {
    return make_float2(0.0f);
  }

  [[nodiscard]] Var<compiler::cycles_svm::packed_float3> attribute_float3(
      Expr<std::int32_t>) const noexcept override {
    Var<compiler::cycles_svm::packed_float3> value;
    value.x = 0.0f;
    value.y = 0.0f;
    value.z = 0.0f;
    return value;
  }

  [[nodiscard]] Float4 attribute_float4(
      Expr<std::int32_t>) const noexcept override {
    return make_float4(0.0f);
  }

  [[nodiscard]] Var<compiler::cycles_svm::uchar4> attribute_uchar4(
      Expr<std::int32_t>) const noexcept override {
    Var<compiler::cycles_svm::uchar4> value;
    value.x = static_cast<std::uint8_t>(0u);
    value.y = static_cast<std::uint8_t>(0u);
    value.z = static_cast<std::uint8_t>(0u);
    value.w = static_cast<std::uint8_t>(0u);
    return value;
  }

  [[nodiscard]] Var<compiler::cycles_svm::packed_normal> attribute_normal(
      Expr<std::int32_t>) const noexcept override {
    Var<compiler::cycles_svm::packed_normal> value;
    value.value = 0u;
    return value;
  }

  [[nodiscard]] UInt3 triangle_vertex_indices(
      Expr<std::uint32_t>) const noexcept override {
    return make_uint3(0u, 1u, 2u);
  }

  [[nodiscard]] Int object_normal_offset(
      Expr<std::uint32_t>) const noexcept override {
    return 0;
  }

  [[nodiscard]] UInt object_num_geom_steps(
      Expr<std::uint32_t>) const noexcept override {
    return 2u;
  }

  [[nodiscard]] Int object_num_vertices(
      Expr<std::uint32_t>) const noexcept override {
    return 3;
  }

  [[nodiscard]] Int object_num_primitives(
      Expr<std::uint32_t>) const noexcept override {
    return 1;
  }

  [[nodiscard]] Float3 object_dupli_generated(
      Expr<std::uint32_t>) const noexcept override {
    return make_float3(0.0f);
  }

  [[nodiscard]] Float3 object_dupli_uv(
      Expr<std::uint32_t>) const noexcept override {
    return make_float3(0.0f);
  }

  [[nodiscard]] UInt camera_type() const noexcept override {
    return device_svm::camera_perspective;
  }

  [[nodiscard]] Float camera_width() const noexcept override {
    return 1.0f;
  }

  [[nodiscard]] Float camera_height() const noexcept override {
    return 1.0f;
  }

  [[nodiscard]] Float3 camera_world_to_ndc(
      const device_svm::ShaderData &,
      Expr<luisa::float3> position) const noexcept override {
    return position;
  }

  [[nodiscard]] Var<compiler::cycles_svm::KernelCurve> curve(
      Expr<std::uint32_t>) const noexcept override {
    Var<compiler::cycles_svm::KernelCurve> value;
    value.shader_id = 0;
    value.first_key = 0;
    value.num_keys = 0;
    value.type = 0;
    return value;
  }

  [[nodiscard]] Bool film_is_rec709() const noexcept override {
    return true;
  }

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
      const device_svm::ShaderData &,
      Expr<luisa::float3> value) const noexcept override {
    return value;
  }

  [[nodiscard]] device_svm::Dual3
  object_inverse_position_transform_if_object_derivative(
      const device_svm::ShaderData &,
      const device_svm::Dual3 &value) const noexcept override {
    return value;
  }

  [[nodiscard]] Float3 object_inverse_position_transform(
      const device_svm::ShaderData &,
      Expr<luisa::float3> value) const noexcept override {
    return value;
  }

  [[nodiscard]] Float4 kernel_image_interp_with_udim(
      device_svm::ShaderData &, Expr<std::int32_t>,
      const device_svm::Dual2 &) const noexcept override {
    return make_float4(1.0f, 0.0f, 1.0f, 1.0f);
  }

  [[nodiscard]] Float4 kernel_image_interp_3d(
      device_svm::ShaderData &, Expr<std::int32_t>,
      Expr<luisa::float3>, Expr<std::int32_t>,
      Expr<bool>) const noexcept override {
    return make_float4(0.0f);
  }
};

} // namespace psycles::test_support
