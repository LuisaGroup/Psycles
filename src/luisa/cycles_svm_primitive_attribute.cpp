/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <type_traits>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

Float3 decode_packed_normal(Var<packed_normal> packed) noexcept {
  constexpr auto mu = (1u << 16u) - 1u;
  constexpr auto inv_hmu = 2.0f / static_cast<float>(mu);
  const UInt bits = packed.value;
  Float nx = (bits & mu).cast<float>() * inv_hmu - 1.0f;
  Float ny = ((bits >> 16u) & mu).cast<float>() * inv_hmu - 1.0f;
  const Float nz = 1.0f - abs(nx) - abs(ny);
  const Float t = max(-nz, 0.0f);
  nx += copysign(t, -nx);
  ny += copysign(t, -ny);
  return normalize(make_float3(nx, ny, nz));
}

namespace {

template <typename T> [[nodiscard]] Var<T> zero_value() noexcept {
  if constexpr (std::is_same_v<T, float>) {
    return 0.0f;
  } else if constexpr (std::is_same_v<T, luisa::float2>) {
    return make_float2(0.0f);
  } else if constexpr (std::is_same_v<T, luisa::float3>) {
    return make_float3(0.0f);
  } else {
    static_assert(std::is_same_v<T, luisa::float4>);
    return make_float4(0.0f);
  }
}

[[nodiscard]] AttributeDescriptor attribute_not_found() noexcept {
  return {.element = static_cast<std::uint32_t>(ATTR_ELEMENT_NONE),
          .type = static_cast<std::uint32_t>(NODE_ATTR_FLOAT),
          .offset = static_cast<std::int32_t>(ATTR_STD_NOT_FOUND)};
}

[[nodiscard]] Bool find_attr_offset(const KernelGlobals &kernel_globals,
                                    UInt &attr_offset,
                                    Expr<luisa::ulong> id) noexcept {
  auto attr_map = kernel_globals.attribute_map(attr_offset);
  Bool found = true;

  $while(attr_map.id != id) {
    $if(attr_map.id == static_cast<luisa::ulong>(ATTR_STD_NONE)) {
      $if(attr_map.element.cast<std::uint32_t>() == 0u) {
        found = false;
        $break;
      }
      $else { attr_offset = attr_map.offset.cast<std::uint32_t>(); };
    }
    $else { attr_offset += static_cast<std::uint32_t>(ATTR_PRIM_TYPES); };
    attr_map = kernel_globals.attribute_map(attr_offset);
  };

  return found;
}

[[nodiscard]] Float3
packed_float3_value(Var<compiler::cycles_svm::packed_float3> value) noexcept {
  return make_float3(value.x, value.y, value.z);
}

[[nodiscard]] Float color_srgb_to_linear(Expr<float> value) noexcept {
  Float result;
  $if(value < 0.04045f) { result = max(value * (1.0f / 12.92f), 0.0f); }
  $else { result = pow((value + 0.055f) * (1.0f / 1.055f), 2.4f); };
  return result;
}

template <typename T>
[[nodiscard]] Var<T> attribute_data_fetch(const KernelGlobals &kernel_globals,
                                          Expr<std::uint32_t> element,
                                          Expr<std::int32_t> offset) noexcept {
  if constexpr (std::is_same_v<T, float>) {
    return kernel_globals.attribute_float(offset);
  } else if constexpr (std::is_same_v<T, luisa::float2>) {
    return kernel_globals.attribute_float2(offset);
  } else if constexpr (std::is_same_v<T, luisa::float3>) {
    Float3 result =
        packed_float3_value(kernel_globals.attribute_float3(offset));
    $if((element & static_cast<std::uint32_t>(ATTR_ELEMENT_IS_NORMAL)) != 0u) {
      result = decode_packed_normal(kernel_globals.attribute_normal(offset));
    };
    return result;
  } else {
    static_assert(std::is_same_v<T, luisa::float4>);
    Float4 result;
    $if((element & static_cast<std::uint32_t>(ATTR_ELEMENT_IS_BYTE)) != 0u) {
      const auto packed = kernel_globals.attribute_uchar4(offset);
      const Float4 encoded =
          make_float4(packed.x.cast<float>(), packed.y.cast<float>(),
                      packed.z.cast<float>(), packed.w.cast<float>()) *
          (1.0f / 255.0f);
      const Float4 linear_rec709 = make_float4(
          color_srgb_to_linear(encoded.x), color_srgb_to_linear(encoded.y),
          color_srgb_to_linear(encoded.z), encoded.w);
      result = make_float4(
          ::psycles::luisa_backend::cycles_svm::detail::rec709_to_rgb(
              kernel_globals, linear_rec709.xyz()),
          linear_rec709.w);
    }
    $else { result = kernel_globals.attribute_float4(offset); };
    return result;
  }
}

template <typename T> struct DualValue {
  Var<T> val;
  Var<T> dx;
  Var<T> dy;
};

template <typename T> [[nodiscard]] DualValue<T> zero_dual_value() noexcept {
  return {.val = zero_value<T>(), .dx = zero_value<T>(), .dy = zero_value<T>()};
}

template <typename T>
[[nodiscard]] Var<T> triangle_interpolate(Expr<float> u, Expr<float> v,
                                          Expr<T> f0, Expr<T> f1,
                                          Expr<T> f2) noexcept {
  return (1.0f - u - v) * f0 + u * f1 + v * f2;
}

template <typename T>
[[nodiscard]] Var<T> triangle_attribute_dfdx(const Differential &du,
                                             const Differential &dv, Expr<T> f0,
                                             Expr<T> f1, Expr<T> f2) noexcept {
  return du.dx * f1 + dv.dx * f2 - (du.dx + dv.dx) * f0;
}

template <typename T>
[[nodiscard]] Var<T> triangle_attribute_dfdy(const Differential &du,
                                             const Differential &dv, Expr<T> f0,
                                             Expr<T> f1, Expr<T> f2) noexcept {
  return du.dy * f1 + dv.dy * f2 - (du.dy + dv.dy) * f0;
}

template <typename T, bool derivatives>
[[nodiscard]] DualValue<T>
triangle_attribute(const KernelGlobals &kernel_globals,
                   const ShaderData &shader_data,
                   const AttributeDescriptor &descriptor) noexcept {
  auto result = zero_dual_value<T>();
  $if((descriptor.element &
       static_cast<std::uint32_t>(ATTR_ELEMENT_VERTEX | ATTR_ELEMENT_CORNER)) !=
      0u) {
    UInt i0;
    UInt i1;
    UInt i2;
    $if((descriptor.element &
         static_cast<std::uint32_t>(ATTR_ELEMENT_VERTEX)) != 0u) {
      const UInt3 indices =
          kernel_globals.triangle_vertex_indices(shader_data.prim);
      i0 = indices.x;
      i1 = indices.y;
      i2 = indices.z;
    }
    $else {
      const UInt tri = shader_data.prim * 3u;
      i0 = tri;
      i1 = tri + 1u;
      i2 = tri + 2u;
    };

    const auto f0 =
        attribute_data_fetch<T>(kernel_globals, descriptor.element,
                                descriptor.offset + i0.cast<std::int32_t>());
    const auto f1 =
        attribute_data_fetch<T>(kernel_globals, descriptor.element,
                                descriptor.offset + i1.cast<std::int32_t>());
    const auto f2 =
        attribute_data_fetch<T>(kernel_globals, descriptor.element,
                                descriptor.offset + i2.cast<std::int32_t>());
    result.val =
        triangle_interpolate<T>(shader_data.u, shader_data.v, f0, f1, f2);
    if constexpr (derivatives) {
      result.dx = triangle_attribute_dfdx<T>(shader_data.du, shader_data.dv, f0,
                                             f1, f2);
      result.dy = triangle_attribute_dfdy<T>(shader_data.du, shader_data.dv, f0,
                                             f1, f2);
    }
  }
  $elif((descriptor.element & static_cast<std::uint32_t>(ATTR_ELEMENT_FACE)) !=
        0u) {
    result.val = attribute_data_fetch<T>(
        kernel_globals, descriptor.element,
        descriptor.offset + shader_data.prim.cast<std::int32_t>());
  };
  return result;
}

template <typename T>
[[nodiscard]] Var<T> curve_attribute_dfdx(const Differential &du, Expr<T> f0,
                                          Expr<T> f1) noexcept {
  return du.dx * (f1 - f0);
}

template <typename T>
[[nodiscard]] Var<T> curve_attribute_dfdy(const Differential &du, Expr<T> f0,
                                          Expr<T> f1) noexcept {
  return du.dy * (f1 - f0);
}

template <typename T, bool derivatives>
[[nodiscard]] DualValue<T>
curve_attribute(const KernelGlobals &kernel_globals,
                const ShaderData &shader_data,
                const AttributeDescriptor &descriptor) noexcept {
  auto result = zero_dual_value<T>();
  $if((descriptor.element &
       static_cast<std::uint32_t>(ATTR_ELEMENT_CURVE_KEY)) != 0u) {
    const auto curve = kernel_globals.curve(shader_data.prim);
    const Int k0 =
        curve.first_key +
        (shader_data.type >> primitive_num_bits).cast<std::int32_t>();
    const Int k1 = k0 + 1;

    const auto f0 = attribute_data_fetch<T>(
        kernel_globals, descriptor.element, descriptor.offset + k0);
    const auto f1 = attribute_data_fetch<T>(
        kernel_globals, descriptor.element, descriptor.offset + k1);

    result.val = f0 + shader_data.u * (f1 - f0);
    if constexpr (derivatives) {
      result.dx = curve_attribute_dfdx<T>(shader_data.du, f0, f1);
      result.dy = curve_attribute_dfdy<T>(shader_data.du, f0, f1);
    }
  }
  $elif((descriptor.element &
         static_cast<std::uint32_t>(ATTR_ELEMENT_CURVE)) != 0u) {
    result.val = attribute_data_fetch<T>(
        kernel_globals, descriptor.element,
        descriptor.offset + shader_data.prim.cast<std::int32_t>());
  };
  return result;
}

template <typename T>
[[nodiscard]] DualValue<T>
point_attribute(const KernelGlobals &kernel_globals,
                const ShaderData &shader_data,
                const AttributeDescriptor &descriptor) noexcept {
  auto result = zero_dual_value<T>();
  $if((descriptor.element &
       static_cast<std::uint32_t>(ATTR_ELEMENT_VERTEX)) != 0u) {
    result.val = attribute_data_fetch<T>(
        kernel_globals, descriptor.element,
        descriptor.offset + shader_data.prim.cast<std::int32_t>());
  };
  return result;
}

template <typename T, bool derivatives>
[[nodiscard]] DualValue<T> primitive_surface_attribute_impl(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeDescriptor &descriptor) noexcept {
  auto result = zero_dual_value<T>();
  $if((descriptor.element &
       static_cast<std::uint32_t>(ATTR_ELEMENT_OBJECT | ATTR_ELEMENT_MESH)) !=
      0u) {
    result.val = attribute_data_fetch<T>(kernel_globals, descriptor.element,
                                         descriptor.offset);
  }
  $elif((shader_data.type & primitive_triangle) != 0u) {
    result = triangle_attribute<T, derivatives>(kernel_globals, shader_data,
                                                descriptor);
  }
  $elif((shader_data.type & primitive_curve) != 0u) {
    result = curve_attribute<T, derivatives>(kernel_globals, shader_data,
                                             descriptor);
  }
  $elif((shader_data.type & primitive_point) != 0u) {
    result = point_attribute<T>(kernel_globals, shader_data, descriptor);
  };
  return result;
}

} // namespace

AttributeDescriptor find_attribute(const KernelGlobals &kernel_globals,
                                   const ShaderData &shader_data,
                                   Expr<luisa::ulong> id) noexcept {
  auto descriptor = attribute_not_found();
  $if(shader_data.object != object_none) {
    UInt attr_offset =
        kernel_globals.object_attribute_map_offset(shader_data.object);
    $if(find_attr_offset(kernel_globals, attr_offset, id)) {
      const auto attr_map = kernel_globals.attribute_map(attr_offset);
      descriptor.element = attr_map.element.cast<std::uint32_t>();
      const Bool primitive_element_without_primitive =
          (shader_data.prim == primitive_none) &
          ((descriptor.element &
            static_cast<std::uint32_t>(ATTR_ELEMENT_MESH | ATTR_ELEMENT_VOXEL |
                                       ATTR_ELEMENT_OBJECT)) == 0u);
      $if(primitive_element_without_primitive) {
        descriptor = attribute_not_found();
      }
      $else {
        descriptor.offset = select(
            attr_map.offset, static_cast<std::int32_t>(ATTR_STD_NOT_FOUND),
            attr_map.element.cast<std::uint32_t>() ==
                static_cast<std::uint32_t>(ATTR_ELEMENT_NONE));
        descriptor.type = attr_map.type.cast<std::uint32_t>();
      };
    };
  };
  return descriptor;
}

Float primitive_surface_attribute_float(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeDescriptor &descriptor) noexcept {
  return primitive_surface_attribute_impl<float, false>(kernel_globals,
                                                        shader_data, descriptor)
      .val;
}

Dual1 primitive_surface_attribute_float_derivative(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeDescriptor &descriptor) noexcept {
  auto value = primitive_surface_attribute_impl<float, true>(
      kernel_globals, shader_data, descriptor);
  return {.val = value.val, .dx = value.dx, .dy = value.dy};
}

Float2 primitive_surface_attribute_float2(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeDescriptor &descriptor) noexcept {
  return primitive_surface_attribute_impl<luisa::float2, false>(
             kernel_globals, shader_data, descriptor)
      .val;
}

Dual2 primitive_surface_attribute_float2_derivative(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeDescriptor &descriptor) noexcept {
  auto value = primitive_surface_attribute_impl<luisa::float2, true>(
      kernel_globals, shader_data, descriptor);
  return {.val = value.val, .dx = value.dx, .dy = value.dy};
}

Float3 primitive_surface_attribute_float3(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeDescriptor &descriptor) noexcept {
  return primitive_surface_attribute_impl<luisa::float3, false>(
             kernel_globals, shader_data, descriptor)
      .val;
}

Dual3 primitive_surface_attribute_float3_derivative(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeDescriptor &descriptor) noexcept {
  auto value = primitive_surface_attribute_impl<luisa::float3, true>(
      kernel_globals, shader_data, descriptor);
  return {.val = value.val, .dx = value.dx, .dy = value.dy};
}

Float4 primitive_surface_attribute_float4(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeDescriptor &descriptor) noexcept {
  return primitive_surface_attribute_impl<luisa::float4, false>(
             kernel_globals, shader_data, descriptor)
      .val;
}

Dual4 primitive_surface_attribute_float4_derivative(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeDescriptor &descriptor) noexcept {
  auto value = primitive_surface_attribute_impl<luisa::float4, true>(
      kernel_globals, shader_data, descriptor);
  return {.val = value.val, .dx = value.dx, .dy = value.dy};
}

} // namespace psycles::luisa_backend::cycles_svm
