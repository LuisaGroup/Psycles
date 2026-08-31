/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <psycles/luisa/cycles_svm.h>

#include <type_traits>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm {
namespace {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

/* Luisa DSL realization of kernel/geom/volume.h. The only host/JIT service
 * boundary below is kernel_image_interp_3d, matching Cycles' device image
 * resource boundary; attribute conversion remains in this SVM implementation. */

[[nodiscard]] Float average(Expr<luisa::float3> value) noexcept {
  return (value.x + value.y + value.z) * (1.0f / 3.0f);
}

template<typename T>
[[nodiscard]] Var<T> zero_value() noexcept {
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

template<typename T>
[[nodiscard]] Var<T> volume_attribute_value(Expr<luisa::float4> value) noexcept {
  if constexpr (std::is_same_v<T, float>) {
    return average(value.xyz());
  } else if constexpr (std::is_same_v<T, luisa::float2>) {
    return value.xy();
  } else if constexpr (std::is_same_v<T, luisa::float3>) {
    return value.xyz();
  } else {
    static_assert(std::is_same_v<T, luisa::float4>);
    return value;
  }
}

template<typename T>
[[nodiscard]] Var<T> primitive_volume_attribute_impl(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    const AttributeDescriptor &descriptor, Expr<bool> stochastic) noexcept {
  auto result = zero_value<T>();
  $if(primitive_is_volume_attribute(shader_data)) {
    result = volume_attribute_value<T>(volume_attribute_float4(
        kernel_globals, shader_data, descriptor, stochastic));
  };
  return result;
}

} // namespace

Bool primitive_is_volume_attribute(const ShaderData &shader_data) noexcept {
  return shader_data.type == primitive_volume;
}

Float4 volume_attribute_float4(const KernelGlobals &kernel_globals,
                               ShaderData &shader_data,
                               const AttributeDescriptor &descriptor,
                               Expr<bool> stochastic) noexcept {
  Float4 result = make_float4(0.0f);
  Bool found = false;

  $if((descriptor.element &
       static_cast<std::uint32_t>(ATTR_ELEMENT_OBJECT | ATTR_ELEMENT_MESH)) !=
      0u) {
    $switch(descriptor.type) {
      $case(static_cast<std::uint32_t>(NODE_ATTR_FLOAT)) {
        const Float f = kernel_globals.attribute_float(descriptor.offset);
        result = make_float4(f, f, f, 1.0f);
        found = true;
      };
      $case(static_cast<std::uint32_t>(NODE_ATTR_FLOAT2)) {
        const Float2 f = kernel_globals.attribute_float2(descriptor.offset);
        result = make_float4(f.x, f.y, 0.0f, 1.0f);
        found = true;
      };
      $case(static_cast<std::uint32_t>(NODE_ATTR_FLOAT3)) {
        const auto packed = kernel_globals.attribute_float3(descriptor.offset);
        result = make_float4(packed.x, packed.y, packed.z, 1.0f);
        found = true;
      };
      $case(static_cast<std::uint32_t>(NODE_ATTR_FLOAT4)) {
        result = kernel_globals.attribute_float4(descriptor.offset);
        found = true;
      };
      $case(static_cast<std::uint32_t>(NODE_ATTR_RGBA)) {
        result = kernel_globals.attribute_float4(descriptor.offset);
        found = true;
      };
      $case(static_cast<std::uint32_t>(NODE_ATTR_MATRIX)) {
        result = make_float4(0.0f);
        found = true;
      };
    };
  };

  $if(!found & ((descriptor.element &
                 static_cast<std::uint32_t>(ATTR_ELEMENT_VOXEL)) != 0u)) {
    Float3 position = shader_data.P;
    position = kernel_globals.object_inverse_position_transform(shader_data,
                                                                position);
    Int interpolation = static_cast<std::int32_t>(INTERPOLATION_NONE);
    $if((shader_data.flag & shader_data_volume_cubic) != 0u) {
      interpolation = static_cast<std::int32_t>(INTERPOLATION_CUBIC);
    };
    const Float4 value = kernel_globals.kernel_image_interp_3d(
        shader_data, descriptor.offset, position, interpolation, stochastic);
    $if((value.w > 1.0e-6f) & (value.w != 1.0f)) {
      result = make_float4(value.xyz() / value.w, value.w);
    }
    $else { result = value; };
  };

  return result;
}

Float primitive_volume_attribute_float(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    const AttributeDescriptor &descriptor, Expr<bool> stochastic) noexcept {
  return primitive_volume_attribute_impl<float>(kernel_globals, shader_data,
                                                descriptor, stochastic);
}

Float2 primitive_volume_attribute_float2(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    const AttributeDescriptor &descriptor, Expr<bool> stochastic) noexcept {
  return primitive_volume_attribute_impl<luisa::float2>(
      kernel_globals, shader_data, descriptor, stochastic);
}

Float3 primitive_volume_attribute_float3(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    const AttributeDescriptor &descriptor, Expr<bool> stochastic) noexcept {
  return primitive_volume_attribute_impl<luisa::float3>(
      kernel_globals, shader_data, descriptor, stochastic);
}

Float4 primitive_volume_attribute_float4(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    const AttributeDescriptor &descriptor, Expr<bool> stochastic) noexcept {
  return primitive_volume_attribute_impl<luisa::float4>(
      kernel_globals, shader_data, descriptor, stochastic);
}

} // namespace psycles::luisa_backend::cycles_svm
