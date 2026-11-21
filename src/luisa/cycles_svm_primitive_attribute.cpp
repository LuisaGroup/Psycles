/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <cstdio>
#include <memory>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <luisa/ast/expression.h>
#include <luisa/ast/function_builder.h>
#include <luisa/dsl/struct.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

struct SurfaceAttrDual1 {
  float val;
  float dx;
  float dy;
};

struct SurfaceAttrDual2 {
  luisa::float2 val;
  luisa::float2 dx;
  luisa::float2 dy;
};

struct SurfaceAttrDual3 {
  luisa::float3 val;
  luisa::float3 dx;
  luisa::float3 dy;
};

struct SurfaceAttrDual4 {
  luisa::float4 val;
  luisa::float4 dx;
  luisa::float4 dy;
};

} // namespace psycles::luisa_backend::cycles_svm::detail

LUISA_STRUCT(psycles::luisa_backend::cycles_svm::detail::SurfaceAttrDual1,
             val, dx, dy){};
LUISA_STRUCT(psycles::luisa_backend::cycles_svm::detail::SurfaceAttrDual2,
             val, dx, dy){};
LUISA_STRUCT(psycles::luisa_backend::cycles_svm::detail::SurfaceAttrDual3,
             val, dx, dy){};
LUISA_STRUCT(psycles::luisa_backend::cycles_svm::detail::SurfaceAttrDual4,
             val, dx, dy){};

namespace psycles::luisa_backend::cycles_svm {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

Float3 decode_packed_normal(Var<packed_normal> packed) noexcept {
  static auto callable = []() noexcept {
    Callable result{[](Var<packed_normal> p) noexcept {
      constexpr auto mu = (1u << 16u) - 1u;
      constexpr auto inv_hmu = 2.0f / static_cast<float>(mu);
      const UInt bits = p.value;
      Float nx = (bits & mu).cast<float>() * inv_hmu - 1.0f;
      Float ny = ((bits >> 16u) & mu).cast<float>() * inv_hmu - 1.0f;
      const Float nz = 1.0f - abs(nx) - abs(ny);
      const Float t = max(-nz, 0.0f);
      nx += copysign(t, -nx);
      ny += copysign(t, -ny);
      return normalize(make_float3(nx, ny, nz));
    }};
    result.set_name("cycles_decode_packed_normal");
    return result;
  }();
  return callable(packed);
}

namespace {

template <typename T> struct DualPodTraits;
template <> struct DualPodTraits<float> {
  using type = detail::SurfaceAttrDual1;
};
template <> struct DualPodTraits<luisa::float2> {
  using type = detail::SurfaceAttrDual2;
};
template <> struct DualPodTraits<luisa::float3> {
  using type = detail::SurfaceAttrDual3;
};
template <> struct DualPodTraits<luisa::float4> {
  using type = detail::SurfaceAttrDual4;
};
template <typename T> using DualPod = typename DualPodTraits<T>::type;

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
  if (auto fast = kernel_globals.find_attribute_map_offset(attr_offset, id)) {
    attr_offset = fast->y;
    return fast->x != 0u;
  }
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

[[nodiscard]] UInt3 emit_find_attr_from_base_packed(
    BufferVar<AttributeMap> &attr_maps, Expr<std::uint32_t> base_offset,
    Expr<std::uint32_t> prim, Expr<luisa::ulong> target_id) noexcept {
  UInt elem = static_cast<std::uint32_t>(ATTR_ELEMENT_NONE);
  UInt type = static_cast<std::uint32_t>(NODE_ATTR_FLOAT);
  Int offset = static_cast<std::int32_t>(ATTR_STD_NOT_FOUND);
  UInt attr_offset = base_offset;
  auto attr_map = attr_maps.read(attr_offset);
  Bool found = true;
  $while(attr_map.id != target_id) {
    $if(attr_map.id == static_cast<luisa::ulong>(ATTR_STD_NONE)) {
      $if(attr_map.element.cast<std::uint32_t>() == 0u) {
        found = false;
        $break;
      }
      $else { attr_offset = attr_map.offset.cast<std::uint32_t>(); };
    }
    $else { attr_offset += static_cast<std::uint32_t>(ATTR_PRIM_TYPES); };
    attr_map = attr_maps.read(attr_offset);
  };
  $if(found) {
    const UInt found_elem = attr_map.element.cast<std::uint32_t>();
    const Bool primitive_element_without_primitive =
        (prim == primitive_none) &
        ((found_elem &
          static_cast<std::uint32_t>(ATTR_ELEMENT_MESH | ATTR_ELEMENT_VOXEL |
                                     ATTR_ELEMENT_OBJECT)) == 0u);
    $if(!primitive_element_without_primitive) {
      elem = found_elem;
      offset = select(
          attr_map.offset, static_cast<std::int32_t>(ATTR_STD_NOT_FOUND),
          found_elem == static_cast<std::uint32_t>(ATTR_ELEMENT_NONE));
      type = attr_map.type.cast<std::uint32_t>();
    };
  };
  return make_uint3(elem, type, offset.bitcast<std::uint32_t>());
}

[[nodiscard]] Float3
packed_float3_value(Var<compiler::cycles_svm::packed_float3> value) noexcept {
  return make_float3(value.x, value.y, value.z);
}

[[nodiscard]] Float color_srgb_to_linear(Expr<float> value) noexcept {
  // Byte-encoded color attributes are normalized to [0, 1], so the 2-ULP
  // t^2 * (1 + u * P4(u)) approximation covers the full domain without pow().
  const auto t = (value + 0.055f) * (1.0f / 1.055f);
  const auto q = sqrt(sqrt(max(t, 0.0f)));
  const auto u = q - 1.0f;
  const auto p =
      ((((-0.0229220558f * u + 0.0167627893f) * u - 0.0649761409f) * u +
        0.479943097f) *
           u +
       1.59999943f) *
          u +
      1.0f;
  return select(t * t * p, max(value * (1.0f / 12.92f), 0.0f),
                value < 0.04045f);
}

[[nodiscard]] Float4 decode_byte_rgba_linear_rec709(
    Var<compiler::cycles_svm::uchar4> packed) noexcept {
  static auto callable = []() noexcept {
    Callable result{[](Var<compiler::cycles_svm::uchar4> p) noexcept {
      const Float4 encoded =
          make_float4(p.x.cast<float>(), p.y.cast<float>(),
                      p.z.cast<float>(), p.w.cast<float>()) *
          (1.0f / 255.0f);
      return make_float4(color_srgb_to_linear(encoded.x),
                         color_srgb_to_linear(encoded.y),
                         color_srgb_to_linear(encoded.z), encoded.w);
    }};
    result.set_name("cycles_decode_byte_rgba_srgb");
    return result;
  }();
  return callable(packed);
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

struct SurfaceAttrSpec {
  bool may_obj_mesh{false};
  bool may_vertex{false};
  bool may_corner{false};
  bool may_face{false};
  bool has_curves{false};
  bool has_points{false};
  bool may_normal{false};
  bool may_byte{false};
  bool may_float4{true};
  bool is_rec709{true};

  [[nodiscard]] std::uint32_t key() const noexcept {
    std::uint32_t k = 0u;
    if (may_obj_mesh) { k |= 1u << 0u; }
    if (may_vertex) { k |= 1u << 1u; }
    if (may_corner) { k |= 1u << 2u; }
    if (may_face) { k |= 1u << 3u; }
    if (has_curves) { k |= 1u << 4u; }
    if (has_points) { k |= 1u << 5u; }
    if (may_normal) { k |= 1u << 6u; }
    if (may_byte) { k |= 1u << 7u; }
    if (may_float4) { k |= 1u << 8u; }
    if (is_rec709) { k |= 1u << 9u; }
    return k;
  }
};

template <typename T>
[[nodiscard]] SurfaceAttrSpec
make_surface_attr_spec(const SurfaceAttributeSummary &summary,
                       bool is_rec709) noexcept {
  SurfaceAttrSpec spec{};
  spec.may_obj_mesh =
      (summary.element_mask &
       static_cast<std::uint32_t>(ATTR_ELEMENT_OBJECT | ATTR_ELEMENT_MESH)) !=
      0u;
  spec.may_vertex =
      (summary.element_mask &
       static_cast<std::uint32_t>(ATTR_ELEMENT_VERTEX)) != 0u;
  spec.may_corner =
      (summary.element_mask &
       static_cast<std::uint32_t>(ATTR_ELEMENT_CORNER)) != 0u;
  spec.may_face =
      (summary.element_mask & static_cast<std::uint32_t>(ATTR_ELEMENT_FACE)) !=
      0u;
  spec.has_curves = summary.has_curves;
  spec.has_points = summary.has_points;
  if constexpr (std::is_same_v<T, luisa::float3>) {
    spec.may_normal =
        (summary.element_mask &
         static_cast<std::uint32_t>(ATTR_ELEMENT_IS_NORMAL)) != 0u;
  } else if constexpr (std::is_same_v<T, luisa::float4>) {
    spec.may_byte =
        (summary.element_mask &
         static_cast<std::uint32_t>(ATTR_ELEMENT_IS_BYTE)) != 0u;
    spec.may_float4 =
        spec.may_byte
            ? ((summary.type_mask & (1u << NODE_ATTR_FLOAT4)) != 0u)
            : true;
    spec.is_rec709 = spec.may_byte ? is_rec709 : true;
  }
  return spec;
}

template <typename T, bool derivatives, typename Fetchers>
[[nodiscard]] DualValue<T>
emit_triangle_attribute(const Fetchers &fetchers, const SurfaceAttrSpec &spec,
                        Expr<std::uint32_t> prim, Expr<float> u, Expr<float> v,
                        const Differential &du, const Differential &dv,
                        Expr<std::uint32_t> element,
                        Expr<std::int32_t> offset) noexcept {
  auto result = zero_dual_value<T>();
  const auto emit_interp = [&]() noexcept {
    UInt i0;
    UInt i1;
    UInt i2;
    if (spec.may_vertex && spec.may_corner) {
      $if((element & static_cast<std::uint32_t>(ATTR_ELEMENT_VERTEX)) != 0u) {
        const UInt3 indices = fetchers.triangle_vertex_indices(prim);
        i0 = indices.x;
        i1 = indices.y;
        i2 = indices.z;
      }
      $else {
        const UInt tri = prim * 3u;
        i0 = tri;
        i1 = tri + 1u;
        i2 = tri + 2u;
      };
    } else if (spec.may_vertex) {
      const UInt3 indices = fetchers.triangle_vertex_indices(prim);
      i0 = indices.x;
      i1 = indices.y;
      i2 = indices.z;
    } else {
      const UInt tri = prim * 3u;
      i0 = tri;
      i1 = tri + 1u;
      i2 = tri + 2u;
    }

    const auto f0 = fetchers.fetch_attr(element, offset + i0.cast<std::int32_t>());
    const auto f1 = fetchers.fetch_attr(element, offset + i1.cast<std::int32_t>());
    const auto f2 = fetchers.fetch_attr(element, offset + i2.cast<std::int32_t>());
    result.val = triangle_interpolate<T>(u, v, f0, f1, f2);
    if constexpr (derivatives) {
      result.dx = triangle_attribute_dfdx<T>(du, dv, f0, f1, f2);
      result.dy = triangle_attribute_dfdy<T>(du, dv, f0, f1, f2);
    }
  };
  if (spec.may_vertex || spec.may_corner) {
    $if((element & static_cast<std::uint32_t>(ATTR_ELEMENT_VERTEX |
                                              ATTR_ELEMENT_CORNER)) != 0u) {
      emit_interp();
    }
    $else {
      if (spec.may_face) {
        $if((element & static_cast<std::uint32_t>(ATTR_ELEMENT_FACE)) != 0u) {
          result.val =
              fetchers.fetch_attr(element, offset + prim.cast<std::int32_t>());
        };
      }
    };
  } else if (spec.may_face) {
    $if((element & static_cast<std::uint32_t>(ATTR_ELEMENT_FACE)) != 0u) {
      result.val =
          fetchers.fetch_attr(element, offset + prim.cast<std::int32_t>());
    };
  }
  return result;
}

template <typename T, bool derivatives, typename Fetchers>
[[nodiscard]] DualValue<T>
emit_curve_attribute(const Fetchers &fetchers, Expr<std::uint32_t> prim_type,
                     Expr<std::uint32_t> prim, Expr<float> u,
                     const Differential &du, Expr<std::uint32_t> element,
                     Expr<std::int32_t> offset) noexcept {
  auto result = zero_dual_value<T>();
  $if((element & static_cast<std::uint32_t>(ATTR_ELEMENT_CURVE_KEY)) != 0u) {
    const auto curve = fetchers.curve(prim);
    const Int k0 =
        curve.first_key + (prim_type >> primitive_num_bits).cast<std::int32_t>();
    const Int k1 = k0 + 1;

    const auto f0 = fetchers.fetch_attr(element, offset + k0);
    const auto f1 = fetchers.fetch_attr(element, offset + k1);

    result.val = f0 + u * (f1 - f0);
    if constexpr (derivatives) {
      result.dx = curve_attribute_dfdx<T>(du, f0, f1);
      result.dy = curve_attribute_dfdy<T>(du, f0, f1);
    }
  }
  $elif((element & static_cast<std::uint32_t>(ATTR_ELEMENT_CURVE)) != 0u) {
    result.val =
        fetchers.fetch_attr(element, offset + prim.cast<std::int32_t>());
  };
  return result;
}

template <typename T, typename Fetchers>
[[nodiscard]] DualValue<T>
emit_point_attribute(const Fetchers &fetchers, Expr<std::uint32_t> prim,
                     Expr<std::uint32_t> element,
                     Expr<std::int32_t> offset) noexcept {
  auto result = zero_dual_value<T>();
  $if((element & static_cast<std::uint32_t>(ATTR_ELEMENT_VERTEX)) != 0u) {
    result.val =
        fetchers.fetch_attr(element, offset + prim.cast<std::int32_t>());
  };
  return result;
}

template <typename T, bool derivatives, typename Fetchers>
[[nodiscard]] DualValue<T> emit_primitive_surface_attribute_body(
    const Fetchers &fetchers, const SurfaceAttrSpec &spec,
    Expr<std::uint32_t> prim_type, Expr<std::uint32_t> prim, Expr<float> u,
    Expr<float> v, const Differential &du, const Differential &dv,
    Expr<std::uint32_t> element, Expr<std::int32_t> offset) noexcept {
  auto result = zero_dual_value<T>();
  const auto emit_prim_branches = [&]() noexcept {
    $if((prim_type & primitive_triangle) != 0u) {
      result = emit_triangle_attribute<T, derivatives>(
          fetchers, spec, prim, u, v, du, dv, element, offset);
    }
    $else {
      if (spec.has_curves) {
        $if((prim_type & primitive_curve) != 0u) {
          result = emit_curve_attribute<T, derivatives>(
              fetchers, prim_type, prim, u, du, element, offset);
        }
        $else {
          if (spec.has_points) {
            $if((prim_type & primitive_point) != 0u) {
              result = emit_point_attribute<T>(fetchers, prim, element, offset);
            };
          }
        };
      } else if (spec.has_points) {
        $if((prim_type & primitive_point) != 0u) {
          result = emit_point_attribute<T>(fetchers, prim, element, offset);
        };
      }
    };
  };
  if (spec.may_obj_mesh) {
    $if((element &
         static_cast<std::uint32_t>(ATTR_ELEMENT_OBJECT | ATTR_ELEMENT_MESH)) !=
        0u) {
      result.val = fetchers.fetch_attr(element, offset);
    }
    $else { emit_prim_branches(); };
  } else {
    emit_prim_branches();
  }
  return result;
}

template <typename T> struct KernelGlobalsFetchers {
  const KernelGlobals &kernel_globals;
  const SurfaceAttrSpec &spec;

  [[nodiscard]] UInt3
  triangle_vertex_indices(Expr<std::uint32_t> prim) const noexcept {
    return kernel_globals.triangle_vertex_indices(prim);
  }

  [[nodiscard]] Var<KernelCurve>
  curve(Expr<std::uint32_t> prim) const noexcept {
    return kernel_globals.curve(prim);
  }

  [[nodiscard]] Var<T> fetch_attr(Expr<std::uint32_t> element,
                                  Expr<std::int32_t> offset) const noexcept {
    if constexpr (std::is_same_v<T, float>) {
      return kernel_globals.attribute_float(offset);
    } else if constexpr (std::is_same_v<T, luisa::float2>) {
      return kernel_globals.attribute_float2(offset);
    } else if constexpr (std::is_same_v<T, luisa::float3>) {
      Float3 result =
          packed_float3_value(kernel_globals.attribute_float3(offset));
      if (spec.may_normal) {
        $if((element & static_cast<std::uint32_t>(ATTR_ELEMENT_IS_NORMAL)) !=
            0u) {
          result = decode_packed_normal(kernel_globals.attribute_normal(offset));
        };
      }
      return result;
    } else {
      static_assert(std::is_same_v<T, luisa::float4>);
      if (!spec.may_byte) {
        return kernel_globals.attribute_float4(offset);
      }
      const auto fetch_byte_rgba = [&]() noexcept {
        const Float4 linear_rec709 =
            decode_byte_rgba_linear_rec709(kernel_globals.attribute_uchar4(offset));
        return make_float4(
            ::psycles::luisa_backend::cycles_svm::detail::rec709_to_rgb(
                kernel_globals, linear_rec709.xyz()),
            linear_rec709.w);
      };
      if (!spec.may_float4) {
        return fetch_byte_rgba();
      }
      Float4 result;
      $if((element & static_cast<std::uint32_t>(ATTR_ELEMENT_IS_BYTE)) != 0u) {
        result = fetch_byte_rgba();
      }
      $else { result = kernel_globals.attribute_float4(offset); };
      return result;
    }
  }
};

template <typename... Tuples>
using tuple_cat_t = decltype(std::tuple_cat(std::declval<Tuples>()...));

template <typename Ret, typename ArgTuple> struct CallableFromTuple;
template <typename Ret, typename... Args>
struct CallableFromTuple<Ret, std::tuple<Args...>> {
  using type = Callable<Ret(Args...)>;
};
template <typename Ret, typename ArgTuple>
using CallableFromTupleT = typename CallableFromTuple<Ret, ArgTuple>::type;

template <typename T, int DataMode> struct DataBufTuple;
template <> struct DataBufTuple<float, 0> {
  using type = std::tuple<Buffer<float>>;
  static constexpr bool needs_rec709 = false;
};
template <> struct DataBufTuple<luisa::float2, 0> {
  using type = std::tuple<Buffer<packed_float2>>;
  static constexpr bool needs_rec709 = false;
};
template <> struct DataBufTuple<luisa::float3, 0> {
  using type = std::tuple<Buffer<packed_float3>>;
  static constexpr bool needs_rec709 = false;
};
template <> struct DataBufTuple<luisa::float3, 1> {
  using type = std::tuple<Buffer<packed_float3>, Buffer<packed_normal>>;
  static constexpr bool needs_rec709 = false;
};
template <> struct DataBufTuple<luisa::float4, 0> {
  using type = std::tuple<Buffer<packed_float4>>;
  static constexpr bool needs_rec709 = false;
};
template <> struct DataBufTuple<luisa::float4, 1> {
  using type = std::tuple<Buffer<compiler::cycles_svm::uchar4>>;
  static constexpr bool needs_rec709 = false;
};
template <> struct DataBufTuple<luisa::float4, 2> {
  using type = std::tuple<Buffer<compiler::cycles_svm::uchar4>>;
  static constexpr bool needs_rec709 = true;
};
template <> struct DataBufTuple<luisa::float4, 3> {
  using type =
      std::tuple<Buffer<packed_float4>, Buffer<compiler::cycles_svm::uchar4>>;
  static constexpr bool needs_rec709 = false;
};
template <> struct DataBufTuple<luisa::float4, 4> {
  using type =
      std::tuple<Buffer<packed_float4>, Buffer<compiler::cycles_svm::uchar4>>;
  static constexpr bool needs_rec709 = true;
};

template <typename T, bool derivatives> struct CachedSurfaceAttrCallable {
  virtual ~CachedSurfaceAttrCallable() noexcept = default;
  [[nodiscard]] virtual DualValue<T>
  invoke(const SurfaceAttributeBuffers &bufs, const ShaderData &shader_data,
         const AttributeDescriptor &descriptor) const noexcept = 0;
};

template <typename T, bool derivatives, bool HasTri, bool HasCurve,
          int DataMode>
struct TypedSurfaceAttrCallable final
    : CachedSurfaceAttrCallable<T, derivatives> {
  using DataInfo = DataBufTuple<T, DataMode>;
  using TriTuple =
      std::conditional_t<HasTri, std::tuple<Buffer<packed_uint3>>, std::tuple<>>;
  using CurveTuple =
      std::conditional_t<HasCurve, std::tuple<Buffer<KernelCurve>>, std::tuple<>>;
  using DataTuple = typename DataInfo::type;
  using CoreTuple = std::conditional_t<
      derivatives, std::tuple<luisa::uint4, luisa::float2, luisa::float4>,
      std::tuple<luisa::uint4, luisa::float2>>;
  using ColorTuple = std::conditional_t<
      DataInfo::needs_rec709,
      std::tuple<luisa::float3, luisa::float3, luisa::float3>, std::tuple<>>;
  using AllArgsTuple =
      tuple_cat_t<TriTuple, CurveTuple, DataTuple, CoreTuple, ColorTuple>;
  using Ret = std::conditional_t<derivatives, DualPod<T>, T>;
  using CallableType = CallableFromTupleT<Ret, AllArgsTuple>;

  static constexpr std::size_t idx_tri = 0u;
  static constexpr std::size_t idx_curve = idx_tri + (HasTri ? 1u : 0u);
  static constexpr std::size_t idx_data0 = idx_curve + (HasCurve ? 1u : 0u);
  static constexpr std::size_t num_data_bufs = std::tuple_size_v<DataTuple>;
  static constexpr std::size_t idx_core = idx_data0 + num_data_bufs;
  static constexpr std::size_t idx_color = idx_core + (derivatives ? 3u : 2u);

  CallableType callable;

  template <typename RefTuple> struct BufferFetchers {
    RefTuple &refs;

    [[nodiscard]] UInt3
    triangle_vertex_indices(Expr<std::uint32_t> prim) const noexcept {
      if constexpr (HasTri) {
        const auto v = std::get<idx_tri>(refs).read(prim);
        return make_uint3(v.x, v.y, v.z);
      } else {
        return make_uint3(0u);
      }
    }

    [[nodiscard]] Var<KernelCurve>
    curve(Expr<std::uint32_t> prim) const noexcept {
      if constexpr (HasCurve) {
        return std::get<idx_curve>(refs).read(prim);
      } else {
        Var<KernelCurve> dummy;
        return dummy;
      }
    }

    [[nodiscard]] Var<T> fetch_attr(Expr<std::uint32_t> element,
                                    Expr<std::int32_t> offset) const noexcept {
      const UInt uoffset = offset.cast<std::uint32_t>();
      if constexpr (std::is_same_v<T, float>) {
        return std::get<idx_data0>(refs).read(uoffset);
      } else if constexpr (std::is_same_v<T, luisa::float2>) {
        const auto v = std::get<idx_data0>(refs).read(uoffset);
        return make_float2(v.x, v.y);
      } else if constexpr (std::is_same_v<T, luisa::float3>) {
        Float3 result = packed_float3_value(std::get<idx_data0>(refs).read(uoffset));
        if constexpr (DataMode == 1) {
          $if((element & static_cast<std::uint32_t>(ATTR_ELEMENT_IS_NORMAL)) !=
              0u) {
            result =
                decode_packed_normal(std::get<idx_data0 + 1u>(refs).read(uoffset));
          };
        }
        return result;
      } else {
        static_assert(std::is_same_v<T, luisa::float4>);
        if constexpr (DataMode == 0) {
          const auto v = std::get<idx_data0>(refs).read(uoffset);
          return make_float4(v.x, v.y, v.z, v.w);
        } else {
          auto fetch_byte = [&]() noexcept {
            constexpr std::size_t byte_idx =
                (DataMode == 1 || DataMode == 2) ? idx_data0 : (idx_data0 + 1u);
            const Float4 linear = decode_byte_rgba_linear_rec709(
                std::get<byte_idx>(refs).read(uoffset));
            if constexpr (DataInfo::needs_rec709) {
              const Float3 &r = std::get<idx_color>(refs);
              const Float3 &g = std::get<idx_color + 1u>(refs);
              const Float3 &b = std::get<idx_color + 2u>(refs);
              return make_float4(make_float3(dot(r, linear.xyz()),
                                             dot(g, linear.xyz()),
                                             dot(b, linear.xyz())),
                                 linear.w);
            } else {
              return linear;
            }
          };
          if constexpr (DataMode == 1 || DataMode == 2) {
            return fetch_byte();
          } else {
            Float4 result;
            $if((element & static_cast<std::uint32_t>(ATTR_ELEMENT_IS_BYTE)) !=
                0u) {
              result = fetch_byte();
            }
            $else {
              const auto v = std::get<idx_data0>(refs).read(uoffset);
              result = make_float4(v.x, v.y, v.z, v.w);
            };
            return result;
          }
        }
      }
    }
  };

  explicit TypedSurfaceAttrCallable(const SurfaceAttrSpec &spec) noexcept
      : callable{[spec](auto &&...all_args) noexcept {
          auto refs = std::forward_as_tuple(all_args...);
          const UInt4 &prim_desc = std::get<idx_core>(refs);
          const Float2 &uv = std::get<idx_core + 1u>(refs);
          const UInt sd_type = prim_desc.x;
          const UInt sd_prim = prim_desc.y;
          const UInt desc_elem = prim_desc.z;
          const Int desc_offset = prim_desc.w.bitcast<std::int32_t>();
          Differential du{.dx = 0.0f, .dy = 0.0f};
          Differential dv{.dx = 0.0f, .dy = 0.0f};
          if constexpr (derivatives) {
            const Float4 &du_dv = std::get<idx_core + 2u>(refs);
            du = {.dx = du_dv.x, .dy = du_dv.y};
            dv = {.dx = du_dv.z, .dy = du_dv.w};
          }
          BufferFetchers<decltype(refs)> fetchers{refs};
          auto value = emit_primitive_surface_attribute_body<T, derivatives>(
              fetchers, spec, sd_type, sd_prim, uv.x, uv.y, du, dv, desc_elem,
              desc_offset);
          if constexpr (derivatives) {
            Var<DualPod<T>> packed;
            packed.val = value.val;
            packed.dx = value.dx;
            packed.dy = value.dy;
            return packed;
          } else {
            return value.val;
          }
        }} {
    int dims = 1;
    if constexpr (std::is_same_v<T, luisa::float2>) {
      dims = 2;
    } else if constexpr (std::is_same_v<T, luisa::float3>) {
      dims = 3;
    } else if constexpr (std::is_same_v<T, luisa::float4>) {
      dims = 4;
    }
    char name_buf[64];
    std::snprintf(name_buf, sizeof(name_buf), "cycles_prim_attr_f%d_d%d_k%03x",
                  dims, derivatives ? 1 : 0, spec.key());
    callable.set_name(name_buf);
  }

  [[nodiscard]] DualValue<T>
  invoke(const SurfaceAttributeBuffers &bufs, const ShaderData &shader_data,
         const AttributeDescriptor &descriptor) const noexcept override {
    auto tri_args = [&]() noexcept {
      if constexpr (HasTri) {
        return std::make_tuple(
            Expr<Buffer<packed_uint3>>{*bufs.triangle_index_buffer});
      } else {
        return std::tuple<>{};
      }
    }();
    auto curve_args = [&]() noexcept {
      if constexpr (HasCurve) {
        return std::make_tuple(Expr<Buffer<KernelCurve>>{*bufs.curve_buffer});
      } else {
        return std::tuple<>{};
      }
    }();
    auto data_args = [&]() noexcept {
      if constexpr (std::is_same_v<T, float>) {
        return std::make_tuple(
            Expr<Buffer<float>>{*bufs.attribute_float_buffer});
      } else if constexpr (std::is_same_v<T, luisa::float2>) {
        return std::make_tuple(
            Expr<Buffer<packed_float2>>{*bufs.attribute_float2_buffer});
      } else if constexpr (std::is_same_v<T, luisa::float3>) {
        if constexpr (DataMode == 0) {
          return std::make_tuple(
              Expr<Buffer<packed_float3>>{*bufs.attribute_float3_buffer});
        } else {
          return std::make_tuple(
              Expr<Buffer<packed_float3>>{*bufs.attribute_float3_buffer},
              Expr<Buffer<packed_normal>>{*bufs.attribute_normal_buffer});
        }
      } else {
        static_assert(std::is_same_v<T, luisa::float4>);
        if constexpr (DataMode == 0) {
          return std::make_tuple(
              Expr<Buffer<packed_float4>>{*bufs.attribute_float4_buffer});
        } else if constexpr (DataMode == 1 || DataMode == 2) {
          return std::make_tuple(
              Expr<Buffer<compiler::cycles_svm::uchar4>>{
                  *bufs.attribute_uchar4_buffer});
        } else {
          return std::make_tuple(
              Expr<Buffer<packed_float4>>{*bufs.attribute_float4_buffer},
              Expr<Buffer<compiler::cycles_svm::uchar4>>{
                  *bufs.attribute_uchar4_buffer});
        }
      }
    }();
    const UInt4 prim_desc =
        make_uint4(shader_data.type, shader_data.prim, descriptor.element,
                   descriptor.offset.bitcast<std::uint32_t>());
    const Float2 uv = make_float2(shader_data.u, shader_data.v);
    auto core_args = [&]() noexcept {
      if constexpr (derivatives) {
        const Float4 du_dv =
            make_float4(shader_data.du.dx, shader_data.du.dy,
                        shader_data.dv.dx, shader_data.dv.dy);
        return std::make_tuple(Expr<luisa::uint4>{prim_desc},
                               Expr<luisa::float2>{uv},
                               Expr<luisa::float4>{du_dv});
      } else {
        return std::make_tuple(Expr<luisa::uint4>{prim_desc},
                               Expr<luisa::float2>{uv});
      }
    }();
    auto color_args = [&]() noexcept {
      if constexpr (DataInfo::needs_rec709) {
        return std::make_tuple(Expr<luisa::float3>{bufs.rec709_to_r},
                               Expr<luisa::float3>{bufs.rec709_to_g},
                               Expr<luisa::float3>{bufs.rec709_to_b});
      } else {
        return std::tuple<>{};
      }
    }();
    auto all_args =
        std::tuple_cat(tri_args, curve_args, data_args, core_args, color_args);
    auto call_result = std::apply(
        [&](auto &&...a) noexcept {
          return callable(std::forward<decltype(a)>(a)...);
        },
        all_args);
    if constexpr (derivatives) {
      return DualValue<T>{.val = call_result.val,
                          .dx = call_result.dx,
                          .dy = call_result.dy};
    } else {
      return DualValue<T>{.val = call_result,
                          .dx = zero_value<T>(),
                          .dy = zero_value<T>()};
    }
  }
};

template <typename T, bool derivatives, bool HasTri, bool HasCurve>
[[nodiscard]] std::unique_ptr<CachedSurfaceAttrCallable<T, derivatives>>
build_entry_for_data_mode(const SurfaceAttrSpec &spec) noexcept {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, luisa::float2>) {
    return std::make_unique<
        TypedSurfaceAttrCallable<T, derivatives, HasTri, HasCurve, 0>>(spec);
  } else if constexpr (std::is_same_v<T, luisa::float3>) {
    if (spec.may_normal) {
      return std::make_unique<
          TypedSurfaceAttrCallable<T, derivatives, HasTri, HasCurve, 1>>(spec);
    }
    return std::make_unique<
        TypedSurfaceAttrCallable<T, derivatives, HasTri, HasCurve, 0>>(spec);
  } else {
    static_assert(std::is_same_v<T, luisa::float4>);
    if (!spec.may_byte) {
      return std::make_unique<
          TypedSurfaceAttrCallable<T, derivatives, HasTri, HasCurve, 0>>(spec);
    }
    if (!spec.may_float4) {
      if (spec.is_rec709) {
        return std::make_unique<
            TypedSurfaceAttrCallable<T, derivatives, HasTri, HasCurve, 1>>(spec);
      }
      return std::make_unique<
          TypedSurfaceAttrCallable<T, derivatives, HasTri, HasCurve, 2>>(spec);
    }
    if (spec.is_rec709) {
      return std::make_unique<
          TypedSurfaceAttrCallable<T, derivatives, HasTri, HasCurve, 3>>(spec);
    }
    return std::make_unique<
        TypedSurfaceAttrCallable<T, derivatives, HasTri, HasCurve, 4>>(spec);
  }
}

template <typename T, bool derivatives>
[[nodiscard]] std::unique_ptr<CachedSurfaceAttrCallable<T, derivatives>>
build_cached_surface_attr_callable(const SurfaceAttrSpec &spec) noexcept {
  if (spec.may_vertex) {
    if (spec.has_curves) {
      return build_entry_for_data_mode<T, derivatives, true, true>(spec);
    }
    return build_entry_for_data_mode<T, derivatives, true, false>(spec);
  }
  if (spec.has_curves) {
    return build_entry_for_data_mode<T, derivatives, false, true>(spec);
  }
  return build_entry_for_data_mode<T, derivatives, false, false>(spec);
}

template <typename T>
[[nodiscard]] bool
has_required_surface_attr_buffers(const SurfaceAttributeBuffers &bufs,
                                  const SurfaceAttrSpec &spec) noexcept {
  if (spec.may_vertex &&
      (!bufs.triangle_index_buffer || !(*bufs.triangle_index_buffer))) {
    return false;
  }
  if (spec.has_curves && (!bufs.curve_buffer || !(*bufs.curve_buffer))) {
    return false;
  }
  if constexpr (std::is_same_v<T, float>) {
    return bufs.attribute_float_buffer && *bufs.attribute_float_buffer;
  } else if constexpr (std::is_same_v<T, luisa::float2>) {
    return bufs.attribute_float2_buffer && *bufs.attribute_float2_buffer;
  } else if constexpr (std::is_same_v<T, luisa::float3>) {
    if (!bufs.attribute_float3_buffer || !(*bufs.attribute_float3_buffer)) {
      return false;
    }
    if (spec.may_normal &&
        (!bufs.attribute_normal_buffer || !(*bufs.attribute_normal_buffer))) {
      return false;
    }
    return true;
  } else {
    static_assert(std::is_same_v<T, luisa::float4>);
    if ((!spec.may_byte || spec.may_float4) &&
        (!bufs.attribute_float4_buffer || !(*bufs.attribute_float4_buffer))) {
      return false;
    }
    if (spec.may_byte &&
        (!bufs.attribute_uchar4_buffer || !(*bufs.attribute_uchar4_buffer))) {
      return false;
    }
    return true;
  }
}

template <typename T, bool derivatives>
[[nodiscard]] const CachedSurfaceAttrCallable<T, derivatives> &
get_cached_surface_attr_callable(const SurfaceAttrSpec &spec) noexcept {
  struct Cache {
    std::mutex mutex;
    std::unordered_map<std::uint32_t,
                       std::unique_ptr<CachedSurfaceAttrCallable<T, derivatives>>>
        entries;
  };
  static auto &cache = *new Cache{};
  const std::uint32_t key = spec.key();
  std::lock_guard lock{cache.mutex};
  auto [it, inserted] = cache.entries.try_emplace(key, nullptr);
  if (inserted || !it->second) {
    it->second = build_cached_surface_attr_callable<T, derivatives>(spec);
  }
  return *it->second;
}

template <typename T, bool derivatives>
[[nodiscard]] DualValue<T> primitive_surface_attribute_impl(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeDescriptor &descriptor) noexcept {
  const auto summary = kernel_globals.surface_attribute_summary();
  if (summary.element_mask == 0u) {
    return zero_dual_value<T>();
  }
  if (auto bufs = kernel_globals.surface_attribute_buffers()) {
    const auto spec = make_surface_attr_spec<T>(summary, bufs->is_rec709);
    if (has_required_surface_attr_buffers<T>(*bufs, spec)) {
      const auto &cached =
          get_cached_surface_attr_callable<T, derivatives>(spec);
      return cached.invoke(*bufs, shader_data, descriptor);
    }
  }
  const auto spec = make_surface_attr_spec<T>(summary, true);
  KernelGlobalsFetchers<T> fetchers{kernel_globals, spec};
  return emit_primitive_surface_attribute_body<T, derivatives>(
      fetchers, spec, shader_data.type, shader_data.prim, shader_data.u,
      shader_data.v, shader_data.du, shader_data.dv, descriptor.element,
      descriptor.offset);
}

} // namespace

namespace detail {

AttributeDescriptor find_attribute_from_base_offset(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    Expr<std::uint32_t> base_attr_offset, Expr<luisa::ulong> id) noexcept {
  if (auto bufs = kernel_globals.surface_attribute_buffers();
      bufs && bufs->attribute_map_buffer && *bufs->attribute_map_buffer) {
    static auto callable = []() noexcept {
      Callable result{[](BufferVar<AttributeMap> attr_maps, UInt base_offset,
                         UInt prim, ULong target_id) noexcept {
        return emit_find_attr_from_base_packed(attr_maps, base_offset, prim,
                                               target_id);
      }};
      result.set_name("cycles_find_attribute_from_base");
      return result;
    }();
    const UInt3 packed = callable(*bufs->attribute_map_buffer, base_attr_offset,
                                  shader_data.prim, id);
    return {.element = packed.x,
            .type = packed.y,
            .offset = packed.z.bitcast<std::int32_t>()};
  }
  auto descriptor = attribute_not_found();
  UInt attr_offset = base_attr_offset;
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
  return descriptor;
}

} // namespace detail

namespace {

struct DirectJitCaseCache {
  const luisa::compute::ScopeStmt *root_scope{nullptr};

  struct FindKey {
    const void *obj_buf;
    const void *map_buf;
    const luisa::compute::Expression *object_expr;
    const luisa::compute::Expression *prim_expr;
    std::uint64_t attr_id;
    bool operator==(const FindKey &) const = default;
  };
  struct FindKeyHash {
    std::size_t operator()(const FindKey &k) const noexcept {
      std::size_t h = std::hash<const void *>{}(k.obj_buf);
      h ^= std::hash<const void *>{}(k.map_buf) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      h ^= std::hash<const void *>{}(k.object_expr) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      h ^= std::hash<const void *>{}(k.prim_expr) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      h ^= std::hash<std::uint64_t>{}(k.attr_id) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      return h;
    }
  };

  struct EvalKey {
    const luisa::compute::Expression *object_expr;
    const luisa::compute::Expression *prim_expr;
    const luisa::compute::Expression *u_expr;
    const luisa::compute::Expression *v_expr;
    std::uint64_t attr_id;
    std::uint32_t output_type;
    bool operator==(const EvalKey &) const = default;
  };
  struct EvalKeyHash {
    std::size_t operator()(const EvalKey &k) const noexcept {
      std::size_t h = std::hash<const void *>{}(k.object_expr);
      h ^= std::hash<const void *>{}(k.prim_expr) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      h ^= std::hash<const void *>{}(k.u_expr) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      h ^= std::hash<const void *>{}(k.v_expr) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      h ^= std::hash<std::uint64_t>{}(k.attr_id) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      h ^= std::hash<std::uint32_t>{}(k.output_type) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      return h;
    }
  };

  struct Dual3Exprs {
    const luisa::compute::Expression *val;
    const luisa::compute::Expression *dx;
    const luisa::compute::Expression *dy;
  };

  std::unordered_map<FindKey, const luisa::compute::Expression *, FindKeyHash>
      find_attr_cache;
  std::unordered_map<EvalKey, const luisa::compute::Expression *, EvalKeyHash>
      surface_eval_cache;
  std::unordered_map<EvalKey, Dual3Exprs, EvalKeyHash> deriv_eval_cache;

  [[nodiscard]] bool at_root_scope() const noexcept {
    const auto *fb = luisa::compute::detail::FunctionBuilder::current();
    return fb != nullptr && root_scope != nullptr &&
           fb->current_scope() == root_scope;
  }
};

thread_local DirectJitCaseCache *tls_direct_jit_case_cache = nullptr;

[[nodiscard]] std::optional<std::uint64_t>
try_const_ulong(luisa::compute::Expr<luisa::ulong> expr) noexcept {
  const auto *raw = expr.expression();
  if (raw != nullptr && raw->tag() == luisa::compute::Expression::Tag::LITERAL) {
    const auto &val =
        static_cast<const luisa::compute::LiteralExpr *>(raw)->value();
    if (const auto *u = luisa::get_if<luisa::ulong>(&val)) {
      return static_cast<std::uint64_t>(*u);
    }
    if (const auto *s = luisa::get_if<luisa::slong>(&val)) {
      return static_cast<std::uint64_t>(*s);
    }
    if (const auto *u32 = luisa::get_if<luisa::uint>(&val)) {
      return static_cast<std::uint64_t>(*u32);
    }
  }
  return std::nullopt;
}

} // namespace

namespace detail {

DirectJitCaseCacheGuard::DirectJitCaseCacheGuard() noexcept {
  auto *cache = new DirectJitCaseCache{};
  const auto *fb = luisa::compute::detail::FunctionBuilder::current();
  cache->root_scope = fb != nullptr ? fb->current_scope() : nullptr;
  _prev = tls_direct_jit_case_cache;
  _owned = cache;
  tls_direct_jit_case_cache = cache;
}

DirectJitCaseCacheGuard::~DirectJitCaseCacheGuard() noexcept {
  tls_direct_jit_case_cache = static_cast<DirectJitCaseCache *>(_prev);
  delete static_cast<DirectJitCaseCache *>(_owned);
}

std::optional<luisa::compute::Float3>
lookup_cached_surface_attr(const ShaderData &shader_data,
                           std::uint64_t attr_id,
                           std::uint32_t output_type) noexcept {
  auto *cache = tls_direct_jit_case_cache;
  if (cache == nullptr) {
    return std::nullopt;
  }
  const DirectJitCaseCache::EvalKey key{
      shader_data.object.expression(), shader_data.prim.expression(),
      shader_data.u.expression(), shader_data.v.expression(),
      attr_id, output_type};
  if (auto it = cache->surface_eval_cache.find(key);
      it != cache->surface_eval_cache.end()) {
    return luisa::compute::Float3{
        luisa::compute::Expr<luisa::float3>{it->second}};
  }
  if (auto it = cache->deriv_eval_cache.find(key);
      it != cache->deriv_eval_cache.end()) {
    return luisa::compute::Float3{
        luisa::compute::Expr<luisa::float3>{it->second.val}};
  }
  return std::nullopt;
}

void store_cached_surface_attr(const ShaderData &shader_data,
                               std::uint64_t attr_id,
                               std::uint32_t output_type,
                               const luisa::compute::Float3 &value) noexcept {
  auto *cache = tls_direct_jit_case_cache;
  if (cache == nullptr || !cache->at_root_scope()) {
    return;
  }
  const DirectJitCaseCache::EvalKey key{
      shader_data.object.expression(), shader_data.prim.expression(),
      shader_data.u.expression(), shader_data.v.expression(),
      attr_id, output_type};
  const luisa::compute::Float3 ssa = value;
  cache->surface_eval_cache.try_emplace(key, ssa.expression());
}

std::optional<Dual3>
lookup_cached_deriv_attr(const ShaderData &shader_data,
                         std::uint64_t attr_id,
                         std::uint32_t output_type) noexcept {
  auto *cache = tls_direct_jit_case_cache;
  if (cache == nullptr) {
    return std::nullopt;
  }
  const DirectJitCaseCache::EvalKey key{
      shader_data.object.expression(), shader_data.prim.expression(),
      shader_data.u.expression(), shader_data.v.expression(),
      attr_id, output_type};
  if (auto it = cache->deriv_eval_cache.find(key);
      it != cache->deriv_eval_cache.end()) {
    return Dual3{
        .val = luisa::compute::Float3{
            luisa::compute::Expr<luisa::float3>{it->second.val}},
        .dx = luisa::compute::Float3{
            luisa::compute::Expr<luisa::float3>{it->second.dx}},
        .dy = luisa::compute::Float3{
            luisa::compute::Expr<luisa::float3>{it->second.dy}}};
  }
  return std::nullopt;
}

void store_cached_deriv_attr(const ShaderData &shader_data,
                             std::uint64_t attr_id,
                             std::uint32_t output_type,
                             const Dual3 &value) noexcept {
  auto *cache = tls_direct_jit_case_cache;
  if (cache == nullptr || !cache->at_root_scope()) {
    return;
  }
  const DirectJitCaseCache::EvalKey key{
      shader_data.object.expression(), shader_data.prim.expression(),
      shader_data.u.expression(), shader_data.v.expression(),
      attr_id, output_type};
  const luisa::compute::Float3 sv = value.val;
  const luisa::compute::Float3 sdx = value.dx;
  const luisa::compute::Float3 sdy = value.dy;
  cache->deriv_eval_cache.try_emplace(
      key, DirectJitCaseCache::Dual3Exprs{sv.expression(), sdx.expression(),
                                          sdy.expression()});
}

} // namespace detail

AttributeDescriptor find_attribute(const KernelGlobals &kernel_globals,
                                   const ShaderData &shader_data,
                                   Expr<luisa::ulong> id) noexcept {
  if (auto bufs = kernel_globals.surface_attribute_buffers();
      bufs && bufs->object_buffer && *bufs->object_buffer &&
      bufs->attribute_map_buffer && *bufs->attribute_map_buffer) {
    auto *cache = tls_direct_jit_case_cache;
    const auto const_id = cache != nullptr ? try_const_ulong(id) : std::nullopt;
    if (cache != nullptr && const_id) {
      const DirectJitCaseCache::FindKey key{
          bufs->object_buffer,
          bufs->attribute_map_buffer,
          shader_data.object.expression(), shader_data.prim.expression(),
          *const_id};
      if (auto it = cache->find_attr_cache.find(key);
          it != cache->find_attr_cache.end()) {
        const Expr<luisa::uint3> packed{it->second};
        return {.element = packed.x,
                .type = packed.y,
                .offset = packed.z.bitcast<std::int32_t>()};
      }
    }
    static auto callable = []() noexcept {
      Callable result{[](BufferVar<KernelObject> objects,
                         BufferVar<AttributeMap> attr_maps, UInt object,
                         UInt prim, ULong target_id) noexcept {
        UInt3 packed = make_uint3(
            static_cast<std::uint32_t>(ATTR_ELEMENT_NONE),
            static_cast<std::uint32_t>(NODE_ATTR_FLOAT),
            static_cast<std::uint32_t>(
                static_cast<std::int32_t>(ATTR_STD_NOT_FOUND)));
        $if(object != object_none) {
          const UInt base_attr_offset = objects.read(object).attribute_map_offset;
          packed = emit_find_attr_from_base_packed(attr_maps, base_attr_offset,
                                                   prim, target_id);
        };
        return packed;
      }};
      result.set_name("cycles_find_attribute");
      return result;
    }();
    const UInt3 packed = callable(*bufs->object_buffer,
                                  *bufs->attribute_map_buffer,
                                  shader_data.object, shader_data.prim, id);
    if (cache != nullptr && const_id && cache->at_root_scope()) {
      const DirectJitCaseCache::FindKey key{
          bufs->object_buffer,
          bufs->attribute_map_buffer,
          shader_data.object.expression(), shader_data.prim.expression(),
          *const_id};
      cache->find_attr_cache.try_emplace(key, packed.expression());
    }
    return {.element = packed.x,
            .type = packed.y,
            .offset = packed.z.bitcast<std::int32_t>()};
  }
  auto descriptor = attribute_not_found();
  $if(shader_data.object != object_none) {
    const UInt base_attr_offset =
        kernel_globals.object_attribute_map_offset(shader_data.object);
    descriptor = detail::find_attribute_from_base_offset(
        kernel_globals, shader_data, base_attr_offset, id);
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
