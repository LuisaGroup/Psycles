#include <psycles/luisa/cycles_transform.h>

namespace psycles::luisa_backend::cycles_transform {
namespace {

[[nodiscard]] luisa::compute::Float3 affine(
    luisa::compute::Expr<luisa::float4x4> transform,
    luisa::compute::Expr<luisa::float3> value,
    bool include_translation) noexcept {
  using namespace luisa::compute;
  const auto c0 = transform[0u];
  const auto c1 = transform[1u];
  const auto c2 = transform[2u];
  const auto c3 = transform[3u];
  const auto tx = include_translation ? c3.x : 0.0f;
  const auto ty = include_translation ? c3.y : 0.0f;
  const auto tz = include_translation ? c3.z : 0.0f;
  return make_float3(
      fma(value.x, c0.x,
          fma(value.y, c1.x, fma(value.z, c2.x, tx))),
      fma(value.x, c0.y,
          fma(value.y, c1.y, fma(value.z, c2.y, ty))),
      fma(value.x, c0.z,
          fma(value.y, c1.z, fma(value.z, c2.z, tz))));
}

} // namespace

luisa::compute::Float3 perspective(
    luisa::compute::Expr<luisa::float4x4> transform,
    luisa::compute::Expr<luisa::float3> value) noexcept {
  using namespace luisa::compute;
  const auto b = make_float4(value, 1.0f);
  const auto row = [&](unsigned i) {
    return make_float4(transform[0u][i], transform[1u][i],
                      transform[2u][i], transform[3u][i]);
  };
  const auto w = dot(row(3u), b);
  Float3 result = make_float3(0.0f);
  $if(w != 0.0f) {
    result = make_float3(dot(row(0u), b), dot(row(1u), b), dot(row(2u), b)) / w;
  };
  return result;
}

luisa::compute::Float3 point(
    luisa::compute::Expr<luisa::float4x4> transform,
    luisa::compute::Expr<luisa::float3> value) noexcept {
  return affine(transform, value, true);
}

luisa::compute::Float3 direction(
    luisa::compute::Expr<luisa::float4x4> transform,
    luisa::compute::Expr<luisa::float3> value) noexcept {
  return affine(transform, value, false);
}

luisa::compute::Float3
direction_transposed(luisa::compute::Expr<luisa::float4x4> transform,
                     luisa::compute::Expr<luisa::float3> value) noexcept {
  using namespace luisa::compute;
  const auto c0 = transform[0u];
  const auto c1 = transform[1u];
  const auto c2 = transform[2u];
  return make_float3(fma(value.x, c0.x, fma(value.y, c0.y, value.z * c0.z)),
                     fma(value.x, c1.x, fma(value.y, c1.y, value.z * c1.z)),
                     fma(value.x, c2.x, fma(value.y, c2.y, value.z * c2.z)));
}

} // namespace psycles::luisa_backend::cycles_transform
