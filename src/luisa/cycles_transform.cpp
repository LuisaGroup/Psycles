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

} // namespace psycles::luisa_backend::cycles_transform
