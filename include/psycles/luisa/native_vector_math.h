#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/native_vector_math.h> through the Psycles::luisa target."
#endif

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::native_vector_math {

namespace detail {

[[nodiscard]] inline luisa::compute::Float3 normalize_in_domain(
    luisa::compute::Float3 value,
    luisa::compute::Float3 fallback,
    luisa::compute::Float length_squared,
    luisa::compute::Bool valid) noexcept {
  using namespace luisa::compute;
  // Luisa records both operands of select. Keep rsqrt inside its positive
  // domain even for an inactive lane, then let each backend lower the native
  // reciprocal-square-root operation. `valid` alone owns the structural
  // zero/threshold contract; no scalar sqrt/div rounding sequence is part of
  // that contract.
  const auto safe_length_squared = select(1.0f, length_squared, valid);
  const auto normalized = value * rsqrt(safe_length_squared);
  return select(fallback, normalized, valid);
}

} // namespace detail

// Cycles normalize: zero is outside the operation domain and therefore stays
// non-finite. Spell it with the native reciprocal-square-root primitive
// because Luisa's generic NORMALIZE operation is intentionally zero-safe on
// some backends.
[[nodiscard]] inline luisa::compute::Float3 normalize_unchecked(
    luisa::compute::Float3 value) noexcept {
  using namespace luisa::compute;
  const auto length_squared = dot(value, value);
  const auto nonzero = length_squared != 0.0f;
  const auto normalized =
      value * rsqrt(select(1.0f, length_squared, nonzero));
  const Float quiet_nan = as<float>(UInt{0x7fc00000u});
  return select(make_float3(quiet_nan), normalized, nonzero);
}

// Cycles safe_normalize: an exactly zero-length vector is returned unchanged.
// NaNs remain in the normalization domain, as they do for `length != 0`.
[[nodiscard]] inline luisa::compute::Float3 safe_normalize_nonzero(
    luisa::compute::Float3 value) noexcept {
  using namespace luisa::compute;
  const auto length_squared = dot(value, value);
  return detail::normalize_in_domain(
      value, value, length_squared, length_squared != 0.0f);
}

// Cycles safe_normalize_fallback: only exact zero selects the fallback.
[[nodiscard]] inline luisa::compute::Float3 safe_normalize_nonzero_or(
    luisa::compute::Float3 value,
    luisa::compute::Float3 fallback) noexcept {
  using namespace luisa::compute;
  const auto length_squared = dot(value, value);
  return detail::normalize_in_domain(
      value, fallback, length_squared, length_squared != 0.0f);
}

// Renderer safety helper for domains that deliberately reject near-zero
// vectors. The threshold is structural; the accepted-domain arithmetic is
// still backend-native.
[[nodiscard]] inline luisa::compute::Float3 normalize_above_or(
    luisa::compute::Float3 value,
    luisa::compute::Float3 fallback,
    luisa::compute::Float minimum_length_squared) noexcept {
  using namespace luisa::compute;
  const auto length_squared = dot(value, value);
  return detail::normalize_in_domain(
      value, fallback, length_squared,
      length_squared > minimum_length_squared);
}

} // namespace psycles::luisa_backend::native_vector_math
