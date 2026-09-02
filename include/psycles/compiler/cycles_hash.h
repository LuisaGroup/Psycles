#pragma once

#include <cstdint>
#include <string_view>

namespace psycles::compiler {

// Exact Cycles util_murmur_hash3 byte-string projection with seed zero.
[[nodiscard]] std::uint32_t
cycles_murmur_hash3(std::string_view value) noexcept;

// Exact Cycles util_hash_to_float projection used by Cryptomatte IDs.
[[nodiscard]] float cycles_hash_to_float(std::uint32_t hash) noexcept;

[[nodiscard]] float
cycles_cryptomatte_hash(std::string_view value) noexcept;

} // namespace psycles::compiler
