#pragma once

#include <cstdint>

namespace psycles::contract::cycles_abi {

// Current Cycles ShaderFlag ABI. Low bits identify Scene::shaders; the high
// bits are per-ray/per-hit flags and must not participate in shader lookup.
inline constexpr std::uint32_t shader_mask =
    0x203fffffu;

}// namespace psycles::contract::cycles_abi
