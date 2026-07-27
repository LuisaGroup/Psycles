#pragma once

#include <cstdint>

namespace psycles::contract {

enum class TransportMode : std::uint32_t {
    radiance = 0u,
    importance = 1u
};

enum SurfaceEvent : std::uint32_t {
    event_none = 0u,
    event_diffuse = 1u << 0u,
    event_glossy = 1u << 1u,
    event_singular = 1u << 2u,
    event_reflection = 1u << 3u,
    event_transmission = 1u << 4u,
    event_transparent = 1u << 5u,
    event_subsurface = 1u << 6u
};

}// namespace psycles::contract
