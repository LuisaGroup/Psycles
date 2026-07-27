#pragma once

#include <compare>
#include <cstdint>
#include <limits>

namespace psycles {

template<typename Tag>
struct Id {
    static constexpr auto invalid_value = std::numeric_limits<std::uint64_t>::max();

    std::uint64_t value{invalid_value};

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return value != invalid_value;
    }

    [[nodiscard]] constexpr bool valid() const noexcept {
        return static_cast<bool>(*this);
    }

    auto operator<=>(const Id &) const noexcept = default;
};

}// namespace psycles

