#pragma once
#include <array>

// Boundary data only: raw binary32 bits avoid host denormal arithmetic.
inline constexpr std::array<std::array<unsigned, 2>, 9> tiny_weight_inputs{{
    {0, 0}, {1, 4}, {0x00040000, 0x00100000},
    {0x00100000, 0x00400000}, {0x00200000, 0x00800000},
    {0x00400000, 0x01000000}, {0x00800000, 0x01800000},
    {0x15800000, 0x16800000}, {0x3e800000, 0x3f800000}}};
