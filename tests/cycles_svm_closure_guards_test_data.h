#pragma once

#include <array>
#include <cstdint>
#include <span>

// Exact external Cycles 5.2.1 word images already frozen in
// test_luisa_cycles_svm_closure.cpp; only the local jump relocation is present.
// Shared test INPUTS, not expected shader values or a host shader evaluator.
namespace psycles::test_support::closure_guards {

constexpr std::array<std::uint32_t, 24u> glossy_ggx_words{
    0x00000001u, 0x00000004u, 0x00000016u, 0x00000017u, 0x0000000bu,
    0x00000001u, 0x00000000u, 0x00000005u, 0x3f2e147bu, 0x3e75c28fu,
    0x3db851ecu, 0x00000002u, 0x0000000cu, 0x000000ffu, 0x00000000u,
    0x00000000u, 0x00000000u, 0x3ecccccdu, 0x00000000u, 0x00000000u,
    0x0000ff00u, 0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 20u> refraction_ggx_words{
    0x00000001u, 0x00000004u, 0x00000012u, 0x00000013u, 0x0000000bu,
    0x00000001u, 0x00000000u, 0x00000005u, 0x3ec3b96bu, 0x3f26ba28u,
    0x3f5e35b5u, 0x00000002u, 0x00000015u, 0x000000ffu, 0x3e0c6480u,
    0x3f947ae1u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 25u> glass_beckmann_words{
    0x00000001u, 0x00000004u, 0x00000017u, 0x00000018u, 0x0000000bu,
    0x00000001u, 0x00000000u, 0x00000005u, 0x3f800000u, 0x3f800000u,
    0x3f800000u, 0x00000002u, 0x00000018u, 0x000000ffu, 0x3f800000u,
    0x3f800000u, 0x3f800000u, 0x3e315cacu, 0x3fc00000u, 0x00000000u,
    0x3faa3d71u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 26u> metallic_conductor_ggx_words{
    0x00000001u, 0x00000004u, 0x00000018u, 0x00000019u, 0x0000000bu,
    0x00000001u, 0x00000000u, 0x00000002u, 0x0000000au, 0x000000ffu,
    0x0000000cu, 0x3e8a3d71u, 0x3f2e147bu, 0x3fa8f5c3u, 0x40670a3du,
    0x4027ae14u, 0x3ff47ae1u, 0x3e3851ecu, 0x00000000u, 0x00000000u,
    0x00000000u, 0x3faa3d71u, 0x0000ff00u, 0x00000000u, 0x00000000u,
    0x00000000u};

inline const std::array<std::span<const std::uint32_t>, 4u> programs{
    glossy_ggx_words, refraction_ggx_words, glass_beckmann_words,
    metallic_conductor_ggx_words};
inline constexpr std::array capacities{0u, 1u, 4u};
inline constexpr unsigned cases_per_flags = 24u;
inline constexpr unsigned value_count = 16u;
}  // namespace psycles::test_support::closure_guards
