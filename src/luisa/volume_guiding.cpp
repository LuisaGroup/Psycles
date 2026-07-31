#include <psycles/luisa/volume_guiding.h>

#include <luisa/dsl/builtin.h>

namespace psycles::luisa_backend::volume_guiding {

using namespace luisa::compute;

UInt encode_rgbe(Float3 rgb) noexcept {
    constexpr auto exponent_bias = 15;
    constexpr auto mantissa_bits = 8;
    constexpr auto maximum = 65280.0f;
    constexpr auto minimum_roundable = 0x1p-24f;

    const auto absolute = abs(rgb);
    const auto maximum_component =
        luisa::compute::min(
            luisa::compute::max(
                absolute.x,
                luisa::compute::max(
                    absolute.y,
                    absolute.z)),
            maximum);
    UInt packed = 0u;
    $if(maximum_component >= minimum_roundable) {
        // Cycles' floor_log2f() extracts the unbiased IEEE-754 exponent.
        // maximum_component is normal in this arm, so no subnormal case is
        // hidden behind a target-specific logarithm implementation.
        Int exponent =
            luisa::compute::max(
                -exponent_bias - 1,
                cast<int>(
                    (as<uint>(maximum_component) >> 23u) &
                    0xffu) -
                    127) +
            1;
        Float scale =
            exp2(cast<float>(mantissa_bits - exponent));
        $if(cast<int>(round(maximum_component * scale)) ==
            (1 << mantissa_bits)) {
            exponent += 1;
            scale *= 0.5f;
        };

        const auto rgb_bits = as<luisa::uint3>(rgb);
        const auto sign_bits =
            ((rgb_bits.x >> 31u) << 7u) |
            ((rgb_bits.y >> 31u) << 6u) |
            ((rgb_bits.z >> 31u) << 5u);
        const auto mantissa =
            luisa::compute::min(
                round(absolute * scale),
                make_float3(255.0f));
        const auto exponent_byte =
            (cast<uint>(exponent + exponent_bias) & 0x1fu) |
            sign_bits;
        packed =
            cast<uint>(mantissa.x) |
            (cast<uint>(mantissa.y) << 8u) |
            (cast<uint>(mantissa.z) << 16u) |
            (exponent_byte << 24u);
    };
    return packed;
}

Float3 decode_rgbe(UInt rgbe) noexcept {
    constexpr auto exponent_bias = 15;
    constexpr auto mantissa_bits = 8;

    Float3 result = make_float3(0.0f);
    $if(rgbe != 0u) {
        const auto exponent = rgbe >> 24u;
        const auto scale =
            exp2(cast<float>(
                cast<int>(exponent & 0x1fu) -
                (exponent_bias + mantissa_bits)));
        auto decoded =
            make_float3(
                cast<float>(rgbe & 0xffu),
                cast<float>((rgbe >> 8u) & 0xffu),
                cast<float>((rgbe >> 16u) & 0xffu)) *
            scale;
        auto decoded_bits = as<luisa::uint3>(decoded);
        decoded_bits.x |= (exponent & 0x80u) << 24u;
        decoded_bits.y |= (exponent & 0x40u) << 25u;
        decoded_bits.z |= (exponent & 0x20u) << 26u;
        result = as<luisa::float3>(decoded_bits);
    };
    return result;
}

}// namespace psycles::luisa_backend::volume_guiding
