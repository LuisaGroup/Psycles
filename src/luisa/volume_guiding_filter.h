#pragma once

#include <cstdint>

#include <luisa/luisa-compute.h>

namespace psycles::luisa_backend::detail {

using VolumeGuidingFilterXShader =
    luisa::compute::Shader2D<
        luisa::compute::Buffer<luisa::float4>,
        luisa::compute::Buffer<luisa::uint>,
        luisa::compute::Buffer<luisa::uint>,
        std::uint32_t>;

using VolumeGuidingFilterYShader =
    luisa::compute::Shader2D<
        luisa::compute::Buffer<luisa::uint>,
        luisa::compute::Buffer<luisa::uint>,
        std::uint32_t,
        std::uint32_t>;

// Host-stage component which records Cycles' two-pass VSPG Gaussian filter.
// X writes quantized RGBE values to a private intermediate buffer; Y reads
// that immutable image, so every output pixel can be dispatched independently
// without the in-place ring buffer required by Cycles' CPU implementation.
class VolumeGuidingFilter final {

  private:
    VolumeGuidingFilterXShader _filter_x;
    VolumeGuidingFilterYShader _filter_y;

  public:
    explicit VolumeGuidingFilter(
        luisa::compute::Device &device);

    void dispatch(
        luisa::compute::Stream &stream,
        const luisa::compute::Buffer<luisa::float4> &raw,
        const luisa::compute::Buffer<luisa::uint> &sample_count,
        const luisa::compute::Buffer<luisa::uint> &intermediate,
        const luisa::compute::Buffer<luisa::uint> &denoised,
        std::uint32_t width,
        std::uint32_t height) const;
};

}// namespace psycles::luisa_backend::detail
