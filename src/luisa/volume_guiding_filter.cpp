#include "volume_guiding_filter.h"

#include <psycles/luisa/volume_guiding.h>

#include <array>

namespace psycles::luisa_backend::detail {
namespace {

using namespace luisa::compute;
namespace guiding = ::psycles::luisa_backend::volume_guiding;

inline constexpr std::array gaussian{
    0.0012273699895602f,
    0.0084674212370284f,
    0.0379843612914121f,
    0.1108921888487800f,
    0.2108379677336155f,
    0.2611813817992076f,
    0.2108379677336155f,
    0.1108921888487800f,
    0.0379843612914121f,
    0.0084674212370284f,
    0.0012273699895602f};

using FilterXKernel =
    Kernel2D<Buffer<luisa::float4>,
             Buffer<luisa::uint>,
             Buffer<luisa::uint>,
             std::uint32_t>;
using FilterYKernel =
    Kernel2D<Buffer<luisa::uint>,
             Buffer<luisa::uint>,
             std::uint32_t,
             std::uint32_t>;

[[nodiscard]] FilterXKernel make_filter_x() {
    FilterXKernel kernel =
        [](BufferFloat4 raw,
           BufferUInt sample_count,
           BufferUInt intermediate,
           UInt width) noexcept {
        set_block_size(8u, 8u, 1u);
        const auto coordinate = dispatch_id().xy();
        const auto pixel =
            coordinate.y * width + coordinate.x;
        Float3 scatter = make_float3(0.0f);
        Float3 transmit = make_float3(0.0f);
        for (auto tap = 0u;
             tap < guiding::filter_width;
             ++tap) {
            const auto neighbor_x =
                cast<int>(coordinate.x) +
                static_cast<int>(tap) -
                static_cast<int>(
                    guiding::filter_radius);
            const auto inside =
                (neighbor_x >= 0) &
                (neighbor_x < cast<int>(width));
            $if(inside) {
                const auto neighbor =
                    coordinate.y * width +
                    cast<uint>(neighbor_x);
                const auto weight =
                    gaussian[tap] /
                    cast<float>(
                        sample_count.read(
                            neighbor));
                const auto raw_base =
                    neighbor *
                    guiding::
                        raw_pixel_stride;
                scatter +=
                    raw.read(
                           raw_base +
                           guiding::
                               raw_scatter_slot)
                        .xyz() *
                    weight;
                transmit +=
                    raw.read(
                           raw_base +
                           guiding::
                               raw_transmit_slot)
                        .xyz() *
                    weight;
            };
        }
        const auto output_base =
            pixel *
            guiding::denoised_pixel_stride;
        intermediate.write(
            output_base +
                guiding::denoised_scatter_slot,
            guiding::encode_rgbe(scatter));
        intermediate.write(
            output_base +
                guiding::denoised_transmit_slot,
            guiding::encode_rgbe(transmit));
        };
    return kernel;
}

[[nodiscard]] FilterYKernel make_filter_y() {
    FilterYKernel kernel =
        [](BufferUInt intermediate,
           BufferUInt denoised,
           UInt width,
           UInt height) noexcept {
        set_block_size(8u, 8u, 1u);
        const auto coordinate = dispatch_id().xy();
        const auto pixel =
            coordinate.y * width + coordinate.x;
        Float3 scatter = make_float3(0.0f);
        Float3 transmit = make_float3(0.0f);
        for (auto tap = 0u;
             tap < guiding::filter_width;
             ++tap) {
            const auto neighbor_y =
                cast<int>(coordinate.y) +
                static_cast<int>(tap) -
                static_cast<int>(
                    guiding::filter_radius);
            const auto inside =
                (neighbor_y >= 0) &
                (neighbor_y < cast<int>(height));
            $if(inside) {
                const auto neighbor =
                    cast<uint>(neighbor_y) *
                        width +
                    coordinate.x;
                const auto input_base =
                    neighbor *
                    guiding::
                        denoised_pixel_stride;
                scatter +=
                    guiding::decode_rgbe(
                        intermediate.read(
                            input_base +
                            guiding::
                                denoised_scatter_slot)) *
                    gaussian[tap];
                transmit +=
                    guiding::decode_rgbe(
                        intermediate.read(
                            input_base +
                            guiding::
                                denoised_transmit_slot)) *
                    gaussian[tap];
            };
        }
        const auto output_base =
            pixel *
            guiding::denoised_pixel_stride;
        denoised.write(
            output_base +
                guiding::denoised_scatter_slot,
            guiding::encode_rgbe(abs(scatter)));
        denoised.write(
            output_base +
                guiding::denoised_transmit_slot,
            guiding::encode_rgbe(abs(transmit)));
        };
    return kernel;
}

}// namespace

VolumeGuidingFilter::VolumeGuidingFilter(
    Device &device)
    : _filter_x{
          device.compile(
              make_filter_x(),
              ShaderOption{
                  .enable_cache = true,
                  .enable_fast_math = false})},
      _filter_y{
          device.compile(
              make_filter_y(),
              ShaderOption{
                  .enable_cache = true,
                  .enable_fast_math = false})} {}

void VolumeGuidingFilter::dispatch(
    Stream &stream,
    const Buffer<luisa::float4> &raw,
    const Buffer<luisa::uint> &sample_count,
    const Buffer<luisa::uint> &intermediate,
    const Buffer<luisa::uint> &denoised,
    std::uint32_t width,
    std::uint32_t height) const {
    stream
        << _filter_x(raw,
                     sample_count,
                     intermediate,
                     width)
               .dispatch(width, height)
        << _filter_y(intermediate,
                     denoised,
                     width,
                     height)
               .dispatch(width, height);
}

}// namespace psycles::luisa_backend::detail
