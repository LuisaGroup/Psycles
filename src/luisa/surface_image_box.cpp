#include "surface_image_box.h"

#include "surface_image_sampling.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] Float4 sample_surface_image_box_texture(
    const SurfaceImageBoxInput &input,
    const SurfaceImageBoxTextureSampler &sampler,
    Float2 uv) noexcept {
    // Blender UVs use a bottom-left origin while decoded host images are
    // uploaded in top-to-bottom row order.
    uv.y = 1.0f - uv.y;
    return decode_surface_image_sample(
        sampler.sample(input.texture_handle, uv),
        input.unassociate_alpha,
        input.encoded_as_srgb);
}

}// namespace

Float4 evaluate_surface_image_box(
    const SurfaceImageBoxInput &input,
    const SurfaceImageBoxTextureSampler &sampler) noexcept {
    auto normal = abs(input.signed_normal);
    auto normal_sum = normal.x + normal.y + normal.z;
    normal /= select(
        1.0f,
        normal_sum,
        normal_sum != 0.0f);

    Float3 weight = make_float3(0.0f);
    const auto limit = 0.5f * (1.0f + input.blend);
    $if((normal.x > limit * (normal.x + normal.y)) &
        (normal.x > limit * (normal.x + normal.z))) {
        weight.x = 1.0f;
    }
    $elif((normal.y > limit * (normal.x + normal.y)) &
          (normal.y > limit * (normal.y + normal.z))) {
        weight.y = 1.0f;
    }
    $elif((normal.z > limit * (normal.x + normal.z)) &
          (normal.z > limit * (normal.y + normal.z))) {
        weight.z = 1.0f;
    }
    $elif(input.blend > 0.0f) {
        $if(normal.z < (1.0f - limit) * (normal.y + normal.x)) {
            weight.x = normal.x / (normal.x + normal.y);
            weight.x = luisa::compute::clamp(
                (weight.x - 0.5f * (1.0f - input.blend)) /
                    input.blend,
                0.0f,
                1.0f);
            weight.y = 1.0f - weight.x;
        }
        $elif(normal.x < (1.0f - limit) * (normal.y + normal.z)) {
            weight.y = normal.y / (normal.y + normal.z);
            weight.y = luisa::compute::clamp(
                (weight.y - 0.5f * (1.0f - input.blend)) /
                    input.blend,
                0.0f,
                1.0f);
            weight.z = 1.0f - weight.y;
        }
        $elif(normal.y < (1.0f - limit) * (normal.x + normal.z)) {
            weight.x = normal.x / (normal.x + normal.z);
            weight.x = luisa::compute::clamp(
                (weight.x - 0.5f * (1.0f - input.blend)) /
                    input.blend,
                0.0f,
                1.0f);
            weight.z = 1.0f - weight.x;
        }
        $else {
            weight.x =
                ((2.0f - limit) * normal.x + (limit - 1.0f)) /
                (2.0f * limit - 1.0f);
            weight.y =
                ((2.0f - limit) * normal.y + (limit - 1.0f)) /
                (2.0f * limit - 1.0f);
            weight.z =
                ((2.0f - limit) * normal.z + (limit - 1.0f)) /
                (2.0f * limit - 1.0f);
        };
    }
    $else {
        weight.x = 1.0f;
    };

    const auto uv_x = make_float2(
        select(
            input.coordinate.y,
            1.0f - input.coordinate.y,
            input.signed_normal.x < 0.0f),
        input.coordinate.z);
    const auto uv_y = make_float2(
        select(
            input.coordinate.x,
            1.0f - input.coordinate.x,
            input.signed_normal.y > 0.0f),
        input.coordinate.z);
    const auto uv_z = make_float2(
        select(
            input.coordinate.y,
            1.0f - input.coordinate.y,
            input.signed_normal.z > 0.0f),
        input.coordinate.x);

    // Texture reads are not total over IEEE-754 values: an unselected face may
    // contain NaN or Inf, for which 0 * sample is observably different from not
    // sampling. Match Cycles' guarded, left-to-right accumulation exactly.
    Float4 sampled = make_float4(0.0f);
    $if(weight.x > 0.0f) {
        sampled += weight.x * sample_surface_image_box_texture(
                                  input, sampler, uv_x);
    };
    $if(weight.y > 0.0f) {
        sampled += weight.y * sample_surface_image_box_texture(
                                  input, sampler, uv_y);
    };
    $if(weight.z > 0.0f) {
        sampled += weight.z * sample_surface_image_box_texture(
                                  input, sampler, uv_z);
    };
    return sampled;
}

}// namespace psycles::luisa_backend::detail
