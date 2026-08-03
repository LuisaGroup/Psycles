#pragma once

#include "path_tracer_internal.h"

#include <psycles/luisa/cycles_nishita.h>

namespace psycles::luisa_backend::detail {

template<typename ScalarParameterBuffer,
         typename VectorParameterBuffer>
class BufferSurfaceParameterServices final
    : public SurfaceParameterServices {

private:
    const ScalarParameterBuffer &_scalar_parameters;
    const VectorParameterBuffer &_vector_parameters;

public:
    explicit BufferSurfaceParameterServices(
        const ScalarParameterBuffer &scalar_parameters,
        const VectorParameterBuffer &vector_parameters) noexcept
        : _scalar_parameters{scalar_parameters},
          _vector_parameters{vector_parameters} {}

    [[nodiscard]] Float parameter_float(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _scalar_parameters->read(block + slot);
    }

    [[nodiscard]] Float3 parameter_float3(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _vector_parameters->read(block + slot);
    }
};

template<typename ScalarParameterBuffer,
         typename VectorParameterBuffer,
         typename CyclesBuffer,
         typename TextureHeap,
         typename GeometryHeap>
class BufferShaderServices final : public ShaderServices {

private:
    const ScalarParameterBuffer &_scalar_parameters;
    const VectorParameterBuffer &_vector_parameters;
    const CyclesBuffer &_cycles_bsdf_tables;
    const TextureHeap &_textures;
    const GeometryHeap &_geometry_heap;
    std::uint32_t _attribute_binding_slot{};
    std::uint32_t _attribute_range_slot{};
    const std::vector<NishitaTextureBinding> &_nishita_textures;
    const contract::ShaderColorSpace &_shader_color_space;

public:
    explicit BufferShaderServices(
        const ScalarParameterBuffer &scalar_parameters,
        const VectorParameterBuffer &vector_parameters,
        const CyclesBuffer &cycles_bsdf_tables,
        const TextureHeap &textures,
        const GeometryHeap &geometry_heap,
        std::uint32_t attribute_binding_slot,
        std::uint32_t attribute_range_slot,
        const std::vector<NishitaTextureBinding>
            &nishita_textures,
        const contract::ShaderColorSpace
            &shader_color_space) noexcept
        : _scalar_parameters{scalar_parameters},
          _vector_parameters{vector_parameters},
          _cycles_bsdf_tables{cycles_bsdf_tables},
          _textures{textures},
          _geometry_heap{geometry_heap},
          _attribute_binding_slot{
              attribute_binding_slot},
          _attribute_range_slot{
              attribute_range_slot},
          _nishita_textures{nishita_textures},
          _shader_color_space{shader_color_space} {}

    [[nodiscard]] Float4 texture_2d(
        Expr<std::uint32_t> handle,
        Expr<luisa::float2> uv,
        Expr<luisa::float2>,
        Expr<luisa::float2>,
        std::uint32_t interpolation,
        std::uint32_t extension) const noexcept override {
        // Implement Cycles' TextureInterpolator explicitly. Relying on a
        // backend sampler here makes clip-border, mirror, and cubic behavior
        // backend-dependent and differs from Cycles at texel boundaries.
        auto texture = _textures->tex2d(handle);
        auto size = texture.size();
        Int width = cast<int>(size.x);
        Int height = cast<int>(size.y);

        const auto split_coordinate =
            [](Float coordinate, Int &index) noexcept -> Float {
                // Match Cycles' frac(): truncation followed by an explicit
                // negative correction (including negative integers).
                index =
                    cast<int>(coordinate) -
                    select(0, 1, coordinate < 0.0f);
                return coordinate - cast<float>(index);
            };
        const auto wrap_periodic =
            [](Int coordinate, Int extent) noexcept -> Int {
                auto wrapped = coordinate % extent;
                return select(
                    wrapped,
                    wrapped + extent,
                    wrapped < 0);
            };
        const auto wrap_mirror =
            [](Int coordinate, Int extent) noexcept -> Int {
                auto adjusted =
                    coordinate +
                    select(0, 1, coordinate < 0);
                auto period = abs(adjusted) % (2 * extent);
                return select(
                    period,
                    2 * extent - period - 1,
                    period >= extent);
            };
        const auto read_clip =
            [&](Int x, Int y) noexcept -> Float4 {
                Float4 value = make_float4(0.0f);
                $if ((x >= 0) & (x < width) &
                     (y >= 0) & (y < height)) {
                    value = texture.read(make_uint2(
                        cast<uint>(x), cast<uint>(y)));
                };
                return value;
            };
        const auto wrap_coordinate =
            [&](Int coordinate, Int extent) noexcept -> Int {
                if (extension == 0u) {
                    return wrap_periodic(coordinate, extent);
                }
                if (extension == 2u) {
                    return clamp(coordinate, 0, extent - 1);
                }
                if (extension == 3u) {
                    return wrap_mirror(coordinate, extent);
                }
                return coordinate;
            };
        const auto read_wrapped =
            [&](Int x, Int y) noexcept -> Float4 {
                return read_clip(
                    wrap_coordinate(x, width),
                    wrap_coordinate(y, height));
            };

        auto coordinate = def(uv);
        if (interpolation == 0u) {
            Int x;
            Int y;
            static_cast<void>(split_coordinate(
                coordinate.x * cast<float>(width), x));
            static_cast<void>(split_coordinate(
                coordinate.y * cast<float>(height), y));
            return read_wrapped(x, y);
        }

        Int x;
        Int y;
        auto tx = split_coordinate(
            coordinate.x * cast<float>(width) - 0.5f, x);
        auto ty = split_coordinate(
            coordinate.y * cast<float>(height) - 0.5f, y);

        if (interpolation == 1u) {
            auto x1 = x + 1;
            auto y1 = y + 1;
            auto row0 =
                (1.0f - tx) * read_wrapped(x, y) +
                tx * read_wrapped(x1, y);
            auto row1 =
                (1.0f - tx) * read_wrapped(x, y1) +
                tx * read_wrapped(x1, y1);
            return (1.0f - ty) * row0 + ty * row1;
        }

        // Cycles treats both Cubic and Smart as cubic. These are the exact
        // cubic B-spline weights used by its CPU and GPU image paths.
        auto cubic_weights = [](Float t) noexcept {
            return std::array<Float, 4u>{
                (((-1.0f / 6.0f) * t + 0.5f) * t -
                 0.5f) *
                        t +
                    (1.0f / 6.0f),
                ((0.5f * t - 1.0f) * t) * t +
                    (2.0f / 3.0f),
                ((-0.5f * t + 0.5f) * t + 0.5f) *
                        t +
                    (1.0f / 6.0f),
                (1.0f / 6.0f) * t * t * t};
        };
        auto wx = cubic_weights(tx);
        auto wy = cubic_weights(ty);
        auto cubic_row = [&](Int row) noexcept {
            return wx[0u] * read_wrapped(x - 1, row) +
                   wx[1u] * read_wrapped(x, row) +
                   wx[2u] * read_wrapped(x + 1, row) +
                   wx[3u] * read_wrapped(x + 2, row);
        };
        return wy[0u] * cubic_row(y - 1) +
               wy[1u] * cubic_row(y) +
               wy[2u] * cubic_row(y + 1) +
               wy[3u] * cubic_row(y + 2);
    }

    [[nodiscard]] ShaderAttribute attribute(
        Expr<std::uint64_t> attribute_id,
        const SurfacePoint &point) const noexcept override {
        auto result = ShaderAttribute::missing();
        // Attribute cardinality is scene data, not shader structure. Look up
        // only the current geometry's compact range so AST/XIR size remains
        // constant as scenes add meshes, UV maps, or color attributes.
        $if (point.geometry_index != ~0u) {
            Var<AttributeRangeGpu> range =
                _geometry_heap
                    ->template buffer<AttributeRangeGpu>(
                        _attribute_range_slot)
                    .read(point.geometry_index);
            UInt local_index = 0u;
            Bool found = false;
            $while ((local_index < range.count) & !found) {
                Var<AttributeBindingGpu> binding =
                    _geometry_heap
                        ->template buffer<
                            AttributeBindingGpu>(
                            _attribute_binding_slot)
                        .read(range.offset + local_index);
                $if (attribute_id == binding.id) {
                    Var<Triangle> triangle =
                        _geometry_heap
                            ->template buffer<Triangle>(
                                range.triangle_slot)
                            .read(point.primitive_id);
                    UInt i0 = triangle.i0;
                    UInt i1 = triangle.i1;
                    UInt i2 = triangle.i2;
                    $if (binding.domain ==
                         attribute_domain_corner) {
                        const auto corner =
                            point.primitive_id * 3u;
                        i0 = corner;
                        i1 = corner + 1u;
                        i2 = corner + 2u;
                    };
                    $if (binding.domain ==
                         attribute_domain_face) {
                        i0 = point.primitive_id;
                        i1 = point.primitive_id;
                        i2 = point.primitive_id;
                    };
                    auto v0 =
                        _geometry_heap
                            ->template buffer<
                                luisa::float4>(
                                binding.value_slot)
                            .read(i0);
                    auto v1 =
                        _geometry_heap
                            ->template buffer<
                                luisa::float4>(
                                binding.value_slot)
                            .read(i1);
                    auto v2 =
                        _geometry_heap
                            ->template buffer<
                                luisa::float4>(
                                binding.value_slot)
                            .read(i2);
                    result.value = triangle_interpolate(
                        point.barycentric, v0, v1, v2);
                    found = true;
                };
                local_index += 1u;
            };
            result.found = found;
        };
        return result;
    }

    [[nodiscard]] Float parameter_float(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _scalar_parameters->read(block + slot);
    }

    [[nodiscard]] Float3 parameter_float3(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _vector_parameters->read(block + slot);
    }

    [[nodiscard]] Float cycles_bsdf_data(
        Expr<std::uint32_t> index) const noexcept override {
        return _cycles_bsdf_tables->read(index);
    }

    [[nodiscard]] Float3 xyz_to_rgb(
        Expr<luisa::float3> xyz_expression)
        const noexcept override {
        Float3 xyz{xyz_expression};
        return make_float3(
            dot(
                make_float3(
                    _shader_color_space.xyz_to_r.x,
                    _shader_color_space.xyz_to_r.y,
                    _shader_color_space.xyz_to_r.z),
                xyz),
            dot(
                make_float3(
                    _shader_color_space.xyz_to_g.x,
                    _shader_color_space.xyz_to_g.y,
                    _shader_color_space.xyz_to_g.z),
                xyz),
            dot(
                make_float3(
                    _shader_color_space.xyz_to_b.x,
                    _shader_color_space.xyz_to_b.y,
                    _shader_color_space.xyz_to_b.z),
                xyz));
    }

    [[nodiscard]] Float3 rec709_to_rgb(
        Expr<luisa::float3> rec709_expression)
        const noexcept override {
        Float3 rec709{rec709_expression};
        return make_float3(
            dot(
                make_float3(
                    _shader_color_space.rec709_to_r.x,
                    _shader_color_space.rec709_to_r.y,
                    _shader_color_space.rec709_to_r.z),
                rec709),
            dot(
                make_float3(
                    _shader_color_space.rec709_to_g.x,
                    _shader_color_space.rec709_to_g.y,
                    _shader_color_space.rec709_to_g.z),
                rec709),
            dot(
                make_float3(
                    _shader_color_space.rec709_to_b.x,
                    _shader_color_space.rec709_to_b.y,
                    _shader_color_space.rec709_to_b.z),
                rec709));
    }

    [[nodiscard]] Float3 nishita_sky(
        Expr<std::uint32_t> block,
        std::uint32_t sky_index,
        Expr<luisa::float3> direction_expression,
        Expr<float> sun_elevation_expression,
        Expr<float> sun_rotation_expression,
        Expr<float> angular_diameter_expression,
        Expr<float> sun_intensity_expression)
        const noexcept override {
        Float3 result = make_float3(0.0f);
        Float3 direction{direction_expression};
        Float sun_elevation{sun_elevation_expression};
        Float sun_rotation{sun_rotation_expression};
        Float angular_diameter{angular_diameter_expression};
        Float sun_intensity{sun_intensity_expression};
        for (const auto &binding : _nishita_textures) {
            if (binding.sky_index != sky_index) {
                continue;
            }
            $if (block == binding.parameter_block) {
                const auto sun_direction = make_float3(
                    -cos(sun_elevation) * sin(sun_rotation),
                    cos(sun_elevation) * cos(sun_rotation),
                    sin(sun_elevation));
                const auto sky_xyz =
                    cycles_nishita::sky_radiance_xyz(
                        _textures->tex2d(binding.texture_slot),
                        direction,
                        sun_rotation);
                const auto sun_xyz =
                    cycles_nishita::sun_disc_radiance_xyz(
                        direction,
                        sun_direction,
                        make_float3(binding.pixel_bottom_xyz),
                        make_float3(binding.pixel_top_xyz),
                        sun_elevation,
                        angular_diameter,
                        sun_intensity);
                result = max(
                    xyz_to_rgb(sky_xyz + sun_xyz),
                    make_float3(0.0f));
            };
        }
        return result;
    }
};


}// namespace psycles::luisa_backend::detail
