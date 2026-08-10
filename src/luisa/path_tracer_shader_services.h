#pragma once

#include "cycles_texture_sampling.h"
#include "path_tracer_color_transforms.h"
#include "path_tracer_internal.h"
#include "path_tracer_normal_maps.h"
#include "path_tracer_shader_tables.h"
#include "path_tracer_texture_sampling.h"
#include "path_tracer_vector_mapping.h"

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

    [[nodiscard]] ULong parameter_uint64(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _vector_parameters->read(block + slot)
            .xy()
            .template bitcast<luisa::ulong>();
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
    const SurfaceClosureSetupProvider *_surface_closure_setup_provider{};
    const CallableTexture2DSamplingProvider *_texture_sampling_provider{};
    CallableSurfaceColorTransformProvider _color_transform_provider;
    CallableSurfaceVectorMappingProvider _vector_mapping_provider;
    CallableSurfaceNormalMapProvider _normal_map_provider;
    CallableSurfaceShaderTableProvider<ScalarParameterBuffer>
        _shader_table_provider;

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
            &shader_color_space,
        const SurfaceClosureSetupProvider
            *surface_closure_setup_provider = nullptr,
        const CallableTexture2DSamplingProvider
            *texture_sampling_provider = nullptr) noexcept
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
          _shader_color_space{shader_color_space},
          _surface_closure_setup_provider{
              surface_closure_setup_provider},
          _texture_sampling_provider{
              texture_sampling_provider},
          _shader_table_provider{scalar_parameters} {}

    [[nodiscard]] const SurfaceClosureSetupProvider *
    surface_closure_setup_provider() const noexcept override {
        return _surface_closure_setup_provider;
    }

    [[nodiscard]] const SurfaceColorTransformProvider *
    surface_color_transform_provider() const noexcept override {
        return &_color_transform_provider;
    }

    [[nodiscard]] const SurfaceVectorMappingProvider *
    surface_vector_mapping_provider() const noexcept override {
        return &_vector_mapping_provider;
    }

    [[nodiscard]] const SurfaceNormalMapProvider *
    surface_normal_map_provider() const noexcept override {
        return &_normal_map_provider;
    }

    [[nodiscard]] const SurfaceShaderTableProvider *
    surface_shader_table_provider() const noexcept override {
        return &_shader_table_provider;
    }

    [[nodiscard]] Float4 texture_2d(
        Expr<std::uint32_t> handle,
        Expr<luisa::float2> uv,
        Expr<luisa::float2>,
        Expr<luisa::float2>,
        std::uint32_t interpolation,
        std::uint32_t extension) const noexcept override {
        if (_texture_sampling_provider != nullptr) {
            return _texture_sampling_provider->sample(
                handle, uv, interpolation, extension);
        }
        return sample_cycles_texture_2d(
            _textures,
            handle,
            uv,
            interpolation,
            extension);
    }

    [[nodiscard]] ShaderAttribute attribute(
        Expr<luisa::ulong> attribute_id,
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

    [[nodiscard]] ULong parameter_uint64(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _vector_parameters->read(block + slot)
            .xy()
            .template bitcast<luisa::ulong>();
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
