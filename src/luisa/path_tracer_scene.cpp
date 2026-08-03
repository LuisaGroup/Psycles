#include "path_tracer_internal.h"
#include "graph_surface_value_expression.h"
#include "path_tracer_environment.h"
#include "path_tracer_generated_coordinates.h"
#include "path_tracer_image_decode.h"
#include "path_tracer_scene_geometry.h"
#include "path_tracer_shader_services.h"
#include "path_tracer_surfaces.h"
#include "path_tracer_volume_capabilities.h"
#include "path_tracer_volume_majorant_scene.h"
#include "cycles_shader_identity.h"

#include <psycles/compiler/core_nodes.h>
#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_nishita.h>
#include <psycles/sampling/light_distribution.h>

#include "cycles_shader_tables_4_5_10.inl"

namespace psycles::luisa_backend {

using namespace detail;

namespace {

[[nodiscard]] constexpr std::uint32_t encode_attribute_domain(
    contract::MeshAttributeDomain domain) noexcept {
    switch (domain) {
        case contract::MeshAttributeDomain::point:
            return attribute_domain_point;
        case contract::MeshAttributeDomain::corner:
            return attribute_domain_corner;
        case contract::MeshAttributeDomain::face:
            return attribute_domain_face;
    }
    return attribute_domain_point;
}

}// namespace

contract::SceneCompilation LuisaPathTracerBackend::compile_scene(
    const SceneSnapshot &snapshot) {
    contract::SceneCompilation result;
    if (!_device) {
        diagnose(result.diagnostics, "Luisa device is invalid.");
        return result;
    }
    if (!snapshot.active_camera) {
        diagnose(result.diagnostics, "Scene has no active camera.");
        return result;
    }
    const auto camera_iter =
        snapshot.cameras.find(*snapshot.active_camera);
    if (camera_iter == snapshot.cameras.end()) {
        diagnose(
            result.diagnostics,
            "Active camera does not exist in the scene.");
        return result;
    }
    for (const auto &[id, instance] : snapshot.instances) {
        static_cast<void>(id);
        if (!instance.motion.empty()) {
            diagnose(
                result.diagnostics,
                "Luisa vertical slice does not yet accept instance "
                "motion; the scene must be exported at a fixed frame.");
        }
        if (instance.visibility_mask !=
                ~std::uint32_t{0u} &&
            instance.visibility_mask > 0xffu) {
            diagnose(
                result.diagnostics,
                "Luisa ray visibility masks are eight-bit; the scene "
                "contains a non-default mask outside that range.");
        }
    }
    std::set<contract::MaterialId>
        volume_materials;
    for (const auto &[id, material] : snapshot.materials) {
        if (material.shader.root(
                contract::ShaderDomain::volume)) {
            volume_materials.emplace(id);
        }
    }
    const auto volume_metadata =
        VolumeSceneMetadataComponent{}.analyze(
            snapshot, volume_materials);
    if (!result.diagnostics.empty()) {
        return result;
    }

    auto data = std::make_shared<LuisaSceneData>();
    data->device =
        luisa::compute::Device{_device.impl_shared()};
    data->revision = snapshot.revision;
    data->camera = camera_iter->second;
    data->volume_metadata = volume_metadata;
    data->shader_color_space = snapshot.shader_color_space;
    data->cycles_background_object_index =
        snapshot.cycles_background_object_index
            .value_or(
                cycles_shader_identity::
                    invalid_index);
    data->cycles_background_light_group =
        snapshot.cycles_background_light_group;
    data->world_visibility_mask =
        snapshot.world_visibility_mask;
    data->cycles_background_shader_flags =
        cycles_shader_identity::background_light_flags(
            snapshot.world_cast_shadow,
            snapshot.world_visibility_mask);

    ShaderCompiler shader_compiler{
        compiler::make_core_node_registry()};
    auto material_update =
        data->materials.update(snapshot, shader_compiler);
    if (!material_update.committed) {
        for (const auto &diagnostic :
             material_update.diagnostics) {
            diagnose(
                result.diagnostics,
                "Material " +
                    std::to_string(diagnostic.material.value) +
                    ": " + diagnostic.message);
        }
        return result;
    }
    const VolumeProgramCapabilityComponent
        volume_capabilities;
    data->volume_metadata
        .closure_allocation_budget =
        cycles_scene_closure_allocation_budget(
            data->materials);

    luisa::vector<float> scalar_parameters;
    luisa::vector<luisa::float3> vector_parameters;
    std::vector<std::uint32_t>
        volume_surface_flags;
    std::map<std::uint64_t, std::uint32_t>
        surface_tags_by_signature;
    for (const auto &[id, material] :
         data->materials.materials()) {
        const auto base = static_cast<std::uint32_t>(
            scalar_parameters.size());
        const auto signature =
            material.surface_program()->structure_signature();
        auto [surface_iter, inserted] =
            surface_tags_by_signature.try_emplace(signature, 0u);
        if (inserted) {
            surface_iter->second =
                data->surfaces.create<GraphSurface>(
                    material.surface_program());
        }
        const auto capabilities =
            data->surfaces.capabilities(
                surface_iter->second);
        const auto emission_estimate =
            compiler::estimate_surface_emission(
                *material.surface_program(),
                material.parameters());
        const auto may_emit =
            emission_estimate != Vec3f{};
        const auto emission_is_constant =
            material.surface_program()
                ->emission_evaluation() !=
            compiler::EmissionEvaluationMode::deferred;
        if (capabilities.may_have_volume) {
            volume_capabilities
                .merge_surface_flags(
                    volume_surface_flags,
                    surface_iter->second,
                    *material.surface_program());
        }
        data->material_bindings.emplace(
            id,
            MaterialBinding{
                .surface_tag = surface_iter->second,
                .parameter_block = base,
                .cycles_shader_index =
                    snapshot.materials.at(id)
                        .cycles_shader_index
                        .value_or(
                            cycles_shader_identity::
                                invalid_index),
                .material_identity =
                    static_cast<std::uint32_t>(
                        data->material_bindings
                            .size()),
                .flags =
                    (capabilities.may_have_volume
                         ? material_flag_has_volume
                         : 0u) |
                    (may_emit
                         ? material_flag_may_emit
                         : 0u) |
                    (emission_is_constant
                         ? material_flag_constant_emission
                         : 0u) |
                    (snapshot.materials.at(id)
                            .use_bump_map_correction
                         ? material_flag_use_bump_map_correction
                         : 0u),
                .volume_sampling =
                    snapshot.materials.at(id)
                        .volume_sampling});
        const auto &program = *material.surface_program();
        const auto scalar_parameter =
            [&](compiler::ValueExpressionId expression)
            -> std::optional<float> {
            if (!expression.valid() ||
                expression.value >=
                    program.value_instructions().size()) {
                return std::nullopt;
            }
            const auto &source =
                program.value_instructions()[expression.value];
            if (source.operation !=
                    compiler::ValueOperation::parameter ||
                !source.parameter.valid()) {
                return std::nullopt;
            }
            const auto *value =
                material.parameters().find(source.parameter);
            if (value == nullptr ||
                value->type !=
                    contract::SocketType::floating) {
                return std::nullopt;
            }
            return std::get<float>(value->value);
        };
        for (const auto &instruction :
             program.value_instructions()) {
            if (instruction.operation !=
                compiler::ValueOperation::nishita_sky) {
                continue;
            }
            const std::array expressions{
                instruction.a,
                instruction.b,
                instruction.c,
                instruction.d,
                instruction.e,
                instruction.f,
                instruction.g,
                instruction.h};
            std::array<float, expressions.size()> values{};
            auto static_parameters = true;
            for (std::size_t i = 0u;
                 i < expressions.size();
                 ++i) {
                const auto value =
                    scalar_parameter(expressions[i]);
                if (!value) {
                    static_parameters = false;
                    break;
                }
                values[i] = *value;
            }
            if (!static_parameters) {
                diagnose(
                    result.diagnostics,
                    "Material '" + material.name() +
                        "' drives Nishita atmosphere properties from "
                        "runtime graph expressions. Cycles treats these "
                        "properties as precompute inputs; Psycles refuses "
                        "to replace them with an analytic approximation.");
                continue;
            }
            data->nishita_texture_bindings.emplace_back(
                NishitaTextureBinding{
                    .parameter_block = base,
                    .sky_index = static_cast<std::uint32_t>(
                        instruction.static_u0),
                    .texture_slot = 0u,
                    .parameters =
                        contract::NishitaSkyDesc{
                            .sun_elevation = values[0u],
                            .sun_rotation = values[1u],
                            .angular_diameter = values[2u],
                            .sun_intensity = values[3u],
                            .altitude = values[4u],
                            .air_density = values[5u],
                            .dust_density = values[6u],
                            .ozone_density = values[7u],
                            .background_strength = 1.0f}});
        }
        for (const auto &parameter :
             material.surface_program()->parameters()) {
            const auto *value =
                material.parameters().find(parameter.id);
            auto scalar = 0.0f;
            auto vector = luisa::make_float3(0.0f);
            if (value != nullptr) {
                switch (surface_value_category(parameter.type)) {
                    case SurfaceValueCategory::scalar:
                        scalar = scalar_parameter_value(*value);
                        break;
                    case SurfaceValueCategory::vector:
                        vector = vector_parameter_value(*value);
                        break;
                }
            }
            scalar_parameters.emplace_back(scalar);
            vector_parameters.emplace_back(vector);
        }
    }
    if (!result.diagnostics.empty()) {
        return result;
    }
    if (snapshot.world_shader) {
        auto iter =
            data->material_bindings.find(*snapshot.world_shader);
        if (iter != data->material_bindings.end()) {
            data->world_surface = iter->second;
            if (iter->second.cycles_shader_index !=
                cycles_shader_identity::invalid_index) {
                data->cycles_background_shader_id =
                    cycles_shader_identity::background_light(
                        iter->second.cycles_shader_index,
                        snapshot.world_cast_shadow,
                        snapshot.world_visibility_mask);
            }
        }
    }
    if (scalar_parameters.empty()) {
        scalar_parameters.emplace_back(0.0f);
        vector_parameters.emplace_back(luisa::make_float3(0.0f));
    }
    data->scalar_parameter_buffer =
        data->device.create_buffer<float>(
            scalar_parameters.size());
    data->vector_parameter_buffer =
        data->device.create_buffer<luisa::float3>(
            vector_parameters.size());
    data->volume_surface_flag_count =
        static_cast<std::uint32_t>(
            volume_surface_flags.size());
    if (volume_surface_flags.empty()) {
        volume_surface_flags.emplace_back(0u);
    }
    data->volume_surface_flag_buffer =
        data->device.create_buffer<luisa::uint>(
            volume_surface_flags.size());

    using namespace cycles45_tables;
    static_assert(
        std::size(table_ggx_E) == ggx_e_size);
    static_assert(
        std::size(table_ggx_Eavg) == ggx_eavg_size);
    static_assert(
        std::size(table_ggx_glass_E) ==
        ggx_glass_e_size);
    static_assert(
        std::size(table_ggx_glass_Eavg) ==
        ggx_glass_eavg_size);
    static_assert(
        std::size(table_ggx_glass_inv_E) ==
        ggx_glass_inv_e_size);
    static_assert(
        std::size(table_ggx_glass_inv_Eavg) ==
        ggx_glass_inv_eavg_size);
    static_assert(
        std::size(table_sheen_ltc) == sheen_ltc_size);
    static_assert(
        std::size(table_ggx_gen_schlick_ior_s) ==
        ggx_gen_schlick_ior_s_size);
    static_assert(
        std::size(table_ggx_gen_schlick_s) ==
        ggx_gen_schlick_s_size);

    luisa::vector<float> cycles_bsdf_values;
    cycles_bsdf_values.reserve(total_size);
    const auto append_cycles_table =
        [&cycles_bsdf_values](const auto &table) noexcept {
            for (const auto value : table) {
                cycles_bsdf_values.emplace_back(value);
            }
        };
    append_cycles_table(table_ggx_E);
    append_cycles_table(table_ggx_Eavg);
    append_cycles_table(table_ggx_glass_E);
    append_cycles_table(table_ggx_glass_Eavg);
    append_cycles_table(table_ggx_glass_inv_E);
    append_cycles_table(table_ggx_glass_inv_Eavg);
    append_cycles_table(table_sheen_ltc);
    append_cycles_table(table_ggx_gen_schlick_ior_s);
    append_cycles_table(table_ggx_gen_schlick_s);
    if (cycles_bsdf_values.size() != total_size) {
        diagnose(
            result.diagnostics,
            "Internal Cycles BSDF table layout mismatch.");
        return result;
    }
    data->cycles_bsdf_table_buffer =
        data->device.create_buffer<float>(
            cycles_bsdf_values.size());

    std::size_t attribute_count = 0u;
    for (const auto &[id, geometry] :
         snapshot.geometries) {
        static_cast<void>(id);
        attribute_count +=
            geometry.color_attributes.size() +
            geometry.uv_layers.size() +
            geometry.uv_tangent_layers.size();
    }
    const auto fixed_geometry_slots =
        snapshot.geometries.size() *
        geometry_bindless_stride;
    const auto attribute_binding_slot =
        fixed_geometry_slots +
        attribute_count;
    const auto attribute_range_slot =
        attribute_binding_slot + 1u;
    const auto bindless_slots =
        std::max<std::size_t>(
            attribute_range_slot + 1u,
            1u);
    data->attribute_binding_slot =
        static_cast<std::uint32_t>(
            attribute_binding_slot);
    data->attribute_range_slot =
        static_cast<std::uint32_t>(
            attribute_range_slot);
    data->heap =
        data->device.create_bindless_array(bindless_slots);
    data->geometries.reserve(snapshot.geometries.size());
    std::vector<GeometryUpload> uploads;
    uploads.reserve(snapshot.geometries.size());
    std::map<contract::GeometryId, std::uint32_t>
        geometry_indices;
    std::map<contract::GeometryId, std::uint32_t>
        cycles_primitive_offsets;
    CyclesPrimitiveIntervalResolver
        cycles_primitive_intervals;
    luisa::vector<GeometryGpu> geometry_gpu;
    luisa::vector<AttributeBindingGpu>
        attribute_bindings;
    attribute_bindings.reserve(attribute_count);
    luisa::vector<AttributeRangeGpu>
        attribute_ranges;
    attribute_ranges.reserve(
        snapshot.geometries.size());
    luisa::vector<MaterialBindingGpu> geometry_materials;
    auto next_attribute_slot =
        static_cast<std::uint32_t>(
            fixed_geometry_slots);
    Stream stream = data->device.create_stream();
    stream << data->scalar_parameter_buffer.copy_from(
                  luisa::span{scalar_parameters})
           << data->vector_parameter_buffer.copy_from(
                  luisa::span{vector_parameters})
           << data->cycles_bsdf_table_buffer.copy_from(
                  luisa::span{cycles_bsdf_values})
           << data->volume_surface_flag_buffer.copy_from(
                  luisa::span{volume_surface_flags});

    std::size_t texture_slot_count = 1u;
    for (const auto &[image_id, image] : snapshot.images) {
        static_cast<void>(image);
        texture_slot_count = std::max(
            texture_slot_count,
            static_cast<std::size_t>(image_id.value) + 1u);
    }
    for (auto &binding :
         data->nishita_texture_bindings) {
        binding.texture_slot =
            static_cast<std::uint32_t>(texture_slot_count);
        ++texture_slot_count;
    }
    if (snapshot.environment) {
        const auto &environment = *snapshot.environment;
        if (environment.nishita) {
            const auto &sky = *environment.nishita;
            data->environment_width =
                cycles_nishita::lut_width;
            data->environment_height =
                cycles_nishita::lut_height;
            const auto cosine_elevation =
                std::cos(sky.sun_elevation);
            data->nishita_environment =
                NishitaEnvironmentRuntime{
                    .parameters = sky,
                    .pixel_bottom_xyz = {},
                    .pixel_top_xyz = {},
                    .sun_direction =
                        luisa::make_float3(
                            -cosine_elevation *
                                std::sin(sky.sun_rotation),
                            cosine_elevation *
                                std::cos(sky.sun_rotation),
                            std::sin(sky.sun_elevation)),
                    .angular_radius =
                        std::max(
                            sky.angular_diameter * 0.5f,
                            0.0f)};
        } else {
            const auto expected_pixels =
                static_cast<std::size_t>(environment.width) *
                static_cast<std::size_t>(environment.height);
            if (environment.width == 0u ||
                environment.height == 0u ||
                environment.pixels.size() != expected_pixels) {
                diagnose(
                    result.diagnostics,
                    "Environment '" + environment.name +
                        "' has an invalid precomputed texture payload.");
                return result;
            }
            data->environment_width = environment.width;
            data->environment_height = environment.height;
        }
        data->environment_texture_slot =
            static_cast<std::uint32_t>(texture_slot_count);
        data->environment_suns = environment.suns;
        ++texture_slot_count;
    }
    data->texture_heap =
        data->device.create_bindless_array(texture_slot_count);
    data->images.reserve(
        snapshot.images.size() +
        data->nishita_texture_bindings.size() +
        (snapshot.environment ? 2u : 1u));
    std::vector<luisa::vector<std::byte>> texture_uploads;
    texture_uploads.reserve(snapshot.images.size() + 1u);
    std::vector<luisa::vector<luisa::float4>>
        float_texture_uploads;
    float_texture_uploads.reserve(
        snapshot.environment ? 1u : 0u);

    auto &dummy_pixels = texture_uploads.emplace_back(4u);
    dummy_pixels[0u] = std::byte{255u};
    dummy_pixels[1u] = std::byte{0u};
    dummy_pixels[2u] = std::byte{255u};
    dummy_pixels[3u] = std::byte{255u};
    auto &dummy_image = data->images.emplace_back(
        data->device.create_image<float>(
            luisa::compute::PixelStorage::BYTE4, 1u, 1u));
    data->texture_heap.emplace_on_update(
        0u,
        dummy_image,
        luisa::compute::Sampler::linear_point_repeat());
    stream << dummy_image.copy_from(
        luisa::span{dummy_pixels});

    for (const auto &[image_id, image] : snapshot.images) {
        if (image.encoded_data.empty() ||
            image.encoded_data.size() >
                static_cast<std::size_t>(
                    std::numeric_limits<int>::max())) {
            diagnose(
                result.diagnostics,
                "Image '" + image.name +
                    "' has no decodable payload.");
            continue;
        }
        auto decoded = decode_scene_image_rgba8(
            image.encoded_data, image.name);
        if (!decoded) {
            diagnose(
                result.diagnostics,
                "Failed to decode image '" + image.name + "'.");
            continue;
        }
        const auto pixel_bytes = decoded->pixels.size();
        auto &pixels =
            texture_uploads.emplace_back(pixel_bytes);
        std::memcpy(
            pixels.data(), decoded->pixels.data(), pixel_bytes);
        if (image.alpha_type ==
                contract::ImageAlphaType::straight &&
            image.color_space !=
                contract::ImageColorSpace::data) {
            for (std::size_t offset = 0u;
                 offset < pixel_bytes;
                 offset += 4u) {
                const auto alpha =
                    static_cast<std::uint32_t>(
                        std::to_integer<std::uint8_t>(
                            pixels[offset + 3u]));
                for (std::size_t channel = 0u;
                     channel < 3u;
                     ++channel) {
                    const auto value =
                        static_cast<std::uint32_t>(
                            std::to_integer<std::uint8_t>(
                                pixels[offset + channel]));
                    pixels[offset + channel] =
                        static_cast<std::byte>(
                            (value * alpha) / 255u);
                }
            }
        } else if (
            image.alpha_type ==
            contract::ImageAlphaType::ignore) {
            for (std::size_t offset = 3u;
                 offset < pixel_bytes;
                 offset += 4u) {
                pixels[offset] =
                    static_cast<std::byte>(255u);
            }
        }

        auto &resource = data->images.emplace_back(
            data->device.create_image<float>(
                luisa::compute::PixelStorage::BYTE4,
                decoded->width,
                decoded->height));
        data->texture_heap.emplace_on_update(
            static_cast<std::uint32_t>(image_id.value),
            resource,
            luisa::compute::Sampler::linear_point_repeat());
        stream << resource.copy_from(luisa::span{pixels});
    }
    const auto has_nishita_environment =
        snapshot.environment &&
        snapshot.environment->nishita.has_value() &&
        data->environment_texture_slot.has_value();
    if (!data->nishita_texture_bindings.empty() ||
        has_nishita_environment) {
        Kernel2D precompute_nishita =
            [](ImageFloat image,
               BufferFloat4 sun_pixels,
               Float sun_elevation,
               Float angular_diameter,
               Float altitude,
               Float air_density,
               Float dust_density,
               Float ozone_density) noexcept {
                set_block_size(8u, 8u, 1u);
                UInt2 coordinate = dispatch_id().xy();
                Float3 xyz =
                    cycles_nishita::sky_lut_texel(
                        coordinate.x,
                        coordinate.y,
                        sun_elevation,
                        altitude,
                        air_density,
                        dust_density,
                        ozone_density);
                Float4 pixel = make_float4(xyz, 1.0f);
                image.write(coordinate, pixel);
                image.write(
                    make_uint2(
                        cycles_nishita::lut_width -
                            coordinate.x - 1u,
                        coordinate.y),
                    pixel);
                $if ((coordinate.x == 0u) &
                     (coordinate.y == 0u)) {
                    const auto diameter =
                        max(abs(angular_diameter), 1.0e-7f);
                    const auto values =
                        cycles_nishita::sun_pixels(
                            sun_elevation,
                            diameter,
                            altitude,
                            air_density,
                            dust_density);
                    sun_pixels.write(
                        0u,
                        make_float4(
                            values.bottom_xyz, 1.0f));
                    sun_pixels.write(
                        1u,
                        make_float4(
                            values.top_xyz, 1.0f));
                };
            };
        auto precompute_shader =
            data->device.compile(precompute_nishita);
        auto sun_output =
            data->device.create_buffer<luisa::float4>(2u);
        luisa::vector<luisa::float4> sun_readback(2u);
        const auto precompute =
            [&](std::uint32_t texture_slot,
                const contract::NishitaSkyDesc &sky,
                luisa::float3 &pixel_bottom,
                luisa::float3 &pixel_top) {
                auto &resource = data->images.emplace_back(
                    data->device.create_image<float>(
                        luisa::compute::PixelStorage::FLOAT4,
                        cycles_nishita::lut_width,
                        cycles_nishita::lut_height));
                data->texture_heap.emplace_on_update(
                    texture_slot,
                    resource,
                    luisa::compute::Sampler::
                        linear_point_edge());
                stream
                    << precompute_shader(
                           resource,
                           sun_output,
                           sky.sun_elevation,
                           sky.angular_diameter,
                           sky.altitude,
                           sky.air_density,
                           sky.dust_density,
                           sky.ozone_density)
                           .dispatch(
                               cycles_nishita::
                                   half_lut_width,
                               cycles_nishita::lut_height)
                    << sun_output.copy_to(
                           luisa::span{sun_readback})
                    << synchronize();
                pixel_bottom = sun_readback[0u].xyz();
                pixel_top = sun_readback[1u].xyz();
            };
        for (auto &binding :
             data->nishita_texture_bindings) {
            precompute(
                binding.texture_slot,
                binding.parameters,
                binding.pixel_bottom_xyz,
                binding.pixel_top_xyz);
        }
        if (has_nishita_environment) {
            precompute(
                *data->environment_texture_slot,
                *snapshot.environment->nishita,
                data->nishita_environment
                    ->pixel_bottom_xyz,
                data->nishita_environment
                    ->pixel_top_xyz);
        }
    }
    if (snapshot.environment &&
        data->environment_texture_slot &&
        !snapshot.environment->nishita) {
        const auto &environment = *snapshot.environment;
        auto &pixels =
            float_texture_uploads.emplace_back();
        pixels.reserve(environment.pixels.size());
        for (const auto value : environment.pixels) {
            pixels.emplace_back(
                luisa::make_float4(
                    value.x, value.y, value.z, 1.0f));
        }
        auto &resource = data->images.emplace_back(
            data->device.create_image<float>(
                luisa::compute::PixelStorage::FLOAT4,
                environment.width,
                environment.height));
        data->texture_heap.emplace_on_update(
            *data->environment_texture_slot,
            resource,
            luisa::compute::Sampler::
                linear_point_repeat());
        stream << resource.copy_from(luisa::span{pixels});
    }
    if (!result.diagnostics.empty()) {
        return result;
    }

    for (const auto &[geometry_id, geometry] :
         snapshot.geometries) {
        if (geometry.positions.empty() ||
            geometry.triangles.empty()) {
            diagnose(
                result.diagnostics,
                "Geometry '" + geometry.name +
                    "' has no triangles.");
            continue;
        }
        const auto cycles_primitive_interval =
            cycles_primitive_intervals.resolve(
                geometry.triangles.size(),
                geometry.cycles_primitive_offset);
        if (!cycles_primitive_interval.offset) {
            diagnose(
                result.diagnostics,
                "Geometry '" + geometry.name +
                    "' has an overlapping or out-of-range Cycles "
                    "primitive interval.");
            continue;
        }
        cycles_primitive_offsets.emplace(
            geometry_id,
            *cycles_primitive_interval.offset);
        auto &upload = uploads.emplace_back();
        upload.positions.reserve(geometry.positions.size());
        const auto generated_mapping =
            make_generated_coordinate_mapping(
                geometry);
        upload.generated_transform =
            to_luisa(
                generated_mapping
                    .object_to_generated);
        for (const auto position : geometry.positions) {
            upload.positions.emplace_back(
                to_luisa(position));
        }
        const auto require_vertex_or_corner =
            [&](std::string_view name,
                contract::MeshAttributeDomain domain) {
                if (domain !=
                        contract::MeshAttributeDomain::point &&
                    domain !=
                        contract::MeshAttributeDomain::corner) {
                    diagnose(
                        result.diagnostics,
                        "Geometry '" + geometry.name +
                            "' uses unsupported " +
                            std::string{name} +
                            " domain. Cycles primary triangle "
                            "attributes must be point or corner.");
                    return false;
                }
                return true;
            };
        const auto normal_domain =
            geometry.normals.values.empty()
                ? contract::MeshAttributeDomain::point
                : geometry.normals.domain;
        const auto uv_domain =
            geometry.uv.values.empty()
                ? contract::MeshAttributeDomain::point
                : geometry.uv.domain;
        const auto uv_tangent_domain =
            geometry.uv_tangents.values.empty()
                ? contract::MeshAttributeDomain::point
                : geometry.uv_tangents.domain;
        const auto generated_domain =
            geometry.generated.values.empty()
                ? contract::MeshAttributeDomain::point
                : geometry.generated.domain;
        if (!require_vertex_or_corner(
                "normal", normal_domain) ||
            !require_vertex_or_corner("UV", uv_domain) ||
            !require_vertex_or_corner(
                "UV tangent", uv_tangent_domain) ||
            !require_vertex_or_corner(
                "Generated", generated_domain)) {
            continue;
        }
        upload.attribute_domains =
            (normal_domain ==
                     contract::MeshAttributeDomain::corner
                 ? geometry_normal_corner
                 : 0u) |
            (uv_domain ==
                     contract::MeshAttributeDomain::corner
                 ? geometry_uv_corner
                 : 0u) |
            (uv_tangent_domain ==
                     contract::MeshAttributeDomain::corner
                 ? geometry_uv_tangent_corner
                 : 0u) |
            (generated_domain ==
                     contract::MeshAttributeDomain::corner
                 ? geometry_generated_corner
                 : 0u);
        if (geometry.normals.values.empty()) {
            upload.normals.assign(
                geometry.positions.size(),
                luisa::make_float3(0.0f));
        } else {
            upload.normals.reserve(
                geometry.normals.values.size());
            for (const auto value :
                 geometry.normals.values) {
                upload.normals.emplace_back(
                    to_luisa(value));
            }
        }
        if (geometry.uv.values.empty()) {
            upload.uv.assign(
                geometry.positions.size(),
                luisa::make_float2(0.0f));
        } else {
            upload.uv.reserve(geometry.uv.values.size());
            for (const auto value : geometry.uv.values) {
                upload.uv.emplace_back(to_luisa(value));
            }
        }
        if (geometry.uv_tangents.values.empty()) {
            upload.uv_tangents.assign(
                geometry.positions.size(),
                luisa::make_float4(0.0f));
        } else {
            upload.uv_tangents.reserve(
                geometry.uv_tangents.values.size());
            for (const auto value :
                 geometry.uv_tangents.values) {
                upload.uv_tangents.emplace_back(
                    luisa::make_float4(
                        value.x,
                        value.y,
                        value.z,
                        value.w));
            }
        }
        if (geometry.generated.values.empty()) {
            upload.generated.reserve(
                geometry.positions.size());
            for (const auto position :
                 geometry.positions) {
                upload.generated.emplace_back(
                    to_luisa(
                        generated_mapping.apply(
                            position)));
            }
        } else {
            upload.generated.reserve(
                geometry.generated.values.size());
            for (const auto value :
                 geometry.generated.values) {
                upload.generated.emplace_back(
                    to_luisa(value));
            }
        }
        upload.attributes.reserve(
            geometry.color_attributes.size() +
            geometry.uv_layers.size() +
            geometry.uv_tangent_layers.size());
        for (const auto &[name, source] :
             geometry.color_attributes) {
            auto &attribute =
                upload.attributes.emplace_back();
            attribute.id = contract::attribute_id(name);
            attribute.domain =
                encode_attribute_domain(source.domain);
            attribute.values.reserve(source.values.size());
            for (const auto value : source.values) {
                attribute.values.emplace_back(
                    luisa::make_float4(
                        value.x,
                        value.y,
                        value.z,
                        value.w));
            }
        }
        for (const auto &[name, source] :
             geometry.uv_layers) {
            auto &attribute =
                upload.attributes.emplace_back();
            attribute.id =
                contract::uv_attribute_id(name);
            attribute.domain =
                encode_attribute_domain(source.domain);
            attribute.values.reserve(source.values.size());
            for (const auto value : source.values) {
                attribute.values.emplace_back(
                    luisa::make_float4(
                        value.x, value.y, 0.0f, 0.0f));
            }
        }
        for (const auto &[name, source] :
             geometry.uv_tangent_layers) {
            auto &attribute =
                upload.attributes.emplace_back();
            attribute.id =
                contract::uv_tangent_attribute_id(name);
            attribute.domain =
                encode_attribute_domain(source.domain);
            attribute.values.reserve(source.values.size());
            for (const auto value : source.values) {
                attribute.values.emplace_back(
                    luisa::make_float4(
                        value.x,
                        value.y,
                        value.z,
                        value.w));
            }
        }
        upload.triangles.reserve(geometry.triangles.size());
        upload.triangle_material_slots.reserve(
            geometry.triangles.size());
        upload.triangle_random_per_island.reserve(
            geometry.triangles.size());
        upload.triangle_smooth.reserve(
            geometry.triangles.size());
        for (std::size_t i = 0u;
             i < geometry.triangles.size();
             ++i) {
            const auto triangle = geometry.triangles[i];
            if (triangle[0u] >= geometry.positions.size() ||
                triangle[1u] >= geometry.positions.size() ||
                triangle[2u] >= geometry.positions.size()) {
                diagnose(
                    result.diagnostics,
                    "Geometry '" + geometry.name +
                        "' contains an out-of-range index.");
                break;
            }
            upload.triangles.emplace_back(Triangle{
                triangle[0u],
                triangle[1u],
                triangle[2u]});
            upload.triangle_material_slots.emplace_back(
                i < geometry.triangle_material_slots.size()
                    ? geometry.triangle_material_slots[i]
                    : 0u);
            upload.triangle_random_per_island.emplace_back(
                i <
                        geometry.triangle_random_per_island
                            .size()
                    ? geometry
                          .triangle_random_per_island[i]
                    : 0.0f);
            upload.triangle_smooth.emplace_back(
                i < geometry.triangle_smooth.size() &&
                        geometry.triangle_smooth[i] != 0u
                    ? 1u
                    : 0u);
        }
        if (!result.diagnostics.empty()) {
            continue;
        }
        const auto index =
            static_cast<std::uint32_t>(
                data->geometries.size());
        const auto bindless_base =
            index * geometry_bindless_stride;
        const auto material_offset =
            static_cast<std::uint32_t>(
                geometry_materials.size());
        for (const auto material_id :
             geometry.material_slots) {
            const auto material_iter =
                data->material_bindings.find(material_id);
            if (material_iter ==
                data->material_bindings.end()) {
                diagnose(
                    result.diagnostics,
                    "Geometry '" + geometry.name +
                        "' references an unavailable material.");
                break;
            }
            geometry_materials.emplace_back(
                to_luisa(material_iter->second));
        }
        if (geometry.material_slots.empty()) {
            geometry_materials.emplace_back(
                MaterialBindingGpu{
                    .cycles_shader_index =
                        cycles_shader_identity::
                            invalid_index});
        }
        if (!result.diagnostics.empty()) {
            continue;
        }

        auto &resource =
            data->geometries.emplace_back();
        resource.positions =
            data->device.create_buffer<luisa::float3>(
                upload.positions.size());
        resource.normals =
            data->device.create_buffer<luisa::float3>(
                upload.normals.size());
        resource.uv =
            data->device.create_buffer<luisa::float2>(
                upload.uv.size());
        resource.uv_tangents =
            data->device.create_buffer<luisa::float4>(
                upload.uv_tangents.size());
        resource.generated =
            data->device.create_buffer<luisa::float3>(
                upload.generated.size());
        resource.triangles =
            data->device.create_buffer<Triangle>(
                upload.triangles.size());
        resource.triangle_material_slots =
            data->device.create_buffer<luisa::uint>(
                upload.triangle_material_slots.size());
        resource.triangle_random_per_island =
            data->device.create_buffer<float>(
                upload.triangle_random_per_island.size());
        resource.triangle_smooth =
            data->device.create_buffer<luisa::uint>(
                upload.triangle_smooth.size());
        resource.attributes.reserve(
            upload.attributes.size());
        resource.mesh = data->device.create_mesh(
            resource.positions, resource.triangles);
        data->heap.emplace_on_update(
            bindless_base, resource.triangles);
        data->heap.emplace_on_update(
            bindless_base + 1u, resource.positions);
        data->heap.emplace_on_update(
            bindless_base + 2u, resource.normals);
        data->heap.emplace_on_update(
            bindless_base + 3u, resource.uv);
        data->heap.emplace_on_update(
            bindless_base + 4u,
            resource.triangle_material_slots);
        data->heap.emplace_on_update(
            bindless_base + 5u,
            resource.generated);
        data->heap.emplace_on_update(
            bindless_base + 6u,
            resource.triangle_random_per_island);
        data->heap.emplace_on_update(
            bindless_base + 7u,
            resource.uv_tangents);
        data->heap.emplace_on_update(
            bindless_base + 8u,
            resource.triangle_smooth);
        const auto attribute_offset =
            static_cast<std::uint32_t>(
                attribute_bindings.size());
        for (const auto &attribute :
             upload.attributes) {
            auto &attribute_resource =
                resource.attributes.emplace_back(
                    data->device.create_buffer<luisa::float4>(
                        attribute.values.size()));
            const auto attribute_slot =
                next_attribute_slot++;
            data->heap.emplace_on_update(
                attribute_slot, attribute_resource);
            attribute_bindings.emplace_back(
                AttributeBindingGpu{
                    .id = attribute.id,
                    .value_slot = attribute_slot,
                    .domain = attribute.domain});
            stream << attribute_resource.copy_from(
                luisa::span{attribute.values});
        }
        attribute_ranges.emplace_back(
            AttributeRangeGpu{
                .offset = attribute_offset,
                .count =
                    static_cast<std::uint32_t>(
                        upload.attributes.size()),
                .triangle_slot = bindless_base,
                .padding = 0u});
        stream << resource.positions.copy_from(
                      luisa::span{upload.positions})
               << resource.normals.copy_from(
                      luisa::span{upload.normals})
               << resource.uv.copy_from(
                      luisa::span{upload.uv})
               << resource.uv_tangents.copy_from(
                      luisa::span{upload.uv_tangents})
               << resource.generated.copy_from(
                      luisa::span{upload.generated})
               << resource.triangles.copy_from(
                      luisa::span{upload.triangles})
               << resource.triangle_material_slots.copy_from(
                      luisa::span{
                          upload.triangle_material_slots})
               << resource.triangle_random_per_island
                      .copy_from(
                          luisa::span{
                              upload
                                  .triangle_random_per_island})
               << resource.triangle_smooth.copy_from(
                      luisa::span{
                          upload.triangle_smooth})
               << resource.mesh.build();
        geometry_indices.emplace(geometry_id, index);
        geometry_gpu.emplace_back(GeometryGpu{
            .bindless_base = bindless_base,
            .material_offset = material_offset,
            .material_count =
                static_cast<std::uint32_t>(
                    std::max<std::size_t>(
                        geometry.material_slots.size(), 1u)),
            .attribute_domains =
                upload.attribute_domains,
            .cycles_primitive_offset =
                cycles_primitive_offsets.at(
                    geometry_id),
            .generated_transform =
                upload.generated_transform});
    }
    if (!result.diagnostics.empty()) {
        return result;
    }

    std::map<contract::MaterialId, bool> material_may_emit;
    std::map<
        contract::MaterialId,
        contract::EmissionSampling>
        material_emission_sampling;
    for (const auto &[material_id, material] :
         data->materials.materials()) {
        const auto source_material =
            snapshot.materials.find(material_id);
        const auto sampling =
            source_material != snapshot.materials.end()
                ? source_material->second.emission_sampling
                : contract::EmissionSampling::automatic;
        material_emission_sampling.emplace(
            material_id, sampling);
        const auto binding =
            data->material_bindings.find(material_id);
        material_may_emit.emplace(
            material_id,
            sampling !=
                    contract::EmissionSampling::none &&
                binding != data->material_bindings.end() &&
                (binding->second.flags &
                 material_flag_may_emit) != 0u);
    }

    luisa::vector<InstanceGpu> instances;
    luisa::vector<MaterialBindingGpu> override_materials;
    luisa::vector<EmissiveTriangleGpu> emissive_triangles;
    std::vector<float> emissive_triangle_areas;
    data->accel = data->device.create_accel();
    for (const auto &[instance_id, instance] :
         snapshot.instances) {
        static_cast<void>(instance_id);
        const auto geometry_iter =
            geometry_indices.find(instance.geometry);
        if (geometry_iter == geometry_indices.end()) {
            diagnose(
                result.diagnostics,
                "Instance '" + instance.name +
                    "' references unavailable geometry.");
            continue;
        }
        const auto override_offset =
            static_cast<std::uint32_t>(
                override_materials.size());
        for (const auto material_id :
             instance.material_overrides) {
            const auto material_iter =
                data->material_bindings.find(material_id);
            if (material_iter ==
                data->material_bindings.end()) {
                diagnose(
                    result.diagnostics,
                    "Instance '" + instance.name +
                        "' references unavailable override "
                        "material.");
                break;
            }
            override_materials.emplace_back(
                to_luisa(material_iter->second));
        }
        if (!result.diagnostics.empty()) {
            continue;
        }
        const auto instance_index =
            static_cast<std::uint32_t>(instances.size());
        const auto normalized_visibility =
            instance.visibility_mask ==
                    ~std::uint32_t{0u}
                ? contract::all_ray_visibility
                : instance.visibility_mask;
        instances.emplace_back(InstanceGpu{
            .geometry_index = geometry_iter->second,
            .override_offset = override_offset,
            .override_count =
                static_cast<std::uint32_t>(
                    instance.material_overrides.size()),
            .visibility_mask =
                normalized_visibility,
            .object_random = std::clamp(
                instance.random, 0.0f, 1.0f),
            .particle_index = instance.particle_index,
            .shadow_terminator_geometry_offset =
                std::max(
                    instance
                        .shadow_terminator_geometry_offset,
                    0.0f),
            .cycles_object_index =
                instance.cycles_object_index.value_or(
                    cycles_shader_identity::
                        invalid_index),
            .cycles_light_group =
                instance.cycles_light_group,
            .is_shadow_catcher =
                instance.is_shadow_catcher ? 1u : 0u});
        const auto &geometry =
            snapshot.geometries.at(instance.geometry);
        const auto light_visible =
            instance.visibility_mask ==
                ~std::uint32_t{0u} ||
            (instance.visibility_mask &
             (diffuse_visibility |
              glossy_visibility |
              transmission_visibility |
              volume_scatter_visibility)) != 0u;
        if (light_visible) {
            for (std::size_t primitive_index = 0u;
                 primitive_index < geometry.triangles.size();
                 ++primitive_index) {
                const auto material_slot =
                    primitive_index <
                            geometry
                                .triangle_material_slots
                                .size()
                        ? geometry.triangle_material_slots
                              [primitive_index]
                        : 0u;
                std::optional<contract::MaterialId> material_id;
                if (material_slot <
                    instance.material_overrides.size()) {
                    material_id =
                        instance.material_overrides[
                            material_slot];
                } else if (!geometry.material_slots.empty()) {
                    material_id =
                        geometry.material_slots[
                            std::min<std::size_t>(
                                material_slot,
                                geometry.material_slots.size() -
                                    1u)];
                }
                if (!material_id ||
                    !material_may_emit[*material_id]) {
                    continue;
                }
                const auto &triangle =
                    geometry.triangles[primitive_index];
                if (triangle[0u] >= geometry.positions.size() ||
                    triangle[1u] >= geometry.positions.size() ||
                    triangle[2u] >= geometry.positions.size()) {
                    diagnose(
                        result.diagnostics,
                        "Instance '" + instance.name +
                            "' has an emissive triangle with "
                            "out-of-range vertex indices.");
                    break;
                }
                const auto tag_iter =
                    data->material_bindings.find(*material_id);
                if (tag_iter ==
                    data->material_bindings.end()) {
                    diagnose(
                        result.diagnostics,
                        "Instance '" + instance.name +
                            "' has an unavailable emissive "
                            "material.");
                    break;
                }
                const auto smooth =
                    primitive_index <
                            geometry.triangle_smooth.size() &&
                        geometry.triangle_smooth[
                            primitive_index] != 0u;
                const auto base_shader_index =
                    tag_iter->second.cycles_shader_index !=
                            cycles_shader_identity::invalid_index
                        ? tag_iter->second.cycles_shader_index
                        : tag_iter->second.material_identity;
                emissive_triangles.emplace_back(
                    EmissiveTriangleGpu{
                        .instance_index = instance_index,
                        .geometry_index =
                            geometry_iter->second,
                        .primitive_index =
                            static_cast<std::uint32_t>(
                                primitive_index),
                        .surface_tag =
                            tag_iter->second.surface_tag,
                        .parameter_block =
                            tag_iter->second.parameter_block,
                        .emission_sampling =
                            static_cast<std::uint32_t>(
                                material_emission_sampling
                                    .at(*material_id)),
                        .emission_is_constant =
                            (tag_iter->second.flags &
                             material_flag_constant_emission) != 0u
                                ? 1u
                                : 0u,
                        .visibility_mask =
                            normalized_visibility,
                        .cycles_primitive_index =
                            cycles_primitive_offsets.at(
                                instance.geometry) +
                            static_cast<std::uint32_t>(
                                primitive_index),
                        .cycles_object_index =
                            instance.cycles_object_index.value_or(
                                instance_index),
                        .cycles_shader_id =
                            cycles_shader_identity::emissive_triangle(
                                base_shader_index,
                                smooth,
                                normalized_visibility,
                                instance.is_shadow_catcher),
                        .cycles_shader_flags =
                            cycles_shader_identity::emissive_triangle_flags(
                                smooth,
                                normalized_visibility,
                                instance.is_shadow_catcher),
                        .cycles_light_group =
                            instance.cycles_light_group});
                emissive_triangle_areas.emplace_back(
                    world_triangle_area(
                        instance.transform,
                        geometry.positions[triangle[0u]],
                        geometry.positions[triangle[1u]],
                        geometry.positions[triangle[2u]]));
            }
        }
        const auto visibility =
            static_cast<std::uint8_t>(
                normalized_visibility);
        data->accel.emplace_back(
            data->geometries[geometry_iter->second].mesh,
            to_luisa(instance.transform),
            visibility,
            false,
            instance_index);
    }
    if (!result.diagnostics.empty()) {
        return result;
    }

    if (instances.empty()) {
        // Luisa requires at least one TLAS instance. Represent an empty
        // Cycles scene with a backend-only triangle whose visibility mask is
        // zero. It cannot be hit by any Psycles ray and never appears in the
        // logical scene, passes, material tables, or emitter distribution.
        luisa::vector<luisa::float3> dummy_positions;
        dummy_positions.emplace_back(
            luisa::make_float3(0.0f, 0.0f, 0.0f));
        dummy_positions.emplace_back(
            luisa::make_float3(1.0f, 0.0f, 0.0f));
        dummy_positions.emplace_back(
            luisa::make_float3(0.0f, 1.0f, 0.0f));
        luisa::vector<Triangle> dummy_triangles;
        dummy_triangles.emplace_back(
            Triangle{0u, 1u, 2u});
        auto &dummy = data->geometries.emplace_back();
        dummy.positions =
            data->device.create_buffer<luisa::float3>(
                dummy_positions.size());
        dummy.triangles =
            data->device.create_buffer<Triangle>(
                dummy_triangles.size());
        dummy.mesh = data->device.create_mesh(
            dummy.positions, dummy.triangles);
        stream << dummy.positions.copy_from(
                      luisa::span{dummy_positions})
               << dummy.triangles.copy_from(
                      luisa::span{dummy_triangles})
               << dummy.mesh.build();
        data->accel.emplace_back(
            dummy.mesh,
            to_luisa(Mat4f{}),
            std::uint8_t{0u},
            false,
            0u);
    }

    luisa::vector<LightGpu> lights;
    Vec3f background{};
    for (const auto &[light_id, light] :
         snapshot.lights) {
        static_cast<void>(light_id);
        if (light.type == LightType::background) {
            background.x += light.color.x * light.power;
            background.y += light.color.y * light.power;
            background.z += light.color.z * light.power;
            continue;
        }
        auto normalized_axis = [](Vec3f axis) noexcept {
            const auto length = std::sqrt(
                axis.x * axis.x +
                axis.y * axis.y +
                axis.z * axis.z);
            if (length <= 1.0e-20f) {
                return std::pair{
                    Vec3f{0.0f, 0.0f, 0.0f},
                    0.0f};
            }
            return std::pair{
                Vec3f{
                    axis.x / length,
                    axis.y / length,
                    axis.z / length},
                length};
        };
        const auto [axis_x, axis_x_length] =
            normalized_axis(matrix_axis(light.transform, 0u));
        const auto [axis_y, axis_y_length] =
            normalized_axis(matrix_axis(light.transform, 1u));
        auto [axis_z, axis_z_length] =
            normalized_axis(matrix_axis(light.transform, 2u));
        if (axis_z == Vec3f{}) {
            axis_z = {0.0f, 0.0f, 1.0f};
        }
        std::uint32_t flags = 0u;
        flags |= light.normalize
                     ? light_flag_normalize
                     : 0u;
        flags |= light.ellipse
                     ? light_flag_ellipse
                     : 0u;
        flags |= light.is_sphere
                     ? light_flag_sphere
                     : 0u;
        flags |=
            light.type == LightType::area &&
                    light.spread >= pi - 1.0e-6f
                ? light_flag_full_spread
                : 0u;
        MaterialBinding light_binding{
            .surface_tag = ~std::uint32_t{0u},
            .parameter_block = 0u,
            .cycles_shader_index =
                cycles_shader_identity::invalid_index};
        if (light.shader) {
            if (const auto iter =
                    data->material_bindings.find(
                        *light.shader);
                iter != data->material_bindings.end()) {
                light_binding = iter->second;
            }
        }
        const auto effective_light_mis =
            light.use_mis &&
            (light.type == LightType::point ||
             light.type == LightType::spot
                 ? light.size > 0.0f
                 : light.type == LightType::area
                       ? light.size * axis_x_length != 0.0f &&
                             (light.size_y > 0.0f
                                  ? light.size_y
                                  : light.size) *
                                     axis_y_length !=
                                 0.0f &&
                             light.spread > 0.0f
                       : light.type == LightType::distant
                             ? light.angle > 0.0f
                             : true);
        const auto cycles_shader_id =
            light.cycles_shader_index
                ? cycles_shader_identity::analytic_light(
                      *light.cycles_shader_index,
                      light.cast_shadow,
                      light.visibility_mask,
                      light.is_shadow_catcher,
                      effective_light_mis)
                : cycles_shader_identity::invalid_index;
        const auto cycles_shader_flags =
            cycles_shader_identity::analytic_light_flags(
                light.cast_shadow,
                light.visibility_mask,
                light.is_shadow_catcher,
                effective_light_mis);
        flags |= effective_light_mis
                     ? light_flag_use_mis
                     : 0u;
        flags |=
            effective_light_mis &&
                    (light.type == LightType::area ||
                     light.type == LightType::point ||
                     light.type == LightType::spot)
                ? light_flag_forward_intersectable
                : 0u;
        flags |=
            (light_binding.surface_tag ==
                 ~std::uint32_t{0u} ||
             (light_binding.flags &
              material_flag_constant_emission) != 0u)
                ? light_flag_constant_emission
                : 0u;
        lights.emplace_back(LightGpu{
            .type =
                static_cast<std::uint32_t>(light.type),
            .position =
                to_luisa(matrix_translation(light.transform)),
            .axis_x = to_luisa(axis_x),
            .axis_y = to_luisa(axis_y),
            .axis_z = to_luisa(axis_z),
            .axis_scale = luisa::make_float3(
                axis_x_length,
                axis_y_length,
                axis_z_length),
            .color = to_luisa(light.color),
            .power = light.power,
            .radius = light.size,
            .size_u =
                light.size * axis_x_length,
            .size_v =
                (light.size_y > 0.0f
                     ? light.size_y
                     : light.size) *
                axis_y_length,
            .spread = light.spread,
            .spot_angle = light.spot_angle,
            .spot_smooth = light.spot_smooth,
            .angle = light.angle,
            .flags = flags,
            .surface_tag =
                light_binding.surface_tag,
            .parameter_block =
                light_binding.parameter_block,
            .cycles_object_index =
                light.cycles_object_index.value_or(
                    cycles_shader_identity::
                        invalid_index),
            .cycles_light_group =
                light.cycles_light_group,
            .cycles_shader_id =
                cycles_shader_id,
            .cycles_shader_flags =
                cycles_shader_flags,
            .cycles_type =
                cycles_shader_identity::light_type(
                    light.type),
            .visibility_mask =
                light.visibility_mask});
    }
    data->background = to_luisa(background);
    data->light_count =
        static_cast<std::uint32_t>(lights.size());
    data->emissive_triangle_count =
        static_cast<std::uint32_t>(
            emissive_triangles.size());
    const auto world_emission_is_constant =
        !data->world_surface ||
        (data->world_surface->flags &
         material_flag_constant_emission) != 0u;
    data->environment_emission_is_constant =
        world_emission_is_constant &&
        !data->environment_texture_slot &&
        data->environment_suns.empty() &&
        !data->nishita_environment;

    bool world_is_spatially_varying = false;
    if (snapshot.world_shader) {
        if (const auto *world_material =
                data->materials.find(*snapshot.world_shader)) {
            world_is_spatially_varying =
                volume_capabilities
                    .analyze(*world_material->surface_program())
                    .has_spatial_values;
        }
    }
    // Cycles LightManager::test_enabled_lights only enables a background
    // emitter when MIS is requested and the raw surface graph is spatially
    // varying (or a portal exists). MANUAL controls map resolution; it does
    // not force a constant graph into the distribution. Portals are outside
    // the currently supported subset.
    const auto include_environment =
        data->world_surface.has_value() &&
        snapshot.world_sampling !=
            contract::WorldSampling::none &&
        world_is_spatially_varying;
    const auto light_distribution =
        sampling::build_cycles_light_distribution(
            emissive_triangle_areas,
            data->light_count,
            include_environment);
    luisa::vector<LightDistributionGpu>
        light_distribution_entries;
    light_distribution_entries.reserve(
        light_distribution.entries.size());
    for (std::size_t emitter_id = 0u;
         emitter_id <
         light_distribution.entries.size();
         ++emitter_id) {
        const auto &entry =
            light_distribution.entries[emitter_id];
        light_distribution_entries.emplace_back(
            LightDistributionGpu{
                .cumulative = entry.cumulative,
                .selection_pdf = entry.selection_pdf,
                .kind = static_cast<std::uint32_t>(
                    entry.kind),
                .index = entry.index,
                .emitter_id =
                    emitter_id <
                            static_cast<std::size_t>(
                                light_distribution
                                    .emitter_count)
                        ? static_cast<std::uint32_t>(
                              emitter_id)
                        : ~std::uint32_t{0u}});
    }
    data->light_distribution_count =
        light_distribution.usable()
            ? light_distribution.emitter_count
            : 0u;
    data->triangle_area_pdf =
        light_distribution.triangle_area_pdf;
    data->light_selection_pdf =
        light_distribution.light_selection_pdf;
    data->environment_in_light_distribution =
        include_environment &&
        data->light_distribution_count > 0u;
    if (light_distribution_entries.empty()) {
        light_distribution_entries.emplace_back(
            LightDistributionGpu{});
    }
    if (lights.empty()) {
        lights.emplace_back(LightGpu{});
    }
    if (emissive_triangles.empty()) {
        emissive_triangles.emplace_back(
            EmissiveTriangleGpu{});
    }
    if (geometry_materials.empty()) {
        geometry_materials.emplace_back(
            MaterialBindingGpu{
                .cycles_shader_index =
                    cycles_shader_identity::
                        invalid_index});
    }
    if (override_materials.empty()) {
        override_materials.emplace_back(
            MaterialBindingGpu{
                .cycles_shader_index =
                    cycles_shader_identity::
                        invalid_index});
    }
    // Empty-world renders are valid Cycles scenes. Luisa buffers cannot be
    // zero-sized, so keep inert storage records while leaving the acceleration
    // structure itself empty; the render kernel only reads these buffers after
    // a committed hit.
    if (geometry_gpu.empty()) {
        geometry_gpu.emplace_back(GeometryGpu{});
    }
    if (attribute_bindings.empty()) {
        attribute_bindings.emplace_back(
            AttributeBindingGpu{});
    }
    if (attribute_ranges.empty()) {
        attribute_ranges.emplace_back(
            AttributeRangeGpu{});
    }
    if (instances.empty()) {
        instances.emplace_back(InstanceGpu{});
    }

    data->geometry_buffer =
        data->device.create_buffer<GeometryGpu>(
            geometry_gpu.size());
    data->attribute_binding_buffer =
        data->device.create_buffer<
            AttributeBindingGpu>(
            attribute_bindings.size());
    data->attribute_range_buffer =
        data->device.create_buffer<
            AttributeRangeGpu>(
            attribute_ranges.size());
    data->heap.emplace_on_update(
        data->attribute_binding_slot,
        data->attribute_binding_buffer);
    data->heap.emplace_on_update(
        data->attribute_range_slot,
        data->attribute_range_buffer);
    data->instance_buffer =
        data->device.create_buffer<InstanceGpu>(
            instances.size());
    data->geometry_material_buffer =
        data->device.create_buffer<MaterialBindingGpu>(
            geometry_materials.size());
    data->override_material_buffer =
        data->device.create_buffer<MaterialBindingGpu>(
            override_materials.size());
    data->light_buffer =
        data->device.create_buffer<LightGpu>(
            lights.size());
    data->emissive_triangle_buffer =
        data->device.create_buffer<EmissiveTriangleGpu>(
            emissive_triangles.size());
    data->light_distribution_buffer =
        data->device.create_buffer<LightDistributionGpu>(
            light_distribution_entries.size());
    stream << data->geometry_buffer.copy_from(
                  luisa::span{geometry_gpu})
           << data->attribute_binding_buffer.copy_from(
                  luisa::span{attribute_bindings})
           << data->attribute_range_buffer.copy_from(
                  luisa::span{attribute_ranges})
           << data->instance_buffer.copy_from(
                  luisa::span{instances})
           << data->geometry_material_buffer.copy_from(
                  luisa::span{geometry_materials})
           << data->override_material_buffer.copy_from(
                  luisa::span{override_materials})
           << data->light_buffer.copy_from(
                  luisa::span{lights})
           << data->emissive_triangle_buffer.copy_from(
                  luisa::span{emissive_triangles})
           << data->light_distribution_buffer.copy_from(
                  luisa::span{light_distribution_entries})
           << data->texture_heap.update()
           << data->heap.update()
           << data->accel.build()
           << synchronize();

    const auto majorants =
        VolumeMajorantSceneComponent{}.build(
            data, stream, snapshot);
    if (!majorants.ok()) {
        diagnose(result.diagnostics, majorants.diagnostic);
        return result;
    }

    configure_background_sampling(
        *data,
        snapshot,
        include_environment);
    build_background_sampling_distribution(
        data,
        stream);

    result.scene =
        std::make_unique<LuisaCompiledScene>(
            std::move(data));
    return result;
}


}// namespace psycles::luisa_backend
