#include "cycles_shader_identity.h"
#include "graph_surface_value_expression.h"
#include "path_tracer_bsdf_tables.h"
#include "path_tracer_curve_scene.h"
#include "path_tracer_displacement_scene.h"
#include "path_tracer_environment.h"
#include "path_tracer_generated_coordinates.h"
#include "path_tracer_image_decode.h"
#include "path_tracer_internal.h"
#include "path_tracer_light_sampling_scene.h"
#include "path_tracer_scene_geometry.h"
#include "path_tracer_scene_upload.h"
#include "path_tracer_shader_services.h"
#include "path_tracer_subsurface_scene.h"
#include "path_tracer_surface_route_policy.h"
#include "path_tracer_surfaces.h"
#include "path_tracer_surface_values.h"
#include "path_tracer_tangent_space.h"
#include "path_tracer_volume_capabilities.h"
#include "path_tracer_volume_majorant_scene.h"
#include "shader_table_data.h"

#include <psycles/compiler/core_nodes.h>
#include <psycles/contract/cycles_pointiness.h>
#include <psycles/luisa/cycles_nishita.h>

#include <string_view>

namespace psycles::luisa_backend {
using namespace detail;

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
    data->populate_surface_once =
        populate_surface_once_requested();
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
    data->world_max_bounces =
        snapshot.world_max_bounces;
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
    std::set<contract::MaterialId> surface_bssrdf_materials;
    std::set<contract::MaterialId> surface_bssrdf_bump_materials;
    const auto reachable_surface_materials =
        collect_reachable_surface_materials(snapshot);
    for (const auto &[material_id, material] :
         data->materials.materials()) {
        if (!reachable_surface_materials.contains(material_id)) {
            continue;
        }
        const auto &source_material =
            snapshot.materials.at(material_id);
        const auto has_bssrdf = compiler::cycles_surface_has_bssrdf(
            *material.surface_program(),
            material.parameters());
        if (has_bssrdf) {
            surface_bssrdf_materials.emplace(material_id);
        }
        if (compiler::cycles_surface_has_bssrdf_bump(
                *material.surface_program(),
                material.parameters(),
                source_material.displacement_method)) {
            surface_bssrdf_bump_materials.emplace(material_id);
        }
    }
    data->has_subsurface = !surface_bssrdf_materials.empty();
    const SubsurfaceSceneComponent subsurface_scene;
    const auto subsurface_scene_plan = subsurface_scene.plan(
        snapshot, surface_bssrdf_materials);
    auto cycles_instance_intersection_plan =
        build_cycles_instance_intersection_plan(
            snapshot, surface_bssrdf_materials);
    std::map<contract::GeometryId, Mat4f>
        cycles_static_transform_by_geometry;
    auto source_instance_index = std::size_t{0u};
    for (const auto &[instance_id, instance] : snapshot.instances) {
        static_cast<void>(instance_id);
        const auto &plan = cycles_instance_intersection_plan[
            source_instance_index++];
        if (plan.transform_applied &&
            snapshot.geometries.contains(instance.geometry)) {
            cycles_static_transform_by_geometry.emplace(
                instance.geometry, instance.transform);
        }
    }
    const VolumeProgramCapabilityComponent
        volume_capabilities;
    data->volume_metadata
        .closure_allocation_budget =
        cycles_scene_closure_allocation_budget(
            data->materials);

    luisa::vector<float> scalar_parameters;
    luisa::vector<luisa::float3> vector_parameters;
    std::vector<PendingShaderTable> shader_tables;
    std::vector<std::uint32_t>
        volume_surface_flags;
    std::map<std::uint64_t, std::uint32_t>
        surface_tags_by_signature;
    std::map<std::uint64_t, compiler::SurfaceClosurePlan>
        closure_plans_by_signature;
    for (const auto &[id, material] :
         data->materials.materials()) {
        static_cast<void>(id);
        const auto &program = *material.surface_program();
        closure_plans_by_signature[
            program.structure_signature()]
            .merge(compiler::analyze_surface_closure_plan(
                program, material.parameters()));
    }
    std::vector<std::shared_ptr<const compiler::SurfaceProgram>>
        surface_programs_by_tag;
    std::vector<compiler::SurfaceClosurePlan>
        surface_closure_plans_by_tag;
    surface_programs_by_tag.reserve(closure_plans_by_signature.size());
    surface_closure_plans_by_tag.reserve(closure_plans_by_signature.size());
    std::set<std::uint32_t> surface_bssrdf_bump_tags;
    std::set<contract::MaterialId>
        pointiness_materials;
    for (const auto &[id, material] :
         data->materials.materials()) {
        const auto base = static_cast<std::uint32_t>(
            scalar_parameters.size());
        const auto signature =
            material.surface_program()->structure_signature();
        auto [surface_iter, inserted] =
            surface_tags_by_signature.try_emplace(signature, 0u);
        if (inserted) {
            const auto tag =
                data->surfaces.create<GraphSurface>(
                    material.surface_program(),
                    closure_plans_by_signature.at(signature));
            if (tag != surface_programs_by_tag.size()) {
                diagnose(
                    result.diagnostics,
                    "Surface runtime tags are not a dense insertion order.");
                return result;
            }
            surface_iter->second = tag;
            surface_programs_by_tag.emplace_back(
                material.surface_program());
            surface_closure_plans_by_tag.emplace_back(
                closure_plans_by_signature.at(signature));
        }
        if (surface_bssrdf_bump_materials.contains(id)) {
            surface_bssrdf_bump_tags.emplace(surface_iter->second);
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
                    snapshot.materials.at(id).cycles_shader_index.value_or(
                        cycles_shader_identity::invalid_index),
                .material_identity =
                    static_cast<std::uint32_t>(data->material_bindings.size()),
                .flags =
                    (capabilities.may_have_volume ? material_flag_has_volume
                                                  : 0u) |
                    (may_emit ? material_flag_may_emit : 0u) |
                    (emission_is_constant ? material_flag_constant_emission
                                          : 0u) |
                    (snapshot.materials.at(id).use_bump_map_correction
                         ? material_flag_use_bump_map_correction
                         : 0u) |
                    (capabilities.may_be_transparent
                         ? material_flag_may_be_transparent
                         : 0u) |
                    (surface_bssrdf_bump_materials.contains(id)
                         ? material_flag_has_bssrdf_bump
                         : 0u),
                .emission_sampling =
                    snapshot.materials.at(id).emission_sampling,
                .volume_sampling = snapshot.materials.at(id).volume_sampling});
        const auto &program = *material.surface_program();
        if (program.root().valid() &&
            std::any_of(
                program.value_instructions().begin(),
                program.value_instructions().end(),
                [](const auto &instruction) noexcept {
                    return instruction.operation ==
                           compiler::ValueOperation::pointiness;
                })) {
            pointiness_materials.emplace(id);
        }
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
                instruction.operand(
                    compiler::value_operand::nishita_sky::elevation),
                instruction.operand(
                    compiler::value_operand::nishita_sky::rotation),
                instruction.operand(
                    compiler::value_operand::nishita_sky::size),
                instruction.operand(
                    compiler::value_operand::nishita_sky::intensity),
                instruction.operand(
                    compiler::value_operand::nishita_sky::altitude),
                instruction.operand(
                    compiler::value_operand::nishita_sky::air),
                instruction.operand(
                    compiler::value_operand::nishita_sky::dust),
                instruction.operand(
                    compiler::value_operand::nishita_sky::ozone)};
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
            if (parameter.type == contract::SocketType::string) {
                if (value == nullptr) {
                    diagnose(
                        result.diagnostics,
                        "Material '" + material.name() +
                            "' has no value for shader table property '" +
                            parameter.socket + "'.");
                } else {
                    const auto descriptor_index =
                        static_cast<std::uint32_t>(
                            vector_parameters.size());
                    auto staged = stage_shader_table(
                        program,
                        parameter,
                        *value,
                        descriptor_index);
                    if (!staged.valid) {
                        diagnose(
                            result.diagnostics,
                            "Material '" + material.name() +
                                "' shader table property '" +
                                parameter.socket + "': " +
                                staged.diagnostic + ".");
                    } else {
                        shader_tables.emplace_back(
                            std::move(staged.table));
                    }
                }
            } else if (value != nullptr) {
                switch (surface_value_category(parameter.type)) {
                    case SurfaceValueCategory::scalar:
                        scalar = scalar_parameter_value(*value);
                        break;
                    case SurfaceValueCategory::vector:
                        vector = vector_parameter_value(*value);
                        break;
                    case SurfaceValueCategory::unsigned_integer:
                        vector = unsigned_parameter_value(*value);
                        break;
                }
            }
            scalar_parameters.emplace_back(scalar);
            vector_parameters.emplace_back(vector);
        }
    }
    std::string shader_table_diagnostic;
    if (!finalize_shader_tables(
            shader_tables,
            scalar_parameters,
            vector_parameters,
            shader_table_diagnostic)) {
        diagnose(
            result.diagnostics,
            "Shader table upload: " +
                shader_table_diagnostic + ".");
    }
    if (!result.diagnostics.empty()) {
        return result;
    }
    data->surface_bssrdf_bump_tags.reserve(
        surface_bssrdf_bump_tags.size());
    for (const auto tag : surface_bssrdf_bump_tags) {
        data->surface_bssrdf_bump_tags.emplace_back(tag);
    }
    if (compact_surface_values_requested()) {
        std::string diagnostic;
        data->surface_values = build_surface_value_runtime(
            data->device,
            surface_programs_by_tag,
            surface_closure_plans_by_tag,
            data->surface_bssrdf_bump_tags,
            diagnostic, surface_value_region_handler_site_budget_requested());
        if (!data->surface_values) {
            diagnose(
                result.diagnostics,
                "Compact surface value execution: " + diagnostic + ".");
            return result;
        }
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

    auto cycles_bsdf_values = make_cycles_bsdf_table_values(snapshot.shader_color_space);
    data->cycles_bsdf_table_buffer =
        data->device.create_buffer<float>(
            cycles_bsdf_values.size());

    std::set<contract::GeometryId>
        override_pointiness_geometries;
    for (const auto &[instance_id, instance] : snapshot.instances) {
        static_cast<void>(instance_id);
        if (std::any_of(
                instance.material_overrides.begin(),
                instance.material_overrides.end(),
                [&](const auto material) noexcept {
                    return pointiness_materials.contains(material);
                })) {
            override_pointiness_geometries.emplace(
                instance.geometry);
        }
    }
    std::size_t attribute_count = 0u;
    for (const auto &[id, geometry] :
         snapshot.geometries) {
        const auto requires_pointiness =
            override_pointiness_geometries.contains(id) ||
            std::any_of(
                geometry.material_slots.begin(),
                geometry.material_slots.end(),
                [&](const auto material) noexcept {
                    return pointiness_materials.contains(material);
                });
        if (requires_pointiness && !geometry.pointiness_source) {
            diagnose(
                result.diagnostics,
                "Geometry '" + geometry.name +
                    "' uses Geometry.Pointiness but has no evaluated "
                    "point normals and original edges.");
        }
        attribute_count +=
            geometry.color_attributes.size() +
            geometry.uv_layers.size() +
            geometry.uv_tangent_layers.size() * 2u +
            (geometry.pointiness_source ? 1u : 0u);
    }
    if (!result.diagnostics.empty()) {
        return result;
    }
    const auto fixed_geometry_slots =
        (snapshot.geometries.size() +
         snapshot.curve_geometries.size()) *
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
        snapshot.geometries.size() +
        snapshot.curve_geometries.size());
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
    if (data->surface_values) {
        upload_surface_value_runtime(
            stream, *data->surface_values);
    }

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
        std::vector<float> pointiness_values;
        if (geometry.pointiness_source) {
            try {
                pointiness_values =
                    contract::make_cycles_pointiness_attribute(
                        geometry.positions,
                        geometry.pointiness_source->point_normals,
                        geometry.pointiness_source->edges);
            } catch (const std::invalid_argument &error) {
                diagnose(
                    result.diagnostics,
                    "Geometry '" + geometry.name +
                        "' has an invalid Cycles Pointiness source: " +
                        error.what() + ".");
                continue;
            }
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
        if (const auto transform_iter =
                cycles_static_transform_by_geometry.find(geometry_id);
            transform_iter !=
            cycles_static_transform_by_geometry.end()) {
            upload.cycles_intersection_positions.reserve(
                geometry.positions.size());
            for (const auto position : geometry.positions) {
                upload.cycles_intersection_positions.emplace_back(
                    to_luisa(cycles_transform_point(
                        transform_iter->second, position)));
            }
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
        upload.default_uv_available =
            geometry.default_uv_available.value_or(
                !geometry.uv.values.empty());
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
            geometry.uv_tangent_layers.size() * 2u +
            (pointiness_values.empty() ? 0u : 1u));
        if (!pointiness_values.empty()) {
            auto &attribute = upload.attributes.emplace_back();
            attribute.id =
                contract::cycles_pointiness_attribute_id;
            attribute.domain = attribute_domain_point;
            attribute.values.reserve(pointiness_values.size());
            for (const auto value : pointiness_values) {
                attribute.values.emplace_back(
                    luisa::make_float4(value, 0.0f, 0.0f, 0.0f));
            }
        }
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
        std::map<std::string, std::size_t, std::less<>>
            uv_attribute_indices;
        for (const auto &[name, source] :
             geometry.uv_layers) {
            uv_attribute_indices.emplace(
                name, upload.attributes.size());
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
            const auto uv = uv_attribute_indices.find(name);
            if (uv == uv_attribute_indices.end()) {
                diagnose(
                    result.diagnostics,
                    "Geometry '" + geometry.name +
                        "' has a named tangent layer without its UV layer.");
                break;
            }
            const auto tangent_index = upload.attributes.size();
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
            const auto tangent_domain = attribute.domain;
            const auto undisplaced_index =
                upload.attributes.size();
            auto &undisplaced =
                upload.attributes.emplace_back();
            undisplaced.id =
                contract::uv_undisplaced_tangent_attribute_id(name);
            undisplaced.domain = tangent_domain;
            upload.uv_tangent_layers.emplace_back(
                UvTangentLayerUpload{
                    .uv_attribute_index = uv->second,
                    .tangent_attribute_index = tangent_index,
                    .undisplaced_tangent_attribute_index =
                        undisplaced_index});
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
        recompute_cycles_tangent_space(upload);
        initialize_cycles_undisplaced_tangent_space(upload);
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
        if (!upload.cycles_intersection_positions.empty()) {
            resource.cycles_intersection_positions.emplace(
                data->device.create_buffer<luisa::float3>(
                    upload.cycles_intersection_positions.size()));
            data->heap.emplace_on_update(
                bindless_base + 9u,
                *resource.cycles_intersection_positions);
        } else {
            data->heap.emplace_on_update(
                bindless_base + 9u,
                resource.positions);
        }
        // The pre-displacement representation is a total geometry
        // interface. Ordinary meshes alias it to their live buffers;
        // MeshDisplacementSceneComponent replaces these bindings with saved
        // immutable buffers before modifying the mesh.
        data->heap.emplace_on_update(
            bindless_base + 10u,
            resource.positions);
        data->heap.emplace_on_update(
            bindless_base + 11u,
            resource.normals);
        data->heap.emplace_on_update(
            bindless_base + 12u,
            resource.uv_tangents);
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
                          upload.triangle_smooth});
        if (resource.cycles_intersection_positions) {
            stream << resource.cycles_intersection_positions->copy_from(
                luisa::span{upload.cycles_intersection_positions});
        }
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

    const auto curve_upload =
        CurveSceneUploadComponent{}.upload(
            data,
            snapshot,
            stream,
            geometry_indices,
            geometry_gpu,
            geometry_materials,
            attribute_ranges);
    if (!curve_upload.ok()) {
        diagnose(result.diagnostics, curve_upload.diagnostic);
        return result;
    }

    const auto displacement = MeshDisplacementSceneComponent{}.build(
        data,
        stream,
        snapshot,
        geometry_indices,
        uploads,
        attribute_bindings,
        attribute_ranges);
    if (!displacement.ok()) {
        diagnose(result.diagnostics, displacement.diagnostic);
        return result;
    }
    std::map<contract::MaterialId, bool> material_may_emit;
    for (const auto &[material_id, material] :
         data->materials.materials()) {
        static_cast<void>(material);
        const auto binding =
            data->material_bindings.find(
                material_id);
        material_may_emit.emplace(
            material_id,
            binding !=
                    data->material_bindings.end() &&
                binding->second.emission_sampling !=
                    contract::EmissionSampling::none &&
                (binding->second.flags &
                 material_flag_may_emit) != 0u);
    }

    luisa::vector<InstanceGpu> instances;
    luisa::vector<MaterialBindingGpu> override_materials;
    luisa::vector<EmissiveTriangleGpu> emissive_triangles;
    std::vector<float> emissive_triangle_areas;
    data->accel = data->device.create_accel();
    subsurface_scene.initialize_accel(data, subsurface_scene_plan);
    source_instance_index = 0u;
    for (const auto &[instance_id, instance] :
         snapshot.instances) {
        static_cast<void>(instance_id);
        const auto &intersection_plan =
            cycles_instance_intersection_plan[
                source_instance_index++];
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
            .cycles_primitive_offset =
                geometry_gpu[geometry_iter->second]
                    .cycles_primitive_offset,
            .cycles_light_group =
                instance.cycles_light_group,
            .is_shadow_catcher =
                instance.is_shadow_catcher ? 1u : 0u,
            .cycles_transform_applied =
                intersection_plan.transform_applied ? 1u : 0u,
            .cycles_world_to_object =
                to_luisa(intersection_plan.world_to_object)});
        if (const auto curve_resource =
                curve_upload.resource_indices.find(instance.geometry);
            curve_resource !=
            curve_upload.resource_indices.end()) {
            data->accel.emplace_back(
                data->curve_geometries[
                    curve_resource->second]
                    .primitive,
                to_luisa(instance.transform),
                static_cast<std::uint8_t>(
                    normalized_visibility),
                instance_index);
            continue;
        }
        const auto &geometry =
            snapshot.geometries.at(instance.geometry);
        const auto light_visible =
            instance.visibility_mask ==
                ~std::uint32_t{0u} ||
            (instance.visibility_mask &
             mesh_light_sampling_visibility) != 0u;
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
                                tag_iter->second
                                    .emission_sampling),
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
                        from_luisa(uploads[geometry_iter->second]
                                       .positions[triangle[0u]]),
                        from_luisa(uploads[geometry_iter->second]
                                       .positions[triangle[1u]]),
                        from_luisa(uploads[geometry_iter->second]
                                       .positions[triangle[2u]])));
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
        subsurface_scene.append_triangle_instance(
            data, subsurface_scene_plan, instance_index,
            geometry_iter->second, instance.transform);
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
                light.visibility_mask,
            .max_bounces =
                light.max_bounces});
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
    auto light_sampling = build_light_sampling_scene_upload(
        snapshot,
        *data,
        uploads,
        lights,
        emissive_triangles,
        emissive_triangle_areas,
        include_environment);
    if (!light_sampling.ok()) {
        diagnose(result.diagnostics, light_sampling.diagnostic);
        return result;
    }
    data->light_distribution_count = light_sampling.distribution_count;
    data->triangle_area_pdf = light_sampling.triangle_area_pdf;
    data->light_selection_pdf = light_sampling.light_selection_pdf;
    data->environment_in_light_distribution =
        light_sampling.environment_in_distribution;
    data->light_tree_node_count =
        static_cast<std::uint32_t>(light_sampling.tree_nodes.size());
    data->light_tree_emitter_count =
        static_cast<std::uint32_t>(light_sampling.tree_emitters.size());
    data->light_tree_root = light_sampling.tree_root;
    data->light_tree_triangle_lookup_count = static_cast<std::uint32_t>(
        light_sampling.tree_triangle_lookup.size());
    const auto table_upload = SceneTableUploadComponent{}.upload(
        data,
        stream,
        {.geometries = geometry_gpu,
         .attribute_bindings = attribute_bindings,
         .attribute_ranges = attribute_ranges,
         .instances = instances,
         .geometry_materials = geometry_materials,
         .override_materials = override_materials,
         .lights = lights,
         .emissive_triangles = emissive_triangles,
         .light_distribution = light_sampling.distribution,
         .light_tree_nodes = light_sampling.tree_nodes,
         .light_tree_emitters = light_sampling.tree_emitters,
         .light_tree_emitter_mappings =
             light_sampling.tree_emitter_mappings,
         .light_tree_triangle_lookup =
             light_sampling.tree_triangle_lookup});
    if (!table_upload.ok()) {
        diagnose(result.diagnostics, table_upload.diagnostic);
        return result;
    }

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
