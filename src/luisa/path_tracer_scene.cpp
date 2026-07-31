#include "path_tracer_internal.h"
#include "path_tracer_environment.h"
#include "path_tracer_shader_services.h"
#include "path_tracer_surfaces.h"
#include "cycles_shader_identity.h"

#include <psycles/compiler/core_nodes.h>
#include <psycles/luisa/background_sampling.h>
#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_nishita.h>
#include <psycles/sampling/background_distribution.h>
#include <psycles/sampling/light_distribution.h>

#include "cycles_shader_tables_4_5_10.inl"

#include <stb/stb_image.h>

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

[[nodiscard]] Vec3f transform_point(
    const Mat4f &transform,
    Vec3f point) noexcept {
    const auto &e = transform.elements;
    return {
        e[0u] * point.x + e[4u] * point.y +
            e[8u] * point.z + e[12u],
        e[1u] * point.x + e[5u] * point.y +
            e[9u] * point.z + e[13u],
        e[2u] * point.x + e[6u] * point.y +
            e[10u] * point.z + e[14u]};
}

[[nodiscard]] float world_triangle_area(
    const Mat4f &transform,
    Vec3f p0,
    Vec3f p1,
    Vec3f p2) noexcept {
    p0 = transform_point(transform, p0);
    p1 = transform_point(transform, p1);
    p2 = transform_point(transform, p2);
    const Vec3f edge01{
        p1.x - p0.x,
        p1.y - p0.y,
        p1.z - p0.z};
    const Vec3f edge02{
        p2.x - p0.x,
        p2.y - p0.y,
        p2.z - p0.z};
    const Vec3f cross{
        edge01.y * edge02.z - edge01.z * edge02.y,
        edge01.z * edge02.x - edge01.x * edge02.z,
        edge01.x * edge02.y - edge01.y * edge02.x};
    return 0.5f * std::sqrt(
        cross.x * cross.x +
        cross.y * cross.y +
        cross.z * cross.z);
}

[[nodiscard]] bool is_spatial_source(
    compiler::ValueOperation operation) noexcept {
    using compiler::ValueOperation;
    switch (operation) {
        case ValueOperation::surface_position:
        case ValueOperation::shading_normal:
        case ValueOperation::geometric_normal:
        case ValueOperation::incoming:
        case ValueOperation::tangent:
        case ValueOperation::uv:
        case ValueOperation::generated:
        case ValueOperation::object_position:
        case ValueOperation::object_location:
        case ValueOperation::object_random:
        case ValueOperation::particle_index:
        case ValueOperation::particle_random:
        case ValueOperation::back_facing:
        case ValueOperation::random_per_island:
        case ValueOperation::image_color:
        case ValueOperation::image_alpha:
        case ValueOperation::attribute_color:
        case ValueOperation::attribute_alpha:
        case ValueOperation::normal_map:
        case ValueOperation::bump:
        case ValueOperation::noise_factor:
        case ValueOperation::noise_color:
        case ValueOperation::white_noise_value:
        case ValueOperation::white_noise_color:
        case ValueOperation::checker_color:
        case ValueOperation::checker_factor:
        case ValueOperation::brick_color:
        case ValueOperation::brick_factor:
        case ValueOperation::gradient:
        case ValueOperation::nishita_sky:
            return true;
        default:
            return false;
    }
}

[[nodiscard]] bool surface_is_spatially_varying(
    const compiler::SurfaceProgram &program) noexcept {
    return std::any_of(
        program.value_instructions().begin(),
        program.value_instructions().end(),
        [](const compiler::ValueInstruction &instruction) {
            return is_spatial_source(instruction.operation);
        });
}

[[nodiscard]] Vec3f normalized_or_z(
    Vec3f direction) noexcept {
    const auto length_squared =
        direction.x * direction.x +
        direction.y * direction.y +
        direction.z * direction.z;
    if (!(length_squared > 1.0e-20f) ||
        !std::isfinite(length_squared)) {
        return {0.0f, 0.0f, 1.0f};
    }
    const auto inverse_length =
        1.0f / std::sqrt(length_squared);
    return {
        direction.x * inverse_length,
        direction.y * inverse_length,
        direction.z * inverse_length};
}

void configure_background_sampling(
    LuisaSceneData &data,
    const SceneSnapshot &snapshot,
    bool include_environment) noexcept {
    data.background_map_width = 1u;
    data.background_map_height = 1u;
    data.background_map_weight =
        include_environment ? 1.0f : 0.0f;
    data.background_guided_sun_weight = 0.0f;
    data.background_guided_sun_axis =
        luisa::make_float3(0.0f, 0.0f, 1.0f);
    data.background_guided_sun_radius = 0.0f;

    if (!include_environment) {
        return;
    }

    // Match Cycles' single-Sun guidance contract. Multiple solar discs stay
    // in the importance map because one analytic cone cannot represent their
    // support without changing the estimator.
    if (data.nishita_environment &&
        data.nishita_environment->angular_radius > 0.0f) {
        data.background_guided_sun_weight = 4.0f;
        data.background_guided_sun_axis =
            data.nishita_environment->sun_direction;
        data.background_guided_sun_radius =
            data.nishita_environment->angular_radius;
    } else if (data.environment_suns.size() == 1u &&
               data.environment_suns.front()
                       .angular_radius >
                   0.0f) {
        const auto &sun = data.environment_suns.front();
        const auto axis = normalized_or_z(sun.direction);
        data.background_guided_sun_weight = 4.0f;
        data.background_guided_sun_axis =
            luisa::make_float3(axis.x, axis.y, axis.z);
        data.background_guided_sun_radius =
            sun.angular_radius;
    }

    if (snapshot.world_sampling ==
        contract::WorldSampling::manual) {
        data.background_map_width =
            std::max(
                snapshot.world_sample_map_resolution,
                2u);
        data.background_map_height =
            std::max(
                data.background_map_width / 2u,
                1u);
        return;
    }

    if (data.nishita_environment &&
        data.background_guided_sun_weight > 0.0f) {
        // Cycles raises an automatically sized guided Nishita map to this
        // resolution even though the atmosphere LUT itself is smaller.
        data.background_map_width = 512u;
        data.background_map_height = 256u;
    } else if (
        snapshot.environment &&
        snapshot.environment->width > 0u &&
        snapshot.environment->height > 0u) {
        data.background_map_width =
            snapshot.environment->width;
        data.background_map_height =
            snapshot.environment->height;
    } else {
        data.background_map_width = 1024u;
        data.background_map_height = 512u;
    }
}

void build_background_sampling_distribution(
    const std::shared_ptr<LuisaSceneData> &data,
    Stream &stream) {
    std::vector<Vec3f> radiance;
    if (data->background_map_weight > 0.0f) {
        const auto pixel_count =
            static_cast<std::size_t>(
                data->background_map_width) *
            static_cast<std::size_t>(
                data->background_map_height);
        auto radiance_buffer =
            data->device.create_buffer<luisa::float4>(
                pixel_count);
        luisa::vector<luisa::float4> readback(
            pixel_count);

        SafeNormalizeCallable safe_normalize =
            [](Float3 value,
               Float3 fallback) noexcept {
                const auto length_squared =
                    dot(value, value);
                return select(
                    fallback,
                    value /
                        sqrt(max(
                            length_squared,
                            1.0e-20f)),
                    length_squared > 1.0e-20f);
            };
        auto surface_callables =
            make_surface_callables(data);
        auto surface_emission =
            surface_callables.emission;
        auto environment_callables =
            make_environment_callables(
                data,
                safe_normalize,
                surface_emission);
        auto environment_base =
            environment_callables.base;
        auto environment_suns =
            environment_callables.suns;
        auto nishita_sun =
            environment_callables.nishita_sun;
        const auto width =
            data->background_map_width;
        const auto height =
            data->background_map_height;
        const auto include_discrete_suns =
            data->background_guided_sun_weight <=
            0.0f;
        const auto background = data->background;

        Kernel2D evaluate_importance = [
            =,
            &surface_emission](
            BufferFloat4 output) noexcept {
            set_block_size(8u, 8u, 1u);
            const auto coordinate =
                dispatch_id().xy();
            const auto u =
                (cast<float>(coordinate.x) + 0.5f) /
                static_cast<float>(width);
            const auto v =
                (cast<float>(coordinate.y) + 0.5f) /
                static_cast<float>(height);
            const auto direction =
                background_sampling::
                    equirectangular_to_direction(
                        u, v);
            Float3 value = environment_base(
                direction,
                make_float3(background),
                pack_shader_evaluation_state(
                    cycles_path_state::
                        light_emission_shader_state(
                            0u, 0u, 0u, 0u, 0u)));
            if (include_discrete_suns) {
                for (const auto &sun :
                     environment_suns) {
                    value += sun(direction);
                }
                value += nishita_sun(direction);
            }
            output.write(
                coordinate.y * width +
                    coordinate.x,
                make_float4(value, 1.0f));
        };
        auto importance_shader =
            data->device.compile(
                evaluate_importance);
        stream
            << importance_shader(
                   radiance_buffer)
                   .dispatch(width, height)
            << radiance_buffer.copy_to(
                   luisa::span{readback})
            << synchronize();

        radiance.reserve(readback.size());
        for (const auto value : readback) {
            const auto finite_or_zero =
                [](float component) noexcept {
                    return std::isfinite(component)
                               ? component
                               : 0.0f;
                };
            radiance.emplace_back(
                finite_or_zero(value.x),
                finite_or_zero(value.y),
                finite_or_zero(value.z));
        }
    } else {
        radiance.emplace_back(1.0f, 1.0f, 1.0f);
    }

    const auto distribution =
        sampling::
            build_cycles_background_map_distribution(
                radiance,
                data->background_map_width,
                data->background_map_height);
    luisa::vector<luisa::float2> conditional;
    conditional.reserve(
        distribution.conditional.size());
    for (const auto entry :
         distribution.conditional) {
        conditional.emplace_back(
            entry.function,
            entry.cumulative);
    }
    luisa::vector<luisa::float2> marginal;
    marginal.reserve(
        distribution.marginal.size());
    for (const auto entry :
         distribution.marginal) {
        marginal.emplace_back(
            entry.function,
            entry.cumulative);
    }
    data->background_conditional_cdf =
        data->device.create_buffer<luisa::float2>(
            conditional.size());
    data->background_marginal_cdf =
        data->device.create_buffer<luisa::float2>(
            marginal.size());
    stream
        << data->background_conditional_cdf
               .copy_from(luisa::span{conditional})
        << data->background_marginal_cdf
               .copy_from(luisa::span{marginal})
        << synchronize();
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
            diagnose(
                result.diagnostics,
                "Material " + std::to_string(id.value) +
                    " ('" + material.name +
                    "') has a Volume closure graph, but the Luisa "
                    "volume-stack integrator is not enabled yet.");
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

    luisa::vector<luisa::float4> parameters;
    std::map<std::uint64_t, std::uint32_t>
        surface_tags_by_signature;
    for (const auto &[id, material] :
         data->materials.materials()) {
        const auto base =
            static_cast<std::uint32_t>(parameters.size());
        const auto signature =
            material.surface_program()->structure_signature();
        auto [surface_iter, inserted] =
            surface_tags_by_signature.try_emplace(signature, 0u);
        if (inserted) {
            surface_iter->second =
                data->surfaces.create<GraphSurface>(
                    material.surface_program());
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
                    data->surfaces
                            .capabilities(
                                surface_iter->second)
                            .may_have_volume
                        ? material_flag_has_volume
                        : 0u});
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
            parameters.emplace_back(
                value != nullptr
                    ? parameter_value(*value)
                    : luisa::make_float4(0.0f));
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
        }
    }
    if (parameters.empty()) {
        parameters.emplace_back(luisa::make_float4(0.0f));
    }
    data->parameter_buffer =
        data->device.create_buffer<luisa::float4>(
            parameters.size());

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
    stream << data->parameter_buffer.copy_from(
                  luisa::span{parameters})
           << data->cycles_bsdf_table_buffer.copy_from(
                  luisa::span{cycles_bsdf_values});

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
        int width = 0;
        int height = 0;
        int channels = 0;
        auto *decoded = stbi_load_from_memory(
            image.encoded_data.data(),
            static_cast<int>(image.encoded_data.size()),
            &width,
            &height,
            &channels,
            STBI_rgb_alpha);
        if (decoded == nullptr || width <= 0 || height <= 0) {
            diagnose(
                result.diagnostics,
                "Failed to decode image '" + image.name + "'.");
            stbi_image_free(decoded);
            continue;
        }
        const auto pixel_bytes =
            static_cast<std::size_t>(width) *
            static_cast<std::size_t>(height) * 4u;
        auto &pixels =
            texture_uploads.emplace_back(pixel_bytes);
        std::memcpy(pixels.data(), decoded, pixel_bytes);
        stbi_image_free(decoded);
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
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height)));
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
        auto &upload = uploads.emplace_back();
        upload.positions.reserve(geometry.positions.size());
        auto bounds_min = geometry.positions.front();
        auto bounds_max = geometry.positions.front();
        for (const auto position : geometry.positions) {
            bounds_min.x = std::min(bounds_min.x, position.x);
            bounds_min.y = std::min(bounds_min.y, position.y);
            bounds_min.z = std::min(bounds_min.z, position.z);
            bounds_max.x = std::max(bounds_max.x, position.x);
            bounds_max.y = std::max(bounds_max.y, position.y);
            bounds_max.z = std::max(bounds_max.z, position.z);
        }
        const auto generated_fallback = [&](
                                            Vec3f position) noexcept {
            const auto map_axis = [](
                                      float value,
                                      float minimum,
                                      float maximum) noexcept {
                const auto extent = maximum - minimum;
                return std::abs(extent) > 1.0e-20f
                           ? (value - minimum) / extent
                           : 0.5f;
            };
            return Vec3f{
                map_axis(position.x, bounds_min.x, bounds_max.x),
                map_axis(position.y, bounds_min.y, bounds_max.y),
                map_axis(position.z, bounds_min.z, bounds_max.z)};
        };
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
                    to_luisa(generated_fallback(position)));
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
                upload.attribute_domains});
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
        material_may_emit.emplace(
            material_id,
            sampling !=
                    contract::EmissionSampling::none &&
            std::any_of(
                material.surface_program()
                    ->closure_instructions()
                    .begin(),
                material.surface_program()
                    ->closure_instructions()
                    .end(),
                [](const compiler::ClosureInstruction &closure) {
                    return closure.operation ==
                           compiler::ClosureOperation::emission;
                }));
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
        instances.emplace_back(InstanceGpu{
            .geometry_index = geometry_iter->second,
            .override_offset = override_offset,
            .override_count =
                static_cast<std::uint32_t>(
                    instance.material_overrides.size()),
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
                instance.cycles_light_group});
        const auto &geometry =
            snapshot.geometries.at(instance.geometry);
        const auto light_visible =
            instance.visibility_mask ==
                ~std::uint32_t{0u} ||
            (instance.visibility_mask &
             (diffuse_visibility |
              glossy_visibility |
              transmission_visibility)) != 0u;
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
                                    .at(*material_id))});
                emissive_triangle_areas.emplace_back(
                    world_triangle_area(
                        instance.transform,
                        geometry.positions[triangle[0u]],
                        geometry.positions[triangle[1u]],
                        geometry.positions[triangle[2u]]));
            }
        }
        const auto visibility =
            instance.visibility_mask ==
                    ~std::uint32_t{0u}
                ? std::uint8_t{0xffu}
                : static_cast<std::uint8_t>(
                      instance.visibility_mask);
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

    bool world_is_spatially_varying = false;
    if (snapshot.world_shader) {
        if (const auto *world_material =
                data->materials.find(*snapshot.world_shader)) {
            world_is_spatially_varying =
                surface_is_spatially_varying(
                    *world_material->surface_program());
        }
    }
    const auto include_environment =
        data->world_surface.has_value() &&
        (snapshot.world_sampling ==
             contract::WorldSampling::manual ||
         (snapshot.world_sampling ==
              contract::WorldSampling::automatic &&
          world_is_spatially_varying));
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
