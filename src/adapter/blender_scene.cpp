#include "blender_scene_geometry.h"
#include "blender_scene_internal.h"

#include <psycles/core/parallel.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace psycles::adapter {

namespace {

using contract::CameraDesc;
using contract::CameraId;
using contract::CameraProjection;
using contract::CameraSensorFit;
using contract::CurveGeometryDesc;
using contract::CurveShape;
using contract::DisplacementMethod;
using contract::EnvironmentDesc;
using contract::EnvironmentSunDesc;
using contract::EmissionSampling;
using contract::GeometryId;
using contract::ImageAlphaType;
using contract::ImageColorSpace;
using contract::ImageDesc;
using contract::ImageId;
using contract::InstanceId;
using contract::LightDesc;
using contract::LightId;
using contract::LightType;
using contract::MaterialDesc;
using contract::MaterialId;
using contract::MeshAttributeDomain;
using contract::NishitaSkyDesc;
using contract::SceneSnapshot;
using contract::ShaderDomain;
using contract::ShaderGraph;
using contract::SocketValue;
using contract::TriangleMeshDesc;
using contract::VolumeInterpolation;
using contract::VolumeSampling;
using contract::WorldSampling;

using detail::Document;
using detail::boolean;
using detail::cycles_default_surface_graph;
using detail::cycles_particle_source;
using detail::emission_graph;
using detail::find_simple_world_nishita;
using detail::float2;
using detail::float3;
using detail::matrix;
using detail::member;
using detail::normalized_material_graph;
using detail::number;
using detail::optional_unsigned_number;
using detail::ray_visibility_mask;
using detail::read_packed_values;
using detail::section_offset;
using detail::signed_number;
using detail::text;
using detail::unsigned_number;

[[nodiscard]] ImageColorSpace image_color_space(
    std::string_view name) noexcept {
    if (name == "sRGB") {
        return ImageColorSpace::srgb;
    }
    if (name == "Linear" || name == "Linear Rec.709" ||
        name == "Linear Rec.2020") {
        return ImageColorSpace::linear;
    }
    return ImageColorSpace::data;
}

[[nodiscard]] ImageAlphaType image_alpha_type(
    std::string_view name) noexcept {
    if (name == "PREMUL") {
        return ImageAlphaType::premultiplied;
    }
    if (name == "CHANNEL_PACKED") {
        return ImageAlphaType::channel_packed;
    }
    if (name == "NONE" || name == "IGNORE") {
        return ImageAlphaType::ignore;
    }
    return ImageAlphaType::straight;
}

[[nodiscard]] EmissionSampling emission_sampling(
    std::string_view name) {
    if (name == "NONE") {
        return EmissionSampling::none;
    }
    if (name == "AUTO") {
        return EmissionSampling::automatic;
    }
    if (name == "FRONT") {
        return EmissionSampling::front;
    }
    if (name == "BACK") {
        return EmissionSampling::back;
    }
    if (name == "FRONT_BACK") {
        return EmissionSampling::front_back;
    }
    throw std::runtime_error(
        "unsupported Cycles emission sampling method: " +
        std::string{name});
}

[[nodiscard]] VolumeSampling volume_sampling(
    std::string_view name) {
    if (name == "DISTANCE") {
        return VolumeSampling::distance;
    }
    if (name == "EQUIANGULAR") {
        return VolumeSampling::equiangular;
    }
    if (name == "MULTIPLE_IMPORTANCE") {
        return VolumeSampling::multiple_importance;
    }
    throw std::runtime_error(
        "unsupported Cycles volume sampling method: " +
        std::string{name});
}

[[nodiscard]] VolumeInterpolation volume_interpolation(
    std::string_view name) {
    if (name == "LINEAR") {
        return VolumeInterpolation::linear;
    }
    if (name == "CUBIC") {
        return VolumeInterpolation::cubic;
    }
    throw std::runtime_error(
        "unsupported Cycles volume interpolation method: " +
        std::string{name});
}

[[nodiscard]] DisplacementMethod displacement_method(
    std::string_view name) {
    if (name == "BUMP") {
        return DisplacementMethod::bump;
    }
    if (name == "DISPLACEMENT") {
        return DisplacementMethod::displacement;
    }
    if (name == "BOTH") {
        return DisplacementMethod::both;
    }
    throw std::runtime_error(
        "unsupported Cycles displacement method: " +
        std::string{name});
}

[[nodiscard]] WorldSampling world_sampling(
    std::string_view name) {
    if (name == "NONE") {
        return WorldSampling::none;
    }
    if (name == "AUTOMATIC") {
        return WorldSampling::automatic;
    }
    if (name == "MANUAL") {
        return WorldSampling::manual;
    }
    throw std::runtime_error(
        "unsupported Cycles world sampling method: " +
        std::string{name});
}

[[nodiscard]] std::filesystem::path utf8_path(std::string_view value) {
    return std::filesystem::path{std::u8string_view{
        reinterpret_cast<const char8_t *>(value.data()), value.size()}};
}

[[nodiscard]] std::string path_to_utf8(const std::filesystem::path &path) {
    const auto u8 = path.u8string();
    return std::string{
        reinterpret_cast<const char *>(u8.data()), u8.size()};
}

[[nodiscard]] std::vector<std::uint8_t> read_file(
    const std::filesystem::path &path) {
    std::ifstream stream{path, std::ios::binary | std::ios::ate};
    if (!stream) {
        throw std::runtime_error(
            "failed to open file: " + path_to_utf8(path));
    }
    const auto end = stream.tellg();
    if (end < 0) {
        throw std::runtime_error(
            "failed to size file: " + path_to_utf8(path));
    }
    std::vector<std::uint8_t> result(
        static_cast<std::size_t>(end));
    stream.seekg(0);
    stream.read(
        reinterpret_cast<char *>(result.data()),
        static_cast<std::streamsize>(result.size()));
    if (!stream) {
        throw std::runtime_error(
            "failed to read file: " + path_to_utf8(path));
    }
    return result;
}

[[nodiscard]] std::vector<Vec3f> read_float3_file(
    const std::filesystem::path &path,
    std::uint32_t width,
    std::uint32_t height) {
    const auto count =
        static_cast<std::size_t>(width) *
        static_cast<std::size_t>(height);
    if (width == 0u || height == 0u ||
        count >
            std::numeric_limits<std::size_t>::max() /
                (3u * sizeof(float))) {
        throw std::runtime_error(
            "invalid float3 image dimensions: " + path_to_utf8(path));
    }
    const auto bytes = read_file(path);
    const auto expected = count * 3u * sizeof(float);
    if (bytes.size() != expected) {
        throw std::runtime_error(
            "float3 image byte count mismatch: " + path_to_utf8(path));
    }
    std::vector<Vec3f> result(count);
    for (std::size_t i = 0u; i < count; ++i) {
        std::array<float, 3u> value{};
        std::memcpy(
            value.data(),
            bytes.data() + i * 3u * sizeof(float),
            3u * sizeof(float));
        result[i] = {value[0u], value[1u], value[2u]};
    }
    return result;
}

[[nodiscard]] BlenderSceneImport load_blender_scene_from_document(
    yyjson_val *root,
    std::span<const std::byte> geometry_bin,
    const std::filesystem::path &directory) {
    BlenderSceneImport result;
    auto error = [&](std::string message) {
        result.diagnostics.emplace_back(BlenderSceneDiagnostic{
            .severity = BlenderSceneDiagnosticSeverity::error,
            .message = std::move(message)});
    };
    auto warning = [&](std::string message) {
        result.diagnostics.emplace_back(BlenderSceneDiagnostic{
            .severity = BlenderSceneDiagnosticSeverity::warning,
            .message = std::move(message)});
    };

    const auto schema = text(member(root, "schema"));
    const auto compact_geometry =
        schema == "psycles.blender-scene.v2";
    if (!compact_geometry &&
        schema != "psycles.blender-scene.v1") {
        error("unsupported Blender scene bundle schema");
        return result;
    }

    try {
        SceneSnapshot scene;
        scene.revision = 1u;
        auto *scene_cycles_sync = member(root, "cycles_sync");
        scene.cycles_object_count = optional_unsigned_number(
            member(scene_cycles_sync, "object_count"));
        scene.cycles_uses_light_linking = boolean(
            member(scene_cycles_sync, "uses_light_linking"), false);

        std::map<std::string, ImageId, std::less<>>
            image_ids;
        std::map<
            std::string,
            ImageColorSpace,
            std::less<>>
            image_color_spaces;
        std::map<
            std::string,
            ImageAlphaType,
            std::less<>>
            image_alpha_types;
        auto *images = member(root, "images");
        yyjson_arr_iter image_metadata_iterator =
            yyjson_arr_iter_with(images);
        std::uint64_t image_metadata_index = 1u;
        while (auto *image = yyjson_arr_iter_next(
                   &image_metadata_iterator)) {
            const auto name = text(member(image, "name"));
            const auto id = ImageId{image_metadata_index++};
            image_ids.emplace(name, id);
            image_color_spaces.emplace(
                name,
                image_color_space(
                    text(member(image, "colorspace"))));
            image_alpha_types.emplace(
                name,
                image_alpha_type(
                    text(member(image, "alpha_mode"))));
        }

        std::map<
            std::string,
            yyjson_val *,
            std::less<>>
            node_groups;
        auto *raw_node_groups = member(root, "node_groups");
        if (raw_node_groups != nullptr &&
            yyjson_is_arr(raw_node_groups)) {
            yyjson_arr_iter group_iterator =
                yyjson_arr_iter_with(raw_node_groups);
            while (auto *group =
                       yyjson_arr_iter_next(
                           &group_iterator)) {
                if (group == nullptr ||
                    yyjson_is_null(group)) {
                    continue;
                }
                node_groups.insert_or_assign(
                    text(member(group, "name")),
                    group);
            }
        }

        std::map<std::string, MaterialId, std::less<>>
            material_ids;
        const MaterialId default_material{1u};
        scene.materials.emplace(
            default_material,
            MaterialDesc{
                .name = "__cycles_default_surface__",
                .shader = cycles_default_surface_graph(),
                .cycles_shader_index = std::nullopt});

        auto *materials = member(root, "materials");
        std::vector<yyjson_val *> material_vals;
        {
            yyjson_arr_iter material_iterator =
                yyjson_arr_iter_with(materials);
            while (auto *material =
                       yyjson_arr_iter_next(&material_iterator)) {
                material_vals.push_back(material);
            }
        }
        std::uint64_t material_index = 2u;
        for (auto *material : material_vals) {
            const auto id = MaterialId{material_index++};
            material_ids.emplace(text(member(material, "name")), id);
        }
        std::vector<MaterialDesc> parsed_materials(material_vals.size());
        std::vector<std::vector<BlenderSceneDiagnostic>> material_diags(
            material_vals.size());
        psycles::parallel_for(material_vals.size(), [&](std::size_t i) {
            auto *material = material_vals[i];
            auto *cycles_sync = member(material, "cycles_sync");
            const auto authored_displacement_method =
                displacement_method(text(
                    member(material, "displacement_method"),
                    "BUMP"));
            parsed_materials[i] = MaterialDesc{
                .name = text(member(material, "name")),
                .shader = normalized_material_graph(
                    material,
                    image_ids,
                    image_color_spaces,
                    image_alpha_types,
                    node_groups,
                    material_diags[i]),
                .use_transparent_shadow =
                    boolean(member(material, "use_transparent_shadow"), true),
                .use_bump_map_correction =
                    boolean(member(material, "use_bump_map_correction"), true),
                .emission_sampling =
                    emission_sampling(text(
                        member(material, "emission_sampling"), "AUTO")),
                .volume_sampling =
                    volume_sampling(text(
                        member(material, "volume_sampling"),
                        "MULTIPLE_IMPORTANCE")),
                .volume_interpolation =
                    volume_interpolation(text(
                        member(material, "volume_interpolation"), "LINEAR")),
                .displacement_method = authored_displacement_method,
                .cycles_shader_index =
                    optional_unsigned_number(
                        member(cycles_sync, "shader_index")),
                .cycles_pass_id = static_cast<std::int32_t>(
                    signed_number(member(cycles_sync, "pass_id"), 0))};
        });
        for (std::size_t i = 0u; i < parsed_materials.size(); ++i) {
            for (auto &diag : material_diags[i]) {
                result.diagnostics.emplace_back(std::move(diag));
            }
            scene.materials.emplace(
                MaterialId{2u + static_cast<std::uint64_t>(i)},
                std::move(parsed_materials[i]));
        }

        std::vector<yyjson_val *> image_vals;
        {
            yyjson_arr_iter image_iterator = yyjson_arr_iter_with(images);
            while (auto *image = yyjson_arr_iter_next(&image_iterator)) {
                image_vals.push_back(image);
            }
        }
        std::vector<ImageDesc> parsed_images(image_vals.size());
        psycles::parallel_for(image_vals.size(), [&](std::size_t i) {
            auto *image = image_vals[i];
            const auto load_failed = boolean(member(image, "load_failed"));
            auto *binary_section = member(image, "binary");
            const auto has_binary =
                binary_section != nullptr && !yyjson_is_null(binary_section);
            std::filesystem::path source_path;
            std::string source_format;
            std::vector<std::uint8_t> encoded_data;
            if (!load_failed && has_binary) {
                const auto byte_count = static_cast<std::size_t>(
                    unsigned_number(member(binary_section, "bytes")));
                encoded_data = read_packed_values<std::uint8_t>(
                    geometry_bin, section_offset(image, "binary"), byte_count);
                source_format = text(member(image, "source_format"), ".png");
            } else if (!load_failed) {
                const auto relative_path =
                    utf8_path(text(member(image, "path")));
                source_path = relative_path.is_absolute()
                                  ? relative_path
                                  : (directory / relative_path);
                source_format = path_to_utf8(source_path.extension());
                encoded_data = read_file(source_path);
            } else {
                source_format = text(member(image, "source_format"));
            }
            parsed_images[i] = ImageDesc{
                .name = text(member(image, "name")),
                .source_format = std::move(source_format),
                .color_space = image_color_space(
                    text(member(image, "colorspace"))),
                .alpha_type = image_alpha_type(
                    text(member(image, "alpha_mode"))),
                .width = static_cast<std::uint32_t>(
                    unsigned_number(member(image, "width"))),
                .height = static_cast<std::uint32_t>(
                    unsigned_number(member(image, "height"))),
                .encoded_data = std::move(encoded_data),
                .load_failed = load_failed};
        });
        for (std::size_t i = 0u; i < parsed_images.size(); ++i) {
            scene.images.emplace(
                ImageId{1u + static_cast<std::uint64_t>(i)},
                std::move(parsed_images[i]));
        }

        auto *render = member(root, "render");
        result.width = static_cast<std::uint32_t>(
            unsigned_number(member(render, "width"), 320u));
        result.height = static_cast<std::uint32_t>(
            unsigned_number(member(render, "height"), 240u));
        const auto percentage = static_cast<std::uint32_t>(
            unsigned_number(member(render, "percentage"), 100u));
        result.width =
            std::max(result.width * percentage / 100u, 1u);
        result.height =
            std::max(result.height * percentage / 100u, 1u);
        result.samples = static_cast<std::uint32_t>(
            unsigned_number(
                member(member(render, "cycles"), "samples"),
                64u));
        result.pass_alpha_threshold = std::clamp(
            number(
                member(render, "pass_alpha_threshold"),
                result.pass_alpha_threshold),
            0.0f,
            1.0f);
        auto *color_management =
            member(render, "color_management");
        result.color_management.display_device = text(
            member(color_management, "display_device"),
            result.color_management.display_device);
        result.color_management.view_transform = text(
            member(color_management, "view_transform"),
            result.color_management.view_transform);
        result.color_management.look = text(
            member(color_management, "look"),
            result.color_management.look);
        result.color_management.sequencer_color_space = text(
            member(color_management, "sequencer_color_space"),
            result.color_management.sequencer_color_space);
        result.color_management.exposure = number(
            member(color_management, "exposure"),
            result.color_management.exposure);
        result.color_management.gamma = std::max(
            number(
                member(color_management, "gamma"),
                result.color_management.gamma),
            1.0e-6f);
        result.color_management.use_curve_mapping = boolean(
            member(color_management, "use_curve_mapping"),
            result.color_management.use_curve_mapping);
        auto *shader_transforms =
            member(color_management, "shader_transforms");
        scene.shader_color_space.blender_luma = float3(
            member(shader_transforms, "blender_luma"),
            scene.shader_color_space.blender_luma);
        auto *xyz_to_rgb =
            member(shader_transforms, "xyz_to_rgb");
        auto *rec709_to_rgb =
            member(shader_transforms, "rec709_to_rgb");
        if (
            yyjson_is_arr(xyz_to_rgb) &&
            yyjson_arr_size(xyz_to_rgb) >= 3u) {
            scene.shader_color_space.xyz_to_r = float3(
                yyjson_arr_get(xyz_to_rgb, 0u),
                scene.shader_color_space.xyz_to_r);
            scene.shader_color_space.xyz_to_g = float3(
                yyjson_arr_get(xyz_to_rgb, 1u),
                scene.shader_color_space.xyz_to_g);
            scene.shader_color_space.xyz_to_b = float3(
                yyjson_arr_get(xyz_to_rgb, 2u),
                scene.shader_color_space.xyz_to_b);
        }
        if (
            yyjson_is_arr(rec709_to_rgb) &&
            yyjson_arr_size(rec709_to_rgb) >= 3u) {
            scene.shader_color_space.rec709_to_r = float3(
                yyjson_arr_get(rec709_to_rgb, 0u),
                scene.shader_color_space.rec709_to_r);
            scene.shader_color_space.rec709_to_g = float3(
                yyjson_arr_get(rec709_to_rgb, 1u),
                scene.shader_color_space.rec709_to_g);
            scene.shader_color_space.rec709_to_b = float3(
                yyjson_arr_get(rec709_to_rgb, 2u),
                scene.shader_color_space.rec709_to_b);
        }
        auto *cycles = member(render, "cycles");
        result.seed = static_cast<std::uint32_t>(
            unsigned_number(
                member(cycles, "effective_seed"),
                unsigned_number(member(cycles, "seed"))));
        result.adaptive_sampling = boolean(
            member(cycles, "use_adaptive_sampling"));
        result.denoising =
            boolean(member(cycles, "use_denoising"));
        result.integrator.max_bounces =
            static_cast<std::uint32_t>(unsigned_number(
                member(cycles, "max_bounces"),
                result.integrator.max_bounces));
        result.integrator.min_bounces =
            static_cast<std::uint32_t>(unsigned_number(
                member(cycles, "min_light_bounces"),
                result.integrator.min_bounces));
        result.integrator.diffuse_bounces =
            static_cast<std::uint32_t>(unsigned_number(
                member(cycles, "diffuse_bounces"),
                result.integrator.diffuse_bounces));
        result.integrator.glossy_bounces =
            static_cast<std::uint32_t>(unsigned_number(
                member(cycles, "glossy_bounces"),
                result.integrator.glossy_bounces));
        result.integrator.transmission_bounces =
            static_cast<std::uint32_t>(unsigned_number(
                member(cycles, "transmission_bounces"),
                result.integrator.transmission_bounces));
        result.integrator.volume_bounces =
            static_cast<std::uint32_t>(unsigned_number(
                member(cycles, "volume_bounces"),
                result.integrator.volume_bounces));
        result.integrator.transparent_min_bounces =
            static_cast<std::uint32_t>(unsigned_number(
                member(cycles, "min_transparent_bounces"),
                result.integrator.transparent_min_bounces));
        result.integrator.transparent_max_bounces =
            static_cast<std::uint32_t>(unsigned_number(
                member(cycles, "transparent_max_bounces"),
                result.integrator.transparent_max_bounces));
        const auto use_fast_gi = boolean(
            member(cycles, "use_fast_gi"), false);
        const auto fast_gi_method = text(
            member(cycles, "fast_gi_method"), "REPLACE");
        const auto ambient_occlusion_factor = std::max(
            number(member(cycles, "ao_factor"), 1.0f), 0.0f);
        if (use_fast_gi && fast_gi_method == "REPLACE") {
            result.integrator.ambient_occlusion_factor =
                ambient_occlusion_factor;
            if (ambient_occlusion_factor != 0.0f) {
                result.integrator.ambient_occlusion_bounces =
                    static_cast<std::uint32_t>(unsigned_number(
                        member(cycles, "ao_bounces_render")));
            }
        } else if (use_fast_gi && fast_gi_method == "ADD") {
            result.integrator.ambient_occlusion_additive_factor =
                ambient_occlusion_factor;
            if (ambient_occlusion_factor != 0.0f) {
                warning(
                    "Cycles Fast GI ADD is preserved in the scene contract, "
                    "but additive surface AO transport is not implemented "
                    "by the Luisa path tracer");
            }
        } else if (use_fast_gi) {
            warning(
                "unsupported Cycles Fast GI method '" +
                fast_gi_method + "'; Fast GI is disabled");
        }
        result.integrator.sample_clamp_direct = std::max(
            number(
                member(cycles, "sample_clamp_direct"),
                result.integrator.sample_clamp_direct),
            0.0f);
        result.integrator.sample_clamp_indirect = std::max(
            number(
                member(cycles, "sample_clamp_indirect"),
                result.integrator.sample_clamp_indirect),
            0.0f);
        result.integrator.filter_glossy = std::max(
            number(
                member(cycles, "blur_glossy"),
                result.integrator.filter_glossy),
            0.0f);
        scene.ambient_occlusion_distance = std::max(
            number(
                member(cycles, "ao_distance"),
                scene.ambient_occlusion_distance),
            0.0f);
        result.integrator.ambient_occlusion_distance =
            scene.ambient_occlusion_distance;
        result.integrator.film_exposure = std::max(
            number(
                member(cycles, "film_exposure"),
                result.integrator.film_exposure),
            0.0f);
        result.integrator.light_sampling_threshold = std::max(
            number(
                member(cycles, "light_sampling_threshold"),
                result.integrator.light_sampling_threshold),
            0.0f);
        result.integrator.reflective_caustics = boolean(
            member(cycles, "caustics_reflective"),
            result.integrator.reflective_caustics);
        result.integrator.refractive_caustics = boolean(
            member(cycles, "caustics_refractive"),
            result.integrator.refractive_caustics);
        result.integrator.use_light_tree = boolean(
            member(cycles, "use_light_tree"),
            result.integrator.use_light_tree);
        const auto direct_light_sampling = text(
            member(cycles, "direct_light_sampling_type"),
            "MULTIPLE_IMPORTANCE_SAMPLING");
        if (direct_light_sampling ==
            "MULTIPLE_IMPORTANCE_SAMPLING") {
            result.integrator.direct_light_sampling =
                contract::DirectLightSampling::
                    multiple_importance_sampling;
        } else if (
            direct_light_sampling ==
            "FORWARD_PATH_TRACING") {
            result.integrator.direct_light_sampling =
                contract::DirectLightSampling::
                    forward_path_tracing;
        } else if (
            direct_light_sampling ==
            "NEXT_EVENT_ESTIMATION") {
            result.integrator.direct_light_sampling =
                contract::DirectLightSampling::
                    next_event_estimation;
        } else {
            throw std::runtime_error(
                "unsupported Cycles direct light sampling type: " +
                direct_light_sampling);
        }
        if (result.adaptive_sampling) {
            warning(
                "Cycles adaptive sampling is enabled in the source scene; "
                "Psycles currently renders a fixed sample count. Official "
                "differential goldens must disable adaptive sampling.");
        }
        if (result.denoising) {
            warning(
                "Cycles denoising is enabled in the source scene; Psycles "
                "outputs un-denoised linear passes. Official differential "
                "goldens must disable denoising.");
        }
        result.transparent_background =
            boolean(member(render, "transparent"));
        const auto requested_filter_width = std::max(
            number(member(render, "filter_width"), 1.0f),
            1.0e-5f);
        const auto pixel_filter_type =
            text(member(render, "pixel_filter_type"));
        if (pixel_filter_type.empty() ||
            pixel_filter_type == "BOX") {
            result.pixel_filter =
                contract::PixelFilter::box;
            // Cycles defines BOX as one complete pixel regardless of the
            // dormant Blender filter_width property. Normalize legacy or
            // third-party bundles here as well as in the exporter.
            result.filter_width = 1.0f;
        } else if (pixel_filter_type == "GAUSSIAN") {
            result.pixel_filter =
                contract::PixelFilter::gaussian;
            result.filter_width = requested_filter_width;
        } else if (
            pixel_filter_type == "BLACKMAN_HARRIS") {
            result.pixel_filter =
                contract::PixelFilter::blackman_harris;
            result.filter_width = requested_filter_width;
        } else {
            throw std::runtime_error(
                "unsupported Cycles pixel filter: " +
                pixel_filter_type);
        }

        auto *camera = member(root, "camera");
        if (camera == nullptr || yyjson_is_null(camera)) {
            throw std::runtime_error(
                "bundle contains no active camera");
        }
        const CameraId camera_id{1u};
        const auto camera_type = text(member(camera, "type"));
        const auto sensor_fit_name =
            text(member(camera, "sensor_fit"));
        const auto sensor_fit =
            sensor_fit_name == "HORIZONTAL"
                ? CameraSensorFit::horizontal
                : sensor_fit_name == "VERTICAL"
                      ? CameraSensorFit::vertical
                      : CameraSensorFit::automatic;
        auto *depth_of_field = member(camera, "dof");
        const auto depth_of_field_enabled =
            boolean(member(depth_of_field, "enabled"));
        const auto f_stop = std::max(
            number(member(depth_of_field, "fstop"), 2.8f),
            1.0e-5f);
        const auto aperture_radius =
            depth_of_field_enabled
                ? camera_type == "ORTHO"
                      ? 1.0f / (2.0f * f_stop)
                      : number(member(camera, "lens"), 50.0f) *
                            1.0e-3f / (2.0f * f_stop)
                : 0.0f;
        std::optional<contract::CameraSensor> camera_sensor;
        if (member(camera, "sensor_width") != nullptr &&
            member(camera, "sensor_height") != nullptr &&
            member(camera, "lens") != nullptr) {
            camera_sensor = contract::CameraSensor{
                number(member(camera, "lens")),
                number(member(camera, "sensor_width")),
                number(member(camera, "sensor_height"))};
            if (!(camera_sensor->lens_mm > 0.0f && camera_sensor->width_mm > 0.0f &&
                  camera_sensor->height_mm > 0.0f)) {
                throw std::runtime_error("invalid physical camera sensor/lens");
            }
        }
        scene.cameras.emplace(
            camera_id,
            CameraDesc{
                .name = text(member(camera, "name")),
                .projection =
                    camera_type == "ORTHO"
                        ? CameraProjection::orthographic
                        : camera_type == "PANO"
                              ? CameraProjection::panorama
                              : CameraProjection::perspective,
                .transform =
                    matrix(member(camera, "transform")),
                .field_of_view = number(
                    member(camera, "angle_y"),
                    number(member(camera, "angle"), 0.7853982f)),
                .horizontal_field_of_view = number(
                    member(camera, "angle_x"),
                    number(member(camera, "angle"), 0.7853982f)),
                .sensor_fit = sensor_fit,
                .orthographic_scale = number(
                    member(camera, "ortho_scale"), 1.0f),
                .lens_shift_x = number(
                    member(camera, "shift_x"), 0.0f),
                .lens_shift_y = number(
                    member(camera, "shift_y"), 0.0f),
                .near_clip = number(
                    member(camera, "clip_start"), 1.0e-4f),
                .far_clip = number(
                    member(camera, "clip_end"), 1.0e5f),
                .aperture_radius = aperture_radius,
                .focal_distance = number(
                    member(depth_of_field, "focus_distance")),
                .aperture_blades =
                    static_cast<std::uint32_t>(
                        unsigned_number(
                            member(depth_of_field, "blades"))),
                .aperture_rotation = number(
                    member(depth_of_field, "rotation")),
                .aperture_ratio = std::max(
                    number(
                        member(depth_of_field, "ratio"),
                        1.0f),
                    1.0e-5f),
                .sensor = camera_sensor,
                .pixel_aspect = {
                    std::max(number(member(camera, "pixel_aspect_x"), 1.0f), 1.0e-5f),
                    std::max(number(member(camera, "pixel_aspect_y"), 1.0f), 1.0e-5f)}});
        scene.active_camera = camera_id;

        if (geometry_bin.size() < 8u) {
            throw std::runtime_error(
                "geometry binary buffer is too small");
        }
        const auto geometry_magic = std::string_view{
            reinterpret_cast<const char *>(geometry_bin.data()), 8u};
        const auto expected_magic =
            compact_geometry
                ? std::string_view{"PSYGEO2\0", 8u}
                : std::string_view{"PSYGEO1\0", 8u};
        if (geometry_magic != expected_magic) {
            throw std::runtime_error(
                "geometry.bin has an invalid header");
        }

        auto *geometries = member(root, "geometries");
        std::vector<yyjson_val *> mesh_vals;
        {
            yyjson_arr_iter geometry_iterator =
                yyjson_arr_iter_with(geometries);
            while (auto *geometry =
                       yyjson_arr_iter_next(&geometry_iterator)) {
                mesh_vals.push_back(geometry);
            }
        }
        std::vector<TriangleMeshDesc> parsed_meshes(mesh_vals.size());
        psycles::parallel_for(mesh_vals.size(), [&](std::size_t i) {
            parsed_meshes[i] = detail::parse_blender_triangle_mesh(
                mesh_vals[i],
                geometry_bin,
                compact_geometry,
                material_ids,
                default_material);
        });
        std::uint64_t geometry_index = 1u;
        for (std::size_t i = 0u; i < parsed_meshes.size(); ++i) {
            scene.geometries.emplace(
                GeometryId{geometry_index++},
                std::move(parsed_meshes[i]));
        }

        const auto mesh_geometry_count = geometry_index - 1u;
        auto *curve_geometries = member(root, "curve_geometries");
        if (curve_geometries != nullptr &&
            yyjson_is_arr(curve_geometries)) {
            std::vector<yyjson_val *> curve_vals;
            {
                yyjson_arr_iter curve_geometry_iterator =
                    yyjson_arr_iter_with(curve_geometries);
                while (auto *geometry = yyjson_arr_iter_next(
                           &curve_geometry_iterator)) {
                    curve_vals.push_back(geometry);
                }
            }
            std::vector<CurveGeometryDesc> parsed_curves(curve_vals.size());
            psycles::parallel_for(curve_vals.size(), [&](std::size_t i) {
                parsed_curves[i] = detail::parse_blender_curve_geometry(
                    curve_vals[i],
                    geometry_bin,
                    material_ids,
                    default_material);
            });
            for (std::size_t i = 0u; i < parsed_curves.size(); ++i) {
                scene.curve_geometries.emplace(
                    GeometryId{geometry_index++},
                    std::move(parsed_curves[i]));
            }
        }

        auto *instances = member(root, "instances");
        yyjson_arr_iter instance_iterator =
            yyjson_arr_iter_with(instances);
        std::uint64_t instance_index = 1u;
        while (auto *instance =
                   yyjson_arr_iter_next(&instance_iterator)) {
            const auto geometry =
                unsigned_number(member(instance, "geometry"));
            const auto geometry_type = text(
                member(instance, "geometry_type"), "MESH");
            const auto geometry_id =
                geometry_type == "CURVE"
                    ? GeometryId{
                          mesh_geometry_count + geometry + 1u}
                    : GeometryId{geometry + 1u};
            if (geometry_type != "MESH" &&
                geometry_type != "CURVE") {
                throw std::runtime_error(
                    "instance has unsupported geometry type '" +
                    geometry_type + "'");
            }
            auto *visibility = member(instance, "visibility");
            auto *cycles_sync =
                member(instance, "cycles_sync");
            scene.instances.emplace(
                InstanceId{instance_index++},
                contract::InstanceDesc{
                    .name = text(member(instance, "name")),
                    .cycles_asset_name = text(
                        member(instance, "asset_name")),
                    .geometry = geometry_id,
                    .transform =
                        matrix(member(instance, "transform")),
                    .motion = {},
                    .material_overrides = {},
                    .random = std::clamp(
                        number(member(instance, "random_id")) /
                            4294967295.0f,
                        0.0f,
                        1.0f),
                    .cycles_random_id = optional_unsigned_number(
                        member(instance, "random_id")),
                    .particle_index = static_cast<std::uint32_t>(
                        unsigned_number(
                            member(instance, "particle_index"))),
                    .shadow_terminator_geometry_offset =
                        std::max(
                            number(
                                member(
                                    instance,
                                    "shadow_terminator_geometry_offset")),
                            0.0f),
                    .ambient_occlusion_distance =
                        std::max(
                            number(
                                member(
                                    instance,
                                    "ambient_occlusion_distance")),
                            0.0f),
                    .visibility_mask =
                        ray_visibility_mask(visibility),
                    .cycles_object_index =
                        optional_unsigned_number(member(
                            cycles_sync,
                            "object_index")),
                    .cycles_light_group =
                        static_cast<std::int32_t>(
                            signed_number(
                                member(
                                    cycles_sync,
                                    "light_group"),
                                -1)),
                    .is_shadow_catcher =
                        boolean(
                            member(
                                instance,
                                "is_shadow_catcher"),
                            false),
                    .use_holdout = boolean(
                        member(instance, "use_holdout"), false),
                    .is_caustics_caster = boolean(
                        member(instance, "is_caustics_caster"), false),
                    .is_caustics_receiver = boolean(
                        member(instance, "is_caustics_receiver"), false),
                    .is_blender_instance =
                        boolean(
                            member(instance, "is_instance"),
                            false),
                    .object_color = float3(
                        member(instance, "object_color")),
                    .object_alpha = number(
                        member(instance, "object_alpha"), 0.0f),
                    .object_pass_id = static_cast<std::int32_t>(
                        signed_number(
                            member(instance, "object_pass_id"), 0)),
                    .dupli_generated = float3(
                        member(instance, "dupli_generated")),
                    .dupli_uv = float2(
                        member(instance, "dupli_uv")),
                    .shadow_terminator_shading_offset = number(
                        member(
                            instance,
                            "shadow_terminator_shading_offset"),
                        0.0f),
                    .cycles_particle_source = cycles_particle_source(
                        member(instance, "cycles_particle_source"))});
        }

        auto *lights = member(root, "lights");
        yyjson_arr_iter light_iterator =
            yyjson_arr_iter_with(lights);
        std::uint64_t light_index = 1u;
        while (auto *light =
                   yyjson_arr_iter_next(&light_iterator)) {
            const auto light_name = text(member(light, "name"));
            const auto light_shader_name = text(
                member(light, "shader_name"),
                light_name + " / Light Shader");
            const auto type = text(member(light, "type"));
            auto *cycles_sync =
                member(light, "cycles_sync");
            auto color = float3(
                member(light, "color"),
                {1.0f, 1.0f, 1.0f});
            const auto temperature_color = float3(
                member(light, "temperature_color"),
                {1.0f, 1.0f, 1.0f});
            color.x *= temperature_color.x;
            color.y *= temperature_color.y;
            color.z *= temperature_color.z;
            const auto size =
                number(member(light, "size"), 0.0f);
            const auto shape =
                text(member(light, "shape"), "POINT");
            // Blender stores size_y for every area-light shape, but Cycles
            // deliberately ignores it for SQUARE and DISK. Those shapes use
            // area_size on both axes; only RECTANGLE and ELLIPSE consume
            // area_sizey (BlenderSync::sync_light).
            const auto size_y =
                shape == "SQUARE" || shape == "DISK"
                    ? size
                    : number(member(light, "size_y"), size);
            std::optional<MaterialId> light_shader;
            auto *light_tree = member(light, "node_tree");
            if (light_tree != nullptr &&
                !yyjson_is_null(light_tree) &&
                member(light_tree, "surface_root") != nullptr &&
                !yyjson_is_null(
                    member(light_tree, "surface_root"))) {
                light_shader = MaterialId{material_index++};
                scene.materials.emplace(
                    *light_shader,
                    MaterialDesc{
                        .name = light_shader_name,
                        .shader =
                            normalized_material_graph(
                                light,
                                image_ids,
                                image_color_spaces,
                                image_alpha_types,
                                node_groups,
                                result.diagnostics),
                        .cycles_shader_index =
                            optional_unsigned_number(member(
                                cycles_sync,
                                "shader_index")),
                        .cycles_pass_id = static_cast<std::int32_t>(
                            signed_number(
                                member(cycles_sync, "pass_id"), 0))});
            }
            scene.lights.emplace(
                LightId{light_index++},
                LightDesc{
                    .name = light_name,
                    .cycles_asset_name = text(
                        member(light, "asset_name")),
                    .type =
                        type == "AREA"
                            ? LightType::area
                            : type == "SUN"
                                  ? LightType::distant
                                  : type == "SPOT"
                                        ? LightType::spot
                                        : LightType::point,
                    .transform =
                        matrix(member(light, "transform")),
                    .color = color,
                    .power =
                        number(member(light, "energy"), 1.0f) *
                        std::exp2(number(
                            member(light, "exposure"),
                            0.0f)),
                    .size = size,
                    .size_y = size_y,
                    .spread =
                        number(member(light, "spread"), 3.14159265f),
                    .spot_angle = number(
                        member(light, "spot_size"),
                        0.78539816339f),
                    .spot_smooth = number(
                        member(light, "spot_blend"),
                        0.15f),
                    .angle =
                        number(member(light, "angle"), 0.0f),
                    .normalize = boolean(
                        member(light, "normalize"), true),
                    .ellipse =
                        shape == "DISK" ||
                        shape == "ELLIPSE",
                    .is_sphere = !boolean(
                        member(light, "use_soft_falloff"),
                        false),
                    .is_portal = boolean(
                        member(light, "is_portal"), false),
                    .shader = light_shader,
                    .use_mis = boolean(
                        member(
                            light,
                            "use_multiple_importance_sampling"),
                        true),
                    .cast_shadow = boolean(
                        member(light, "cast_shadow"), true),
                    .visibility_mask =
                        ray_visibility_mask(
                            member(light, "visibility")),
                    .is_shadow_catcher = boolean(
                        member(
                            light,
                            "is_shadow_catcher")),
                    .use_holdout = boolean(
                        member(light, "use_holdout"), false),
                    .is_caustics_caster = boolean(
                        member(light, "is_caustics_caster"), false),
                    .is_caustics_receiver = boolean(
                        member(light, "is_caustics_receiver"), false),
                    .cycles_shader_index =
                        optional_unsigned_number(member(
                            cycles_sync,
                            "shader_index")),
                    .cycles_object_index =
                        optional_unsigned_number(member(
                            cycles_sync,
                            "object_index")),
                    .cycles_light_group =
                        static_cast<std::int32_t>(
                            signed_number(
                                member(
                                    cycles_sync,
                                    "light_group"),
                                -1)),
                    .max_bounces =
                        static_cast<std::uint32_t>(
                            unsigned_number(
                                member(light, "max_bounces"),
                                1024u)),
                    .object_color = float3(
                        member(light, "object_color")),
                    .object_alpha = number(
                        member(light, "object_alpha"), 0.0f),
                    .object_pass_id = static_cast<std::int32_t>(
                        signed_number(
                            member(light, "object_pass_id"), 0)),
                    .object_random = std::clamp(
                        number(member(light, "random_id")) /
                            4294967295.0f,
                        0.0f,
                        1.0f),
                    .cycles_random_id = optional_unsigned_number(
                        member(light, "random_id")),
                    .particle_index = static_cast<std::uint32_t>(
                        unsigned_number(
                            member(light, "particle_index"))),
                    .dupli_generated = float3(
                        member(light, "dupli_generated")),
                    .dupli_uv = float2(
                        member(light, "dupli_uv")),
                    .shadow_terminator_shading_offset = number(
                        member(
                            light,
                            "shadow_terminator_shading_offset"),
                        0.0f),
                    .shadow_terminator_geometry_offset = number(
                        member(
                            light,
                            "shadow_terminator_geometry_offset"),
                        0.1f),
                    .cycles_particle_source = cycles_particle_source(
                        member(light, "cycles_particle_source"))});
        }

        auto *world = member(root, "world");
        if (world != nullptr && !yyjson_is_null(world)) {
            scene.world_sampling = world_sampling(text(
                member(world, "sampling_method"),
                "AUTOMATIC"));
            scene.world_sample_map_resolution =
                static_cast<std::uint32_t>(
                    unsigned_number(
                        member(
                            world,
                            "sample_map_resolution"),
                        1024u));
            scene.world_max_bounces =
                static_cast<std::uint32_t>(
                    unsigned_number(
                        member(world, "max_bounces"),
                        1024u));
            scene.world_cast_shadow =
                boolean(
                    member(world, "use_shadows"),
                    true);
            scene.world_visibility_mask =
                ray_visibility_mask(
                    member(world, "visibility"));
            Vec3f world_color =
                float3(member(world, "color"), {0.05f, 0.05f, 0.05f});
            auto *tree = member(world, "node_tree");
            auto *world_cycles_sync =
                member(world, "cycles_sync");
            const auto world_id =
                MaterialId{material_index++};
            auto world_graph =
                tree != nullptr &&
                        !yyjson_is_null(tree) &&
                        member(tree, "surface_root") != nullptr
                    ? normalized_material_graph(
                          world,
                          image_ids,
                          image_color_spaces,
                          image_alpha_types,
                          node_groups,
                          result.diagnostics)
                    : emission_graph(world_color, 1.0f);
            scene.materials.emplace(
                world_id,
                MaterialDesc{
                    .name =
                        "__world__" + text(member(world, "name")),
                    .shader = std::move(world_graph),
                    .volume_sampling =
                        volume_sampling(text(
                            member(world, "volume_sampling"),
                            "MULTIPLE_IMPORTANCE")),
                    .volume_interpolation =
                        volume_interpolation(text(
                            member(world, "volume_interpolation"),
                            "LINEAR")),
                    .cycles_shader_index =
                        optional_unsigned_number(member(
                            world_cycles_sync,
                            "shader_index")),
                    .cycles_pass_id = static_cast<std::int32_t>(
                        signed_number(
                            member(world_cycles_sync, "pass_id"), 0))});
            scene.world_shader = world_id;
            scene.cycles_background_asset_name =
                text(member(world, "name"));
            scene.cycles_background_object_index =
                optional_unsigned_number(member(
                    world_cycles_sync,
                    "object_index"));
            scene.cycles_background_light_group =
                static_cast<std::int32_t>(
                    signed_number(
                        member(
                            world_cycles_sync,
                            "light_group"),
                        -1));
            if (auto nishita =
                    find_simple_world_nishita(world)) {
                scene.environment = EnvironmentDesc{
                    .name =
                        "__world_nishita__" +
                        text(member(world, "name")),
                    .width = 0u,
                    .height = 0u,
                    .pixels = {},
                    .suns = {},
                    .nishita = std::move(nishita)};
            }
        }

        auto *environment = member(root, "world_environment");
        if (environment != nullptr &&
            !yyjson_is_null(environment) &&
            (!scene.environment ||
             !scene.environment->nishita)) {
            const auto width = static_cast<std::uint32_t>(
                unsigned_number(member(environment, "width")));
            const auto height = static_cast<std::uint32_t>(
                unsigned_number(member(environment, "height")));
            EnvironmentDesc imported_environment{
                .name = text(
                    member(environment, "name"),
                    "__world_environment__"),
                .width = width,
                .height = height,
                .pixels = read_float3_file(
                    directory /
                        utf8_path(text(member(environment, "path"))),
                    width,
                    height),
                .suns = {},
                .nishita = std::nullopt};
            auto *suns = member(environment, "suns");
            yyjson_arr_iter sun_iterator =
                yyjson_arr_iter_with(suns);
            while (auto *sun =
                       yyjson_arr_iter_next(&sun_iterator)) {
                imported_environment.suns.emplace_back(
                    EnvironmentSunDesc{
                        .direction = float3(
                            member(sun, "direction"),
                            {0.0f, 0.0f, 1.0f}),
                        .radiance = float3(
                            member(sun, "radiance")),
                        .angular_radius = number(
                            member(sun, "angular_radius"))});
            }
            scene.environment =
                std::move(imported_environment);
        }

        result.scene = std::move(scene);
    } catch (const std::exception &exception) {
        error(exception.what());
    }
    return result;
}

}// namespace

BlenderSceneImport load_blender_scene_bundle(
    const std::filesystem::path &directory) {
    BlenderSceneImport result;
    auto error = [&](std::string message) {
        result.diagnostics.emplace_back(BlenderSceneDiagnostic{
            .severity = BlenderSceneDiagnosticSeverity::error,
            .message = std::move(message)});
    };
    const auto json_path = directory / "scene.json";
    try {
        auto json_bytes = read_file(json_path);
        yyjson_read_err read_error{};
        Document document{yyjson_read_opts(
            reinterpret_cast<char *>(json_bytes.data()),
            json_bytes.size(), 0u, nullptr, &read_error)};
        if (!document) {
            error("failed to parse '" + path_to_utf8(json_path) + "': " +
                  (read_error.msg == nullptr ? std::string{"unknown JSON error"}
                                             : std::string{read_error.msg}));
            return result;
        }
        const auto geometry_bytes = read_file(directory / "geometry.bin");
        return load_blender_scene_from_document(
            yyjson_doc_get_root(document.get()),
            std::as_bytes(std::span{geometry_bytes}),
            directory);
    } catch (const std::exception &exception) {
        error(exception.what());
        return result;
    }
}

BlenderSceneImport load_blender_scene_memory(
    std::string_view scene_json,
    std::span<const std::byte> geometry_bin,
    const std::filesystem::path &asset_base_dir) {
    BlenderSceneImport result;
    if (scene_json.empty()) {
        result.diagnostics.emplace_back(BlenderSceneDiagnostic{
            .severity = BlenderSceneDiagnosticSeverity::error,
            .message = "scene_json buffer is empty"});
        return result;
    }
    yyjson_read_err read_error{};
    Document document{yyjson_read_opts(
        const_cast<char *>(scene_json.data()),
        scene_json.size(), 0u, nullptr, &read_error)};
    if (!document) {
        result.diagnostics.emplace_back(BlenderSceneDiagnostic{
            .severity = BlenderSceneDiagnosticSeverity::error,
            .message = std::string{"failed to parse in-memory scene JSON: "} +
                       (read_error.msg == nullptr ? "unknown JSON error"
                                                  : read_error.msg)});
        return result;
    }
    return load_blender_scene_from_document(
        yyjson_doc_get_root(document.get()), geometry_bin, asset_base_dir);
}

}// namespace psycles::adapter
