#include "blender_scene_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
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
using detail::signed_number;
using detail::text;
using detail::unsigned_number;

[[nodiscard]] std::uint64_t section_offset(
    yyjson_val *geometry,
    const char *name) noexcept {
    return unsigned_number(member(member(geometry, name), "offset"));
}

[[nodiscard]] MeshAttributeDomain mesh_attribute_domain(
    yyjson_val *value,
    MeshAttributeDomain fallback =
        MeshAttributeDomain::point) {
    const auto name = text(value);
    if (name.empty()) {
        return fallback;
    }
    if (name == "POINT") {
        return MeshAttributeDomain::point;
    }
    if (name == "CORNER") {
        return MeshAttributeDomain::corner;
    }
    if (name == "FACE") {
        return MeshAttributeDomain::face;
    }
    throw std::runtime_error(
        "unsupported mesh attribute domain '" + name + "'");
}

[[nodiscard]] CurveShape curve_shape(yyjson_val *value) {
    const auto name = text(value);
    if (name == "RIBBON" || name == "RIBBONS") {
        return CurveShape::ribbon;
    }
    if (name == "THICK") {
        return CurveShape::thick;
    }
    if (name == "THICK_LINEAR") {
        return CurveShape::thick_linear;
    }
    throw std::runtime_error(
        "unsupported Cycles curve shape '" + name + "'");
}

template<typename T>
[[nodiscard]] std::vector<T> read_values(
    std::ifstream &stream,
    std::uint64_t offset,
    std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::runtime_error("binary scene section is too large");
    }
    std::vector<T> result(count);
    stream.clear();
    stream.seekg(static_cast<std::streamoff>(offset));
    stream.read(
        reinterpret_cast<char *>(result.data()),
        static_cast<std::streamsize>(count * sizeof(T)));
    if (!stream) {
        throw std::runtime_error("failed to read geometry.bin section");
    }
    return result;
}

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

[[nodiscard]] std::vector<std::uint8_t> read_file(
    const std::filesystem::path &path) {
    std::ifstream stream{path, std::ios::binary | std::ios::ate};
    if (!stream) {
        throw std::runtime_error(
            "failed to open file: " + path.string());
    }
    const auto end = stream.tellg();
    if (end < 0) {
        throw std::runtime_error(
            "failed to size file: " + path.string());
    }
    std::vector<std::uint8_t> result(
        static_cast<std::size_t>(end));
    stream.seekg(0);
    stream.read(
        reinterpret_cast<char *>(result.data()),
        static_cast<std::streamsize>(result.size()));
    if (!stream) {
        throw std::runtime_error(
            "failed to read file: " + path.string());
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
            "invalid float3 image dimensions: " + path.string());
    }
    const auto bytes = read_file(path);
    const auto expected = count * 3u * sizeof(float);
    if (bytes.size() != expected) {
        throw std::runtime_error(
            "float3 image byte count mismatch: " + path.string());
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

}// namespace

BlenderSceneImport load_blender_scene_bundle(
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

    const auto json_path = directory / "scene.json";
    yyjson_read_err read_error{};
    Document document{
        yyjson_read_file(
            json_path.string().c_str(),
            0u,
            nullptr,
            &read_error)};
    if (!document) {
        error(
            "failed to parse '" + json_path.string() +
            "': " +
            (read_error.msg == nullptr
                 ? std::string{"unknown JSON error"}
                 : std::string{read_error.msg}));
        return result;
    }
    auto *root = yyjson_doc_get_root(document.get());
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
        yyjson_arr_iter material_iterator =
            yyjson_arr_iter_with(materials);
        std::uint64_t material_index = 2u;
        while (auto *material =
                   yyjson_arr_iter_next(&material_iterator)) {
            const auto id = MaterialId{material_index++};
            const auto name = text(member(material, "name"));
            auto *cycles_sync =
                member(material, "cycles_sync");
            const auto authored_displacement_method =
                displacement_method(text(
                    member(material, "displacement_method"),
                    "BUMP"));
            material_ids.emplace(name, id);
            scene.materials.emplace(
                id,
                MaterialDesc{
                    .name = name,
                    .shader = normalized_material_graph(
                        material,
                        image_ids,
                        image_color_spaces,
                        image_alpha_types,
                        node_groups,
                        result.diagnostics),
                    .use_transparent_shadow =
                        boolean(member(
                                    material,
                                    "use_transparent_shadow"),
                                true),
                    .use_bump_map_correction =
                        boolean(member(
                                    material,
                                    "use_bump_map_correction"),
                            true),
                    .emission_sampling =
                        emission_sampling(text(
                            member(
                                material,
                                "emission_sampling"),
                            "AUTO")),
                    .volume_sampling =
                        volume_sampling(text(
                            member(
                                material,
                                "volume_sampling"),
                            "MULTIPLE_IMPORTANCE")),
                    .displacement_method =
                        authored_displacement_method,
                    .cycles_shader_index =
                        optional_unsigned_number(member(
                            cycles_sync,
                            "shader_index")),
                    .cycles_pass_id = static_cast<std::int32_t>(
                        signed_number(
                            member(cycles_sync, "pass_id"), 0))});
        }

        yyjson_arr_iter image_iterator =
            yyjson_arr_iter_with(images);
        std::uint64_t image_index = 1u;
        while (auto *image =
                   yyjson_arr_iter_next(&image_iterator)) {
            const auto relative_path =
                std::filesystem::path{
                    text(member(image, "path"))};
            const auto source_path =
                directory / relative_path;
            scene.images.emplace(
                ImageId{image_index++},
                ImageDesc{
                    .name = text(member(image, "name")),
                    .source_format =
                        source_path.extension().string(),
                    .color_space = image_color_space(
                        text(member(image, "colorspace"))),
                    .alpha_type = image_alpha_type(
                        text(member(image, "alpha_mode"))),
                    .width = static_cast<std::uint32_t>(
                        unsigned_number(
                            member(image, "width"))),
                    .height = static_cast<std::uint32_t>(
                        unsigned_number(
                            member(image, "height"))),
                    .encoded_data = read_file(source_path)});
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
                    1.0e-5f)});
        scene.active_camera = camera_id;

        std::ifstream geometry_stream{
            directory / "geometry.bin", std::ios::binary};
        if (!geometry_stream) {
            throw std::runtime_error(
                "failed to open geometry.bin");
        }
        std::array<char, 8u> magic{};
        geometry_stream.read(
            magic.data(),
            static_cast<std::streamsize>(magic.size()));
        const auto geometry_magic =
            std::string_view{magic.data(), magic.size()};
        const auto expected_magic =
            compact_geometry
                ? std::string_view{"PSYGEO2\0", 8u}
                : std::string_view{"PSYGEO1\0", 8u};
        if (!geometry_stream ||
            geometry_magic != expected_magic) {
            throw std::runtime_error(
                "geometry.bin has an invalid header");
        }

        auto *geometries = member(root, "geometries");
        yyjson_arr_iter geometry_iterator =
            yyjson_arr_iter_with(geometries);
        std::uint64_t geometry_index = 1u;
        while (auto *geometry =
                   yyjson_arr_iter_next(&geometry_iterator)) {
            const auto triangle_count = static_cast<std::size_t>(
                unsigned_number(
                    member(geometry, "triangle_count")));
            const auto point_count = static_cast<std::size_t>(
                unsigned_number(
                    member(
                        geometry,
                        compact_geometry
                            ? "point_count"
                            : "vertex_count")));
            const auto corner_count = triangle_count * 3u;
            if (compact_geometry &&
                static_cast<std::size_t>(
                    unsigned_number(
                        member(geometry, "corner_count"))) !=
                    corner_count) {
                throw std::runtime_error(
                    "compact geometry has an invalid corner count");
            }
            const auto domain_count =
                [&](MeshAttributeDomain domain) noexcept {
                    switch (domain) {
                        case MeshAttributeDomain::point:
                            return point_count;
                        case MeshAttributeDomain::corner:
                            return corner_count;
                        case MeshAttributeDomain::face:
                            return triangle_count;
                    }
                    return std::size_t{0u};
                };
            const auto normal_domain =
                compact_geometry
                    ? mesh_attribute_domain(
                          member(geometry, "normal_domain"))
                    : MeshAttributeDomain::point;
            const auto uv_domain =
                compact_geometry
                    ? mesh_attribute_domain(
                          member(geometry, "uv_domain"))
                    : MeshAttributeDomain::point;
            const auto uv_tangent_domain =
                compact_geometry
                    ? mesh_attribute_domain(
                          member(
                              geometry,
                              "uv_tangent_domain"))
                    : MeshAttributeDomain::point;
            const auto generated_domain =
                compact_geometry
                    ? mesh_attribute_domain(
                          member(
                              geometry,
                              "generated_domain"))
                    : MeshAttributeDomain::point;
            const auto normal_count =
                domain_count(normal_domain);
            const auto uv_count = domain_count(uv_domain);
            const auto uv_tangent_count =
                domain_count(uv_tangent_domain);
            const auto generated_count =
                domain_count(generated_domain);
            auto position_values = read_values<float>(
                geometry_stream,
                section_offset(geometry, "positions"),
                point_count * 3u);
            auto normal_values = read_values<float>(
                geometry_stream,
                section_offset(geometry, "normals"),
                normal_count * 3u);
            auto uv_values = read_values<float>(
                geometry_stream,
                section_offset(geometry, "uv"),
                uv_count * 2u);
            std::vector<float> uv_tangent_values;
            if (member(geometry, "uv_tangents") != nullptr) {
                uv_tangent_values = read_values<float>(
                    geometry_stream,
                    section_offset(geometry, "uv_tangents"),
                    uv_tangent_count * 4u);
            }
            auto generated_values = read_values<float>(
                geometry_stream,
                section_offset(geometry, "generated"),
                generated_count * 3u);
            auto index_values = read_values<std::uint32_t>(
                geometry_stream,
                section_offset(geometry, "indices"),
                triangle_count * 3u);
            auto material_values =
                read_values<std::uint32_t>(
                    geometry_stream,
                    section_offset(
                        geometry,
                        "triangle_material_slots"),
                    triangle_count);
            std::vector<std::uint32_t> smooth_values(
                triangle_count, 0u);
            if (member(geometry, "triangle_smooth") != nullptr) {
                smooth_values =
                    read_values<std::uint32_t>(
                        geometry_stream,
                        section_offset(
                            geometry,
                            "triangle_smooth"),
                        triangle_count);
            }
            auto random_per_island_values =
                read_values<float>(
                    geometry_stream,
                    section_offset(
                        geometry,
                        "triangle_random_per_island"),
                    triangle_count);
            std::vector<float> pointiness_normal_values;
            std::vector<std::uint32_t> pointiness_edge_values;
            auto *pointiness_source =
                member(geometry, "pointiness_source");
            const auto has_pointiness_source =
                pointiness_source != nullptr &&
                !yyjson_is_null(pointiness_source);
            if (has_pointiness_source) {
                const auto edge_count =
                    static_cast<std::size_t>(unsigned_number(
                        member(pointiness_source, "edge_count")));
                pointiness_normal_values = read_values<float>(
                    geometry_stream,
                    section_offset(
                        pointiness_source, "point_normals"),
                    point_count * 3u);
                pointiness_edge_values =
                    read_values<std::uint32_t>(
                        geometry_stream,
                        section_offset(pointiness_source, "edges"),
                        edge_count * 2u);
            }

            TriangleMeshDesc mesh;
            mesh.name = text(member(geometry, "name"));
            mesh.positions.reserve(point_count);
            for (std::size_t i = 0u; i < point_count; ++i) {
                mesh.positions.emplace_back(Vec3f{
                    position_values[i * 3u],
                    position_values[i * 3u + 1u],
                    position_values[i * 3u + 2u]});
            }
            mesh.normals.domain = normal_domain;
            mesh.normals.values.reserve(normal_count);
            for (std::size_t i = 0u; i < normal_count; ++i) {
                mesh.normals.values.emplace_back(Vec3f{
                    normal_values[i * 3u],
                    normal_values[i * 3u + 1u],
                    normal_values[i * 3u + 2u]});
            }
            mesh.uv.domain = uv_domain;
            if (member(geometry, "default_uv_available") != nullptr) {
                mesh.default_uv_available = boolean(
                    member(geometry, "default_uv_available"),
                    false);
            } else if (auto *layers = member(geometry, "uv_layers");
                       layers != nullptr && yyjson_is_arr(layers)) {
                // Compatibility with bundles exported before the explicit
                // availability bit: Blender writes one entry per real UV
                // layer, while the standard UV byte range may be all-zero
                // placeholder storage.
                mesh.default_uv_available =
                    yyjson_arr_size(layers) != 0u;
            }
            mesh.uv.values.reserve(uv_count);
            for (std::size_t i = 0u; i < uv_count; ++i) {
                mesh.uv.values.emplace_back(Vec2f{
                    uv_values[i * 2u],
                    uv_values[i * 2u + 1u]});
            }
            mesh.uv_tangents.domain = uv_tangent_domain;
            mesh.uv_tangents.values.reserve(
                uv_tangent_count);
            for (std::size_t i = 0u;
                 i < uv_tangent_count;
                 ++i) {
                mesh.uv_tangents.values.emplace_back(
                    uv_tangent_values.size() ==
                            uv_tangent_count * 4u
                        ? Vec4f{
                              uv_tangent_values[i * 4u],
                              uv_tangent_values[i * 4u + 1u],
                              uv_tangent_values[i * 4u + 2u],
                              uv_tangent_values[i * 4u + 3u]}
                        : Vec4f{});
            }
            mesh.generated.domain = generated_domain;
            mesh.generated.values.reserve(generated_count);
            for (std::size_t i = 0u;
                 i < generated_count;
                 ++i) {
                mesh.generated.values.emplace_back(Vec3f{
                    generated_values[i * 3u],
                    generated_values[i * 3u + 1u],
                    generated_values[i * 3u + 2u]});
            }
            if (member(
                    geometry,
                    "generated_transform") != nullptr) {
                mesh.generated_transform =
                    matrix(member(
                        geometry,
                        "generated_transform"));
            }
            if (has_pointiness_source) {
                auto &source = mesh.pointiness_source.emplace();
                source.point_normals.reserve(point_count);
                for (std::size_t i = 0u; i < point_count; ++i) {
                    source.point_normals.emplace_back(Vec3f{
                        pointiness_normal_values[i * 3u],
                        pointiness_normal_values[i * 3u + 1u],
                        pointiness_normal_values[i * 3u + 2u]});
                }
                source.edges.reserve(
                    pointiness_edge_values.size() / 2u);
                for (std::size_t i = 0u;
                     i < pointiness_edge_values.size();
                     i += 2u) {
                    source.edges.emplace_back(
                        std::array<std::uint32_t, 2u>{
                            pointiness_edge_values[i],
                            pointiness_edge_values[i + 1u]});
                }
            }
            auto *uv_layers = member(geometry, "uv_layers");
            if (uv_layers != nullptr &&
                yyjson_is_arr(uv_layers)) {
                yyjson_arr_iter uv_layer_iterator =
                    yyjson_arr_iter_with(uv_layers);
                while (auto *uv_layer =
                           yyjson_arr_iter_next(
                               &uv_layer_iterator)) {
                    const auto name =
                        text(member(uv_layer, "name"));
                    const auto domain =
                        compact_geometry
                            ? mesh_attribute_domain(
                                  member(uv_layer, "domain"),
                                  MeshAttributeDomain::corner)
                            : MeshAttributeDomain::point;
                    const auto value_count =
                        domain_count(domain);
                    auto values = read_values<float>(
                        geometry_stream,
                        section_offset(uv_layer, "values"),
                        value_count * 2u);
                    auto tangents = read_values<float>(
                        geometry_stream,
                        section_offset(uv_layer, "tangents"),
                        value_count * 4u);
                    auto &uv_destination =
                        mesh.uv_layers[name];
                    auto &tangent_destination =
                        mesh.uv_tangent_layers[name];
                    uv_destination.domain = domain;
                    tangent_destination.domain = domain;
                    uv_destination.values.reserve(value_count);
                    tangent_destination.values.reserve(
                        value_count);
                    for (std::size_t i = 0u;
                         i < value_count;
                         ++i) {
                        uv_destination.values.emplace_back(Vec2f{
                            values[i * 2u],
                            values[i * 2u + 1u]});
                        tangent_destination.values.emplace_back(
                            Vec4f{
                                tangents[i * 4u],
                                tangents[i * 4u + 1u],
                                tangents[i * 4u + 2u],
                                tangents[i * 4u + 3u]});
                    }
                }
            }
            auto *color_attributes =
                member(geometry, "color_attributes");
            if (auto *default_color =
                    member(geometry, "default_color_attribute");
                default_color != nullptr && yyjson_is_str(default_color)) {
                mesh.default_color_attribute = text(default_color);
            }
            if (color_attributes != nullptr &&
                yyjson_is_arr(color_attributes)) {
                yyjson_arr_iter attribute_iterator =
                    yyjson_arr_iter_with(color_attributes);
                while (auto *attribute =
                           yyjson_arr_iter_next(
                               &attribute_iterator)) {
                    const auto name =
                        text(member(attribute, "name"));
                    const auto domain =
                        compact_geometry
                            ? mesh_attribute_domain(
                                  member(attribute, "domain"))
                            : MeshAttributeDomain::point;
                    const auto value_count =
                        domain_count(domain);
                    auto values = read_values<float>(
                        geometry_stream,
                        section_offset(attribute, "values"),
                        value_count * 4u);
                    auto &destination =
                        mesh.color_attributes[name];
                    destination.domain = domain;
                    destination.values.reserve(value_count);
                    for (std::size_t i = 0u;
                         i < value_count;
                         ++i) {
                        destination.values.emplace_back(
                            Vec4f{
                                values[i * 4u],
                                values[i * 4u + 1u],
                                values[i * 4u + 2u],
                                values[i * 4u + 3u]});
                    }
                    if (text(member(attribute, "data_type")) ==
                            "BYTE_COLOR" &&
                        domain == MeshAttributeDomain::corner) {
                        auto *byte_section =
                            member(attribute, "byte_values");
                        if (byte_section != nullptr &&
                            !yyjson_is_null(byte_section)) {
                            auto bytes = read_values<std::uint8_t>(
                                geometry_stream,
                                section_offset(
                                    attribute, "byte_values"),
                                value_count * 4u);
                            auto &raw =
                                mesh.cycles_byte_color_attributes[name];
                            raw.domain = domain;
                            raw.values.reserve(value_count);
                            for (std::size_t i = 0u;
                                 i < value_count;
                                 ++i) {
                                raw.values.emplace_back(
                                    std::array<std::uint8_t, 4u>{
                                        bytes[i * 4u],
                                        bytes[i * 4u + 1u],
                                        bytes[i * 4u + 2u],
                                        bytes[i * 4u + 3u]});
                            }
                        }
                    }
                }
            }
            mesh.triangles.reserve(triangle_count);
            for (std::size_t i = 0u;
                 i < triangle_count;
                 ++i) {
                mesh.triangles.emplace_back(
                    std::array<std::uint32_t, 3u>{
                        index_values[i * 3u],
                        index_values[i * 3u + 1u],
                        index_values[i * 3u + 2u]});
            }
            mesh.triangle_material_slots =
                std::move(material_values);
            mesh.triangle_smooth.reserve(triangle_count);
            for (const auto smooth : smooth_values) {
                mesh.triangle_smooth.emplace_back(
                    static_cast<std::uint8_t>(smooth != 0u));
            }
            mesh.triangle_random_per_island =
                std::move(random_per_island_values);
            mesh.cycles_primitive_offset =
                optional_unsigned_number(member(
                    member(geometry, "cycles_sync"),
                    "primitive_offset"));
            mesh.uses_adaptive_subdivision = boolean(
                member(geometry, "uses_adaptive_subdivision"),
                false);

            auto *slots = member(geometry, "material_slots");
            yyjson_arr_iter slot_iterator =
                yyjson_arr_iter_with(slots);
            while (auto *slot =
                       yyjson_arr_iter_next(&slot_iterator)) {
                if (yyjson_is_null(slot)) {
                    mesh.material_slots.emplace_back(
                        default_material);
                    continue;
                }
                const auto iter =
                    material_ids.find(text(slot));
                mesh.material_slots.emplace_back(
                    iter == material_ids.end()
                        ? default_material
                        : iter->second);
            }
            if (mesh.material_slots.empty()) {
                mesh.material_slots.emplace_back(
                    default_material);
            }
            scene.geometries.emplace(
                GeometryId{geometry_index++},
                std::move(mesh));
        }

        const auto mesh_geometry_count = geometry_index - 1u;
        auto *curve_geometries = member(root, "curve_geometries");
        if (curve_geometries != nullptr &&
            yyjson_is_arr(curve_geometries)) {
            yyjson_arr_iter curve_geometry_iterator =
                yyjson_arr_iter_with(curve_geometries);
            while (auto *geometry = yyjson_arr_iter_next(
                       &curve_geometry_iterator)) {
                const auto key_count = static_cast<std::size_t>(
                    unsigned_number(member(geometry, "key_count")));
                const auto curve_count = static_cast<std::size_t>(
                    unsigned_number(member(geometry, "curve_count")));
                auto key_values = read_values<float>(
                    geometry_stream,
                    section_offset(geometry, "keys"),
                    key_count * 4u);
                auto first_key_values = read_values<std::uint32_t>(
                    geometry_stream,
                    section_offset(geometry, "curve_first_key"),
                    curve_count);
                auto material_values = read_values<std::uint32_t>(
                    geometry_stream,
                    section_offset(
                        geometry, "curve_material_slots"),
                    curve_count);
                auto intercept_values = read_values<float>(
                    geometry_stream,
                    section_offset(geometry, "intercept"),
                    key_count);
                auto length_values = read_values<float>(
                    geometry_stream,
                    section_offset(geometry, "length"),
                    curve_count);
                auto random_values = read_values<float>(
                    geometry_stream,
                    section_offset(geometry, "random"),
                    curve_count);

                CurveGeometryDesc curves;
                curves.name = text(member(geometry, "name"));
                curves.shape = curve_shape(
                    member(geometry, "shape"));
                curves.subdivisions =
                    static_cast<std::uint32_t>(unsigned_number(
                        member(geometry, "subdivisions"), 2u));
                curves.keys.reserve(key_count);
                for (std::size_t i = 0u; i < key_count; ++i) {
                    curves.keys.emplace_back(Vec4f{
                        key_values[i * 4u],
                        key_values[i * 4u + 1u],
                        key_values[i * 4u + 2u],
                        key_values[i * 4u + 3u]});
                }
                curves.curve_first_key =
                    std::move(first_key_values);
                curves.curve_material_slots =
                    std::move(material_values);
                if (auto *default_uv =
                        member(geometry, "default_uv_layer");
                    default_uv != nullptr && yyjson_is_str(default_uv)) {
                    curves.default_uv_layer = text(default_uv);
                }
                if (auto *uv_layers = member(geometry, "uv_layers");
                    uv_layers != nullptr && yyjson_is_arr(uv_layers)) {
                    yyjson_arr_iter uv_iterator =
                        yyjson_arr_iter_with(uv_layers);
                    while (auto *layer =
                               yyjson_arr_iter_next(&uv_iterator)) {
                        const auto name = text(member(layer, "name"));
                        auto packed = read_values<float>(
                            geometry_stream,
                            section_offset(layer, "values"),
                            curve_count * 2u);
                        auto &values = curves.uv_layers[name];
                        values.reserve(curve_count);
                        for (std::size_t i = 0u; i < curve_count; ++i) {
                            values.emplace_back(Vec2f{
                                packed[i * 2u], packed[i * 2u + 1u]});
                        }
                    }
                }
                if (auto *color_attributes =
                        member(geometry, "color_attributes");
                    color_attributes != nullptr &&
                    yyjson_is_arr(color_attributes)) {
                    yyjson_arr_iter color_iterator =
                        yyjson_arr_iter_with(color_attributes);
                    while (auto *attribute =
                               yyjson_arr_iter_next(&color_iterator)) {
                        const auto name =
                            text(member(attribute, "name"));
                        auto packed = read_values<float>(
                            geometry_stream,
                            section_offset(attribute, "values"),
                            curve_count * 4u);
                        auto &values = curves.color_attributes[name];
                        values.reserve(curve_count);
                        for (std::size_t i = 0u; i < curve_count; ++i) {
                            values.emplace_back(Vec4f{
                                packed[i * 4u],
                                packed[i * 4u + 1u],
                                packed[i * 4u + 2u],
                                packed[i * 4u + 3u]});
                        }
                    }
                }
                curves.intercept = std::move(intercept_values);
                curves.length = std::move(length_values);
                curves.random = std::move(random_values);
                curves.cycles_curve_offset =
                    optional_unsigned_number(member(
                        member(geometry, "cycles_sync"),
                        "curve_offset"));
                curves.cycles_segment_offset =
                    optional_unsigned_number(member(
                        member(geometry, "cycles_sync"),
                        "segment_offset"));

                auto *slots = member(geometry, "material_slots");
                yyjson_arr_iter slot_iterator =
                    yyjson_arr_iter_with(slots);
                while (auto *slot =
                           yyjson_arr_iter_next(&slot_iterator)) {
                    if (yyjson_is_null(slot)) {
                        curves.material_slots.emplace_back(
                            default_material);
                        continue;
                    }
                    const auto iter =
                        material_ids.find(text(slot));
                    curves.material_slots.emplace_back(
                        iter == material_ids.end()
                            ? default_material
                            : iter->second);
                }
                if (curves.material_slots.empty()) {
                    curves.material_slots.emplace_back(
                        default_material);
                }

                std::size_t segment_count = 0u;
                for (std::size_t curve = 0u;
                     curve < curves.curve_first_key.size();
                     ++curve) {
                    const auto begin = static_cast<std::size_t>(
                        curves.curve_first_key[curve]);
                    const auto end =
                        curve + 1u <
                                curves.curve_first_key.size()
                            ? static_cast<std::size_t>(
                                  curves.curve_first_key[
                                      curve + 1u])
                            : curves.keys.size();
                    if (end > begin) {
                        segment_count += end - begin - 1u;
                    }
                }
                if (segment_count != static_cast<std::size_t>(
                                         unsigned_number(
                                             member(
                                                 geometry,
                                                 "segment_count")))) {
                    throw std::runtime_error(
                        "curve geometry has an invalid segment count");
                }
                scene.curve_geometries.emplace(
                    GeometryId{geometry_index++},
                    std::move(curves));
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
                        .name =
                            text(member(light, "name")) +
                            " / Light Shader",
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
                    .name = text(member(light, "name")),
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
                        std::filesystem::path{
                            text(member(environment, "path"))},
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

}// namespace psycles::adapter
