#include "cycles_shader_identity.h"
#include "path_tracer_bsdf_tables.h"
#include "path_tracer_curve_scene.h"
#include "path_tracer_cycles_svm_scene.h"
#include "path_tracer_displacement_scene.h"
#include "path_tracer_environment.h"
#include "path_tracer_generated_coordinates.h"
#include "path_tracer_image_decode.h"
#include "path_tracer_internal.h"
#include "path_tracer_light_sampling_scene.h"
#include "path_tracer_scene_geometry.h"
#include "path_tracer_scene_upload.h"
#include "path_tracer_subsurface_scene.h"
#include "path_tracer_ambient_occlusion_scene.h"
#include "path_tracer_analytic_light_scene.h"
#include "path_tracer_attribute_residency.h"
#include "path_tracer_tangent_space.h"
#include "path_tracer_volume_majorant_scene.h"
#include "virtual_texture_cache.h"
#include "virtual_texture_cache_mipmap.h"

#include <psycles/compiler/cycles_emission_sampling.h>
#include <psycles/compiler/cycles_nishita.h>
#include <psycles/contract/cycles_pointiness.h>
#include <psycles/core/parallel.h>
#include <psycles/luisa/cycles_nishita.h>

#include <cstdlib>
#include <cstring>
#include <string_view>
#include <unordered_map>

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
    if (!result.diagnostics.empty()) {
        return result;
    }

    auto data = std::make_shared<LuisaSceneData>();
    data->device =
        luisa::compute::Device{_device.impl_shared()};
    data->revision = snapshot.revision;
    data->camera = camera_iter->second;
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

    const auto material_reachability =
        build_scene_material_reachability(snapshot);
    {
        std::string diagnostic;
        const bool enable_texture_derivatives =
            (_options.direct_texture_mip_mode ==
             LuisaTextureMipMode::isotropic) ||
            (_options.virtual_texture_cache.enabled &&
             _options.virtual_texture_cache.mip_mode ==
                 LuisaTextureMipMode::isotropic);
        data->cycles_svm = build_cycles_svm_runtime(
            data, snapshot, diagnostic, enable_texture_derivatives);
        if (!data->cycles_svm) {
            diagnose(
                result.diagnostics,
                "Cycles 5.2 SVM scene compilation: " + diagnostic + ".");
            return result;
        }
    }
    std::set<contract::MaterialId> surface_bssrdf_materials;
    data->cycles_background_object_index =
        data->cycles_svm->object_identities.background_index.value_or(
            cycles_shader_identity::invalid_index);
    const auto &reachable_surface_materials =
        material_reachability.surface_materials;
    for (const auto material_id : reachable_surface_materials) {
        const auto &metadata = cycles_svm_material_metadata(*data, material_id);
        if (metadata.has_surface_bssrdf) {
            surface_bssrdf_materials.emplace(material_id);
        }
    }
    data->has_subsurface = !surface_bssrdf_materials.empty();
    const SubsurfaceSceneComponent subsurface_scene;
    const auto subsurface_scene_plan = subsurface_scene.plan(
        snapshot, surface_bssrdf_materials);
    std::uint32_t ao_triangle_instance_count = 0u;
    for (const auto &[instance_id, instance] : snapshot.instances) {
        static_cast<void>(instance_id);
        if (snapshot.geometries.contains(instance.geometry)) {
            ++ao_triangle_instance_count;
        }
    }
    const AmbientOcclusionSceneComponent ambient_occlusion_scene{
        data->cycles_svm->compilation, ao_triangle_instance_count};
    ambient_occlusion_scene.initialize(data);
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
    {
        std::set<contract::MaterialId> volume_materials;
        for (const auto &[material_id, shader_index] :
             data->cycles_svm->material_shader_indices) {
            if (shader_index <
                    data->cycles_svm->compilation.shader_metadata.size() &&
                data->cycles_svm->compilation.shader_metadata[shader_index]
                    .has_volume) {
                volume_materials.emplace(material_id);
            }
        }
        data->volume_metadata =
            VolumeSceneMetadataComponent{}.analyze(snapshot, volume_materials);
    }
    std::set<contract::InstanceId> volume_object_instances;
    for (const auto &vobj : data->volume_metadata.objects) {
        volume_object_instances.emplace(vobj.instance);
    }
    if (!volume_object_instances.empty()) {
        data->volume_boundary_accel.emplace(
            data->device.create_accel(static_scene_accel_option()));
        data->primary_to_volume_local_host.assign(
            snapshot.instances.size(), ~luisa::uint{0u});
    }
    if (data->volume_metadata.has_volumes()) {
        data->volume_shadow_guard_hits.emplace(
            data->device.create_buffer<luisa::uint>(1u));
    }
    data->volume_metadata
        .closure_allocation_budget =
        data->cycles_svm->compilation.max_closures;

    for (const auto &[id, shader_index] : data->cycles_svm->material_shader_indices) {
        static_cast<void>(shader_index);
        const auto identity = static_cast<std::uint32_t>(data->material_bindings.size());
        data->material_bindings.emplace(
            id, make_cycles_svm_material_binding(*data, id, 0u, 0u, identity));
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
    auto cycles_bsdf_values = make_cycles_bsdf_table_values(snapshot.shader_color_space);
    data->cycles_bsdf_table_buffer =
        data->device.create_buffer<float>(
            cycles_bsdf_values.size());

    const auto attribute_residency =
        build_scene_attribute_residency_plan(
            snapshot, data->cycles_svm->compilation,
            data->cycles_svm->material_shader_indices);
    for (const auto &[id, geometry] :
         snapshot.geometries) {
        const auto requires_pointiness =
            attribute_residency.geometry(id).contains(
                contract::cycles_pointiness_attribute_id);
        if (requires_pointiness && !geometry.pointiness_source) {
            diagnose(
                result.diagnostics,
                "Geometry '" + geometry.name +
                    "' uses Geometry.Pointiness but has no evaluated "
                    "point normals and original edges.");
        }
    }
    if (!result.diagnostics.empty()) {
        return result;
    }
    const auto attribute_count =
        attribute_residency.resident_binding_count;
    LUISA_INFO(
        "Psycles named-attribute residency: bindings={}/{} "
        "device_bytes={}/{}.",
        attribute_count,
        attribute_residency.source_binding_count,
        attribute_residency.resident_device_bytes,
        attribute_residency.source_device_bytes);
    const auto fixed_geometry_slots =
        (snapshot.geometries.size() +
         snapshot.curve_geometries.size()) *
        geometry_bindless_stride;
    const auto bindless_slots =
        std::max<std::size_t>(
            fixed_geometry_slots,
            1u);
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
    luisa::vector<MaterialBindingGpu> geometry_materials;
    auto next_attribute_slot =
        static_cast<std::uint32_t>(
            fixed_geometry_slots);
    Stream stream = data->device.create_stream();
    stream << data->cycles_bsdf_table_buffer.copy_from(
                  luisa::span{cycles_bsdf_values});
    ambient_occlusion_scene.upload_distance(
        stream, data, snapshot.ambient_occlusion_distance);
    upload_cycles_svm_runtime(stream, *data->cycles_svm);

    std::size_t texture_slot_count = 1u;
    for (const auto &[image_id, image] : snapshot.images) {
        static_cast<void>(image);
        texture_slot_count = std::max(
            texture_slot_count,
            static_cast<std::size_t>(image_id.value) + 1u);
    }
    for (const auto &image : data->cycles_svm->image_bindings) {
        texture_slot_count = std::max(
            texture_slot_count,
            static_cast<std::size_t>(image.texture_slot) + 1u);
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
        data->environment_suns = environment.suns;
        if (data->cycles_background_shader_id == ~0u) {
            // Native world nodes own their image/SkyLoader resources.
            // EnvironmentDesc contributes sampling metadata only; do not
            // upload or precompute a second world texture or solar spectrum.
            data->environment_texture_slot =
                static_cast<std::uint32_t>(texture_slot_count);
            ++texture_slot_count;
        }
    }
    data->texture_heap =
        data->device.create_bindless_array(texture_slot_count);
    data->images.reserve(
        snapshot.images.size() +
        data->cycles_svm->nishita_images.size() +
        (snapshot.environment ? 2u : 1u));
    std::vector<std::vector<std::uint8_t>> compact_byte_uploads;
    compact_byte_uploads.reserve(snapshot.images.size() + 1u);
    std::vector<std::vector<float>> compact_float_uploads;
    compact_float_uploads.reserve(snapshot.images.size());
    std::vector<luisa::vector<luisa::float4>>
        float_texture_uploads;
    float_texture_uploads.reserve(
        data->cycles_svm->nishita_images.size() +
        (snapshot.environment ? 1u : 0u));
    std::vector<luisa::uint4> direct_mip_desc_upload;

    auto &dummy_pixels = compact_byte_uploads.emplace_back(4u);
    dummy_pixels[0u] = 255u;
    dummy_pixels[1u] = 0u;
    dummy_pixels[2u] = 255u;
    dummy_pixels[3u] = 255u;
    auto &dummy_image = data->images.emplace_back(
        data->device.create_image<float>(
            luisa::compute::PixelStorage::BYTE4, 1u, 1u));
    data->texture_heap.emplace_on_update(
        0u,
        dummy_image,
        luisa::compute::Sampler::linear_point_repeat());
    stream << dummy_image.copy_from(
        luisa::span{dummy_pixels});

    const auto bind_scene_image = [&](contract::ImageId image_id,
                                       const Image<float> &resource) {
        // Retained only for the private displacement bridge. Native handles
        // have separate immutable sampler descriptors sharing this storage.
        data->texture_heap.emplace_on_update(
            static_cast<std::uint32_t>(image_id.value), resource,
            luisa::compute::Sampler::linear_point_repeat());
        const auto &handles = data->cycles_svm->compilation.images;
        for (auto i = std::size_t{}; i < handles.size(); ++i) {
            const auto &handle = handles[i];
            if (!handle.nishita && handle.resource_id == image_id.value) {
                data->texture_heap.emplace_on_update(
                    data->cycles_svm->image_bindings[i].texture_slot, resource,
                    cycles_svm_texture_sampler(handle.interpolation,
                                               handle.extension));
            }
        }
    };

    data->direct_texture_mip_mode = _options.direct_texture_mip_mode;
    const bool use_vt =
        _options.virtual_texture_cache.enabled && !snapshot.images.empty();
    const bool build_direct_mips =
        (_options.direct_texture_mip_mode == LuisaTextureMipMode::isotropic) ||
        (use_vt &&
         _options.virtual_texture_cache.mip_mode ==
             LuisaTextureMipMode::isotropic);
    std::set<contract::ImageId> vt_force_direct_ids;
    if (use_vt && data->cycles_svm) {
        const auto shader_count =
            static_cast<std::uint32_t>(data->cycles_svm->compilation.table.ir.shaders.size());
        std::vector<std::uint32_t> all_shaders(shader_count);
        for (std::uint32_t s = 0u; s < shader_count; ++s) {
            all_shaders[s] = s;
        }
        vt_force_direct_ids = collect_shader_routine_image_ids(
            data->cycles_svm->compilation,
            all_shaders,
            compiler::cycles_svm::SHADER_TYPE_VOLUME);
    }

    struct DecodedImageSlot {
        contract::ImageId id{};
        bool skip{false};
        bool has_clip{false};
        std::optional<std::size_t> duplicate_of;
        std::uint64_t encoded_hash{};
        std::string diagnostic;
        bool is_unorm8{false};
        std::uint8_t channels{4u};
        DecodedImageChannelMode channel_mode{DecodedImageChannelMode::rgba};
        std::uint32_t width{};
        std::uint32_t height{};
        std::uint32_t mip_count{1u};
        std::vector<std::uint8_t> byte_pixels;
        std::vector<float> float_pixels;
        std::vector<VtUnorm8MipLevel> unorm8_mips;
        std::vector<VtFloat4MipLevel> float_mips;
        bool is_virtual{false};
        VtProbedSceneImage vt_probed{};
    };
    std::set<contract::ImageId> clip_extension_image_ids;
    if (data->cycles_svm) {
        for (const auto &handle : data->cycles_svm->compilation.images) {
            if (!handle.nishita &&
                handle.extension ==
                    compiler::cycles_svm::ImageExtension::clip) {
                clip_extension_image_ids.emplace(
                    contract::ImageId{handle.resource_id});
            }
        }
    }
    std::vector<std::pair<contract::ImageId, const contract::ImageDesc *>>
        image_entries;
    image_entries.reserve(snapshot.images.size());
    for (const auto &[image_id, image] : snapshot.images) {
        image_entries.emplace_back(image_id, &image);
    }
    std::vector<DecodedImageSlot> decoded_images(image_entries.size());
    if (!use_vt) {
        psycles::parallel_for(image_entries.size(), [&](std::size_t i) {
            const auto [image_id, image_ptr] = image_entries[i];
            const auto &image = *image_ptr;
            auto &slot = decoded_images[i];
            slot.id = image_id;
            slot.has_clip = clip_extension_image_ids.contains(image_id);
            if (image.load_failed) {
                slot.skip = true;
                return;
            }
            if (image.encoded_data.empty() ||
                image.encoded_data.size() >
                    static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                slot.diagnostic =
                    "Image '" + image.name + "' has no decodable payload.";
                return;
            }
            std::uint64_t h = 14695981039346656037ull;
            for (const std::uint8_t b : image.encoded_data) {
                h ^= static_cast<std::uint64_t>(b);
                h *= 1099511628211ull;
            }
            slot.encoded_hash = h;
        });
        std::unordered_map<std::uint64_t, std::vector<std::size_t>> hash_buckets;
        hash_buckets.reserve(image_entries.size());
        for (std::size_t i = 0u; i < image_entries.size(); ++i) {
            auto &slot = decoded_images[i];
            if (slot.skip || !slot.diagnostic.empty()) {
                continue;
            }
            const auto &img = *image_entries[i].second;
            auto &bucket = hash_buckets[slot.encoded_hash];
            for (const std::size_t canonical : bucket) {
                const auto &c_img = *image_entries[canonical].second;
                if (decoded_images[canonical].has_clip == slot.has_clip &&
                    c_img.alpha_type == img.alpha_type &&
                    c_img.color_space == img.color_space &&
                    c_img.encoded_data.size() == img.encoded_data.size() &&
                    std::memcmp(c_img.encoded_data.data(),
                                img.encoded_data.data(),
                                img.encoded_data.size()) == 0) {
                    slot.duplicate_of = canonical;
                    break;
                }
            }
            if (!slot.duplicate_of) {
                bucket.push_back(i);
            }
        }
    }
    psycles::parallel_for(image_entries.size(), [&](std::size_t i) {
        const auto [image_id, image_ptr] = image_entries[i];
        const auto &image = *image_ptr;
        auto &slot = decoded_images[i];
        slot.id = image_id;
        if (slot.skip || !slot.diagnostic.empty() || slot.duplicate_of) {
            return;
        }
        if (image.load_failed) {
            slot.skip = true;
            if (use_vt) {
                slot.vt_probed.id = image_id;
                slot.vt_probed.name = image.name;
                slot.vt_probed.load_failed = true;
            }
            return;
        }
        if (image.encoded_data.empty() ||
            image.encoded_data.size() >
                static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            slot.diagnostic =
                "Image '" + image.name + "' has no decodable payload.";
            return;
        }
        const bool force_direct =
            !use_vt || vt_force_direct_ids.contains(image_id);
        if (use_vt) {
            auto probed = probe_scene_image_info(image.encoded_data, image.name);
            if (!probed) {
                slot.diagnostic =
                    "Failed to probe image '" + image.name + "'.";
                return;
            }
            const auto [lim_w, lim_h] = compute_texture_limit_extent(
                probed->width, probed->height, snapshot.texture_limit);
            probed->width = lim_w;
            probed->height = lim_h;
            slot.vt_probed = VtProbedSceneImage{
                .id = image_id,
                .name = image.name,
                .encoded_data = image.encoded_data,
                .alpha_type = image.alpha_type,
                .color_space = image.color_space,
                .load_failed = false,
                .force_direct = force_direct,
                .texture_limit = snapshot.texture_limit,
                .probed = *probed};
            if (!force_direct) {
                slot.is_virtual = true;
                return;
            }
        }
        CompactDecodedImage compacted;
        bool decoded_ok = false;
        const bool allow_opaque_gray = !use_vt && !slot.has_clip;
        if (!use_vt && !slot.has_clip) {
            if (const auto probed =
                    probe_scene_image_info(image.encoded_data, image.name);
                probed && probed->storage == io::DecodedImageStorage::unorm8 &&
                probed->channels == 1u && !probed->has_alpha &&
                compute_texture_limit_shift(
                    probed->width, probed->height, snapshot.texture_limit) == 0u) {
                if (auto gray = decode_scene_image_gray8(
                        image.encoded_data, image.name)) {
                    compacted = std::move(*gray);
                    decoded_ok = true;
                }
            }
        }
        if (!decoded_ok) {
            auto decoded = decode_scene_image(image.encoded_data, image.name);
            if (!decoded) {
                slot.diagnostic =
                    "Failed to decode image '" + image.name + "'.";
                return;
            }
            apply_scene_image_alpha(
                *decoded, image.alpha_type, image.color_space);
            downscale_scene_image_to_limit(*decoded, snapshot.texture_limit);
            if (!use_vt) {
                compacted = compact_decoded_scene_image(
                    std::move(*decoded), allow_opaque_gray);
            } else {
                compacted.width = decoded->width;
                compacted.height = decoded->height;
                compacted.is_unorm8 =
                    decoded->storage == io::DecodedImageStorage::unorm8;
                compacted.channels = 4u;
                compacted.channel_mode = DecodedImageChannelMode::rgba;
                if (compacted.is_unorm8) {
                    compacted.byte_pixels = std::move(decoded->unorm8_pixels);
                } else {
                    compacted.float_pixels = std::move(decoded->float_pixels);
                }
            }
        }
        slot.width = compacted.width;
        slot.height = compacted.height;
        slot.is_unorm8 = compacted.is_unorm8;
        slot.channels = compacted.channels;
        slot.channel_mode = compacted.channel_mode;
        if (slot.is_unorm8) {
            slot.byte_pixels = std::move(compacted.byte_pixels);
        } else {
            slot.float_pixels = std::move(compacted.float_pixels);
        }
        if (build_direct_mips) {
            slot.mip_count = vt_compute_mip_count(slot.width, slot.height);
            if (slot.is_unorm8) {
                slot.unorm8_mips = vt_build_unorm8_channel_tail_mips(
                    std::span<const std::uint8_t>{slot.byte_pixels},
                    slot.width,
                    slot.height,
                    slot.channels,
                    image.color_space == contract::ImageColorSpace::srgb);
            } else {
                slot.float_mips = vt_build_float_channel_tail_mips(
                    std::span<const float>{slot.float_pixels},
                    slot.width,
                    slot.height,
                    slot.channels);
            }
        }
    });
    std::vector<VtProbedSceneImage> vt_entries;
    if (use_vt) {
        vt_entries.reserve(decoded_images.size());
    }
    std::vector<std::size_t> slot_image_index(
        decoded_images.size(), std::numeric_limits<std::size_t>::max());
    std::size_t dedup_count = 0u;
    std::size_t byte1_count = 0u;
    std::size_t byte2_count = 0u;
    std::size_t float1_count = 0u;
    std::size_t float2_count = 0u;
    std::size_t saved_base_bytes = 0u;
    for (std::size_t slot_idx = 0u; slot_idx < decoded_images.size();
         ++slot_idx) {
        auto &slot = decoded_images[slot_idx];
        if (!slot.diagnostic.empty()) {
            diagnose(result.diagnostics, std::move(slot.diagnostic));
            continue;
        }
        if (slot.skip) {
            if (use_vt) {
                vt_entries.push_back(std::move(slot.vt_probed));
            }
            continue;
        }
        if (use_vt && slot.is_virtual) {
            bind_scene_image(slot.id, dummy_image);
            vt_entries.push_back(std::move(slot.vt_probed));
            continue;
        }
        if (use_vt) {
            vt_entries.push_back(std::move(slot.vt_probed));
        }
        if (slot.duplicate_of) {
            const auto canonical_idx = *slot.duplicate_of;
            const auto img_idx = slot_image_index[canonical_idx];
            if (img_idx < data->images.size()) {
                const auto &c_slot = decoded_images[canonical_idx];
                slot.width = c_slot.width;
                slot.height = c_slot.height;
                slot.mip_count = c_slot.mip_count;
                slot.is_unorm8 = c_slot.is_unorm8;
                slot.channels = c_slot.channels;
                slot.channel_mode = c_slot.channel_mode;
                slot_image_index[slot_idx] = img_idx;
                bind_scene_image(slot.id, data->images[img_idx]);
                ++dedup_count;
                saved_base_bytes +=
                    static_cast<std::size_t>(slot.width) *
                    static_cast<std::size_t>(slot.height) *
                    (slot.is_unorm8 ? 4u : 16u);
            }
            continue;
        }
        const auto px_count =
            static_cast<std::size_t>(slot.width) *
            static_cast<std::size_t>(slot.height);
        if (slot.is_unorm8) {
            auto &pixels =
                compact_byte_uploads.emplace_back(std::move(slot.byte_pixels));
            const std::uint32_t mip_levels =
                static_cast<std::uint32_t>(1u + slot.unorm8_mips.size());
            const auto storage =
                slot.channels == 1u
                    ? luisa::compute::PixelStorage::BYTE1
                    : (slot.channels == 2u
                           ? luisa::compute::PixelStorage::BYTE2
                           : luisa::compute::PixelStorage::BYTE4);
            if (slot.channels == 1u) {
                ++byte1_count;
                saved_base_bytes += px_count * 3u;
            } else if (slot.channels == 2u) {
                ++byte2_count;
                saved_base_bytes += px_count * 2u;
            }
            slot_image_index[slot_idx] = data->images.size();
            auto &resource = data->images.emplace_back(
                data->device.create_image<float>(
                    storage,
                    slot.width,
                    slot.height,
                    mip_levels));
            bind_scene_image(slot.id, resource);
            stream << resource.view(0u).copy_from(luisa::span{pixels});
            for (std::size_t m = 0u; m < slot.unorm8_mips.size(); ++m) {
                const auto &src_mip = slot.unorm8_mips[m];
                auto &mip_upload =
                    compact_byte_uploads.emplace_back(src_mip.rgba);
                stream << resource.view(static_cast<std::uint32_t>(m + 1u))
                              .copy_from(luisa::span{mip_upload});
            }
        } else {
            auto &pixels =
                compact_float_uploads.emplace_back(std::move(slot.float_pixels));
            const std::uint32_t mip_levels =
                static_cast<std::uint32_t>(1u + slot.float_mips.size());
            const auto storage =
                slot.channels == 1u
                    ? luisa::compute::PixelStorage::FLOAT1
                    : (slot.channels == 2u
                           ? luisa::compute::PixelStorage::FLOAT2
                           : luisa::compute::PixelStorage::FLOAT4);
            if (slot.channels == 1u) {
                ++float1_count;
                saved_base_bytes += px_count * 12u;
            } else if (slot.channels == 2u) {
                ++float2_count;
                saved_base_bytes += px_count * 8u;
            }
            slot_image_index[slot_idx] = data->images.size();
            auto &resource = data->images.emplace_back(
                data->device.create_image<float>(
                    storage,
                    slot.width,
                    slot.height,
                    mip_levels));
            bind_scene_image(slot.id, resource);
            stream << resource.view(0u).copy_from(luisa::span{pixels});
            for (std::size_t m = 0u; m < slot.float_mips.size(); ++m) {
                const auto &src_mip = slot.float_mips[m];
                auto &mip_upload =
                    compact_float_uploads.emplace_back(src_mip.rgba);
                stream << resource.view(static_cast<std::uint32_t>(m + 1u))
                              .copy_from(luisa::span{mip_upload});
            }
        }
    }
    if (dedup_count != 0u || byte1_count != 0u || byte2_count != 0u ||
        float1_count != 0u || float2_count != 0u) {
        LUISA_INFO(
            "Psycles scene images: byte1={}, byte2={}, float1={}, float2={}, "
            "dedup={}, saved_mip0_mb={:.1f}.",
            byte1_count, byte2_count, float1_count, float2_count,
            dedup_count, static_cast<double>(saved_base_bytes) / 1.0e6);
    }
    if (data->cycles_svm) {
        const auto &handles = data->cycles_svm->compilation.images;
        auto &bindings = data->cycles_svm->image_bindings;
        const std::size_t mip_desc_count =
            std::max<std::size_t>(handles.size(), 1u);
        direct_mip_desc_upload.assign(
            mip_desc_count, luisa::make_uint4(1u, 1u, 1u, 0u));
        std::vector<std::pair<std::uint64_t, std::size_t>> id_to_slot;
        id_to_slot.reserve(decoded_images.size());
        for (std::size_t si = 0u; si < decoded_images.size(); ++si) {
            id_to_slot.emplace_back(decoded_images[si].id.value, si);
        }
        std::sort(
            id_to_slot.begin(),
            id_to_slot.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
        bool updated_bindings = false;
        for (std::size_t i = 0u; i < handles.size(); ++i) {
            const auto &h = handles[i];
            if (h.nishita) {
                const std::uint32_t sampler_bits =
                    i < bindings.size() ? bindings[i].sampler : 0u;
                direct_mip_desc_upload[i] = luisa::make_uint4(
                    compiler::cycles_svm::nishita_texture_width,
                    compiler::cycles_svm::nishita_texture_height,
                    1u,
                    sampler_bits);
            } else {
                const auto it = std::lower_bound(
                    id_to_slot.begin(),
                    id_to_slot.end(),
                    h.resource_id,
                    [](const auto &entry, std::uint64_t key) {
                        return entry.first < key;
                    });
                if (it != id_to_slot.end() && it->first == h.resource_id) {
                    const auto &s = decoded_images[it->second];
                    if (i < bindings.size() &&
                        s.channel_mode != DecodedImageChannelMode::rgba) {
                        const auto mode_bits =
                            static_cast<std::uint32_t>(s.channel_mode)
                            << cycles_svm_image_channel_mode_shift;
                        bindings[i].sampler =
                            (bindings[i].sampler &
                             ~cycles_svm_image_channel_mode_mask) |
                            mode_bits;
                        updated_bindings = true;
                    }
                    const std::uint32_t sampler_bits =
                        i < bindings.size() ? bindings[i].sampler : 0u;
                    direct_mip_desc_upload[i] = luisa::make_uint4(
                        std::max(s.width, 1u),
                        std::max(s.height, 1u),
                        std::max(s.mip_count, 1u),
                        sampler_bits);
                }
            }
        }
        if (updated_bindings && data->cycles_svm->image_binding_buffer) {
            stream << data->cycles_svm->image_binding_buffer->copy_from(
                luisa::span{bindings});
        }
        data->direct_image_mip_descs =
            data->device.create_buffer<luisa::uint4>(mip_desc_count);
        stream << data->direct_image_mip_descs.copy_from(
            luisa::span{direct_mip_desc_upload});
    }
    if (use_vt && result.diagnostics.empty()) {
        data->virtual_texture_cache = VirtualTextureCache::create(
            data->device,
            stream,
            _options.virtual_texture_cache,
            vt_entries,
            data->cycles_svm->compilation.images,
            data->cycles_svm->image_bindings,
            &data->cycles_svm->compilation);
    }

    const auto &nishita_bindings = data->cycles_svm->nishita_images;
    std::vector<luisa::vector<luisa::float4>> nishita_uploads(
        nishita_bindings.size());
    psycles::parallel_for(nishita_bindings.size(), [&](std::size_t i) {
        const auto &binding = nishita_bindings[i];
        const auto source = compiler::cycles_svm::precompute_nishita_texture(
            binding.parameters);
        const auto pixel_count =
            compiler::cycles_svm::nishita_texture_width *
            compiler::cycles_svm::nishita_texture_height;
        auto &pixels = nishita_uploads[i];
        pixels.reserve(pixel_count);
        for (std::size_t pixel = 0u; pixel < pixel_count; ++pixel) {
            const auto offset =
                pixel * compiler::cycles_svm::nishita_texture_channels;
            pixels.emplace_back(luisa::make_float4(
                source[offset],
                source[offset + 1u],
                source[offset + 2u],
                source[offset + 3u]));
        }
    });
    for (std::size_t i = 0u; i < nishita_bindings.size(); ++i) {
        const auto &binding = nishita_bindings[i];
        auto &pixels = float_texture_uploads.emplace_back(
            std::move(nishita_uploads[i]));
        auto &resource = data->images.emplace_back(
            data->device.create_image<float>(
                luisa::compute::PixelStorage::FLOAT4,
                compiler::cycles_svm::nishita_texture_width,
                compiler::cycles_svm::nishita_texture_height));
        data->texture_heap.emplace_on_update(
            binding.texture_slot,
            resource,
            luisa::compute::Sampler::linear_point_edge());
        stream << resource.copy_from(luisa::span{pixels});
    }
    const auto has_nishita_environment =
        snapshot.environment &&
        snapshot.environment->nishita.has_value() &&
        data->environment_texture_slot.has_value();
    if (has_nishita_environment) {
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
        precompute(
            *data->environment_texture_slot,
            *snapshot.environment->nishita,
            data->nishita_environment
                ->pixel_bottom_xyz,
            data->nishita_environment
                ->pixel_top_xyz);
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

    // Every texture inserted into this scene's heap is created in the two
    // explicit branches above: BYTE4 for decoded UNORM8 and FLOAT4 for wider
    // or generated data. No R10G10B10A2 resource can enter this closed upload
    // transaction, so the path-kernel HIP codegen may remove its packed
    // sampler fallback. Keep unknown/imported upload routes conservative by
    // leaving LuisaSceneData::may_have_packed_textures at its true default.
    data->may_have_packed_textures = false;
    const bool apply_static_mesh_bvh = []() noexcept {
        if (const char *env = std::getenv("PSYCLES_APPLY_STATIC_MESH_BVH");
            env != nullptr && env[0] != '\0') {
            return env[0] != '0';
        }
        return true;
    }();

    struct MeshPrepEntry {
        contract::GeometryId id{};
        const contract::TriangleMeshDesc *geometry{};
        std::uint32_t primitive_offset{};
    };
    std::vector<MeshPrepEntry> mesh_entries;
    mesh_entries.reserve(snapshot.geometries.size());
    for (const auto &[geometry_id, geometry] : snapshot.geometries) {
        if (geometry.positions.empty() || geometry.triangles.empty()) {
            diagnose(
                result.diagnostics,
                "Geometry '" + geometry.name + "' has no triangles.");
            continue;
        }
        const auto cycles_primitive_interval =
            cycles_primitive_intervals.resolve(
                geometry.triangles.size(), geometry.cycles_primitive_offset);
        if (!cycles_primitive_interval.offset) {
            diagnose(
                result.diagnostics,
                "Geometry '" + geometry.name +
                    "' has an overlapping or out-of-range Cycles "
                    "primitive interval.");
            continue;
        }
        cycles_primitive_offsets.emplace(
            geometry_id, *cycles_primitive_interval.offset);
        mesh_entries.push_back(MeshPrepEntry{
            .id = geometry_id,
            .geometry = &geometry,
            .primitive_offset = *cycles_primitive_interval.offset});
    }
    if (!result.diagnostics.empty()) {
        return result;
    }

    uploads.resize(mesh_entries.size());
    std::vector<std::vector<std::string>> mesh_diags(mesh_entries.size());
    psycles::parallel_for(mesh_entries.size(), [&](std::size_t i) {
        const auto &entry = mesh_entries[i];
        const auto geometry_id = entry.id;
        const auto &geometry = *entry.geometry;
        auto &diags = mesh_diags[i];
        const auto &geometry_attribute_residency =
            attribute_residency.geometry(geometry_id);

        std::vector<float> pointiness_values;
        if (geometry.pointiness_source &&
            geometry_attribute_residency.contains(
                contract::cycles_pointiness_attribute_id)) {
            try {
                pointiness_values =
                    contract::make_cycles_pointiness_attribute(
                        geometry.positions,
                        geometry.pointiness_source->point_normals,
                        geometry.pointiness_source->edges);
            } catch (const std::invalid_argument &error) {
                diags.emplace_back(
                    "Geometry '" + geometry.name +
                    "' has an invalid Cycles Pointiness source: " +
                    error.what() + ".");
                return;
            }
        }

        auto &upload = uploads[i];
        upload.positions.reserve(geometry.positions.size());
        const auto generated_mapping =
            make_generated_coordinate_mapping(geometry);
        upload.generated_transform =
            to_luisa(generated_mapping.object_to_generated);
        for (const auto position : geometry.positions) {
            upload.positions.emplace_back(to_luisa(position));
        }
        if (const auto transform_iter =
                cycles_static_transform_by_geometry.find(geometry_id);
            transform_iter != cycles_static_transform_by_geometry.end()) {
            upload.cycles_intersection_positions.reserve(
                geometry.positions.size());
            for (const auto position : geometry.positions) {
                upload.cycles_intersection_positions.emplace_back(
                    to_luisa(cycles_transform_point(
                        transform_iter->second, position)));
            }
        }
        const auto require_vertex_or_corner =
            [&](std::string_view name, contract::MeshAttributeDomain domain) {
                if (domain != contract::MeshAttributeDomain::point &&
                    domain != contract::MeshAttributeDomain::corner) {
                    diags.emplace_back(
                        "Geometry '" + geometry.name +
                        "' uses unsupported " + std::string{name} +
                        " domain. Cycles primary triangle attributes must "
                        "be point or corner.");
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
            geometry.default_uv_available.value_or(!geometry.uv.values.empty());
        const auto uv_tangent_domain =
            geometry.uv_tangents.values.empty()
                ? contract::MeshAttributeDomain::point
                : geometry.uv_tangents.domain;
        const auto generated_domain =
            geometry.generated.values.empty()
                ? contract::MeshAttributeDomain::point
                : geometry.generated.domain;
        if (!require_vertex_or_corner("normal", normal_domain) ||
            !require_vertex_or_corner("UV", uv_domain) ||
            !require_vertex_or_corner("UV tangent", uv_tangent_domain) ||
            !require_vertex_or_corner("Generated", generated_domain)) {
            return;
        }
        upload.attribute_domains =
            (normal_domain == contract::MeshAttributeDomain::corner
                 ? geometry_normal_corner
                 : 0u) |
            (uv_domain == contract::MeshAttributeDomain::corner
                 ? geometry_uv_corner
                 : 0u) |
            (uv_tangent_domain == contract::MeshAttributeDomain::corner
                 ? geometry_uv_tangent_corner
                 : 0u) |
            (generated_domain == contract::MeshAttributeDomain::corner
                 ? geometry_generated_corner
                 : 0u);
        if (geometry.normals.values.empty()) {
            upload.normals.assign(
                geometry.positions.size(), luisa::make_float3(0.0f));
        } else {
            upload.normals.reserve(geometry.normals.values.size());
            for (const auto value : geometry.normals.values) {
                upload.normals.emplace_back(to_luisa(value));
            }
        }
        if (geometry.uv.values.empty()) {
            upload.uv.assign(
                geometry.positions.size(), luisa::make_float2(0.0f));
        } else {
            upload.uv.reserve(geometry.uv.values.size());
            for (const auto value : geometry.uv.values) {
                upload.uv.emplace_back(to_luisa(value));
            }
        }
        if (geometry.uv_tangents.values.empty()) {
            upload.uv_tangents.assign(
                geometry.positions.size(), luisa::make_float4(0.0f));
        } else {
            upload.uv_tangents.reserve(geometry.uv_tangents.values.size());
            for (const auto value : geometry.uv_tangents.values) {
                upload.uv_tangents.emplace_back(luisa::make_float4(
                    value.x, value.y, value.z, value.w));
            }
        }
        if (geometry.generated.values.empty()) {
            upload.generated.reserve(geometry.positions.size());
            for (const auto position : geometry.positions) {
                upload.generated.emplace_back(
                    to_luisa(generated_mapping.apply(position)));
            }
        } else {
            upload.generated.reserve(geometry.generated.values.size());
            for (const auto value : geometry.generated.values) {
                upload.generated.emplace_back(to_luisa(value));
            }
        }
        upload.attributes.reserve(
            geometry.color_attributes.size() +
            geometry.uv_layers.size() +
            geometry.uv_tangent_layers.size() * 2u +
            (pointiness_values.empty() ? 0u : 1u));
        if (!pointiness_values.empty()) {
            auto &attribute = upload.attributes.emplace_back();
            attribute.id = contract::cycles_pointiness_attribute_id;
            attribute.domain = attribute_domain_point;
            attribute.values.reserve(pointiness_values.size());
            for (const auto value : pointiness_values) {
                attribute.values.emplace_back(
                    luisa::make_float4(value, 0.0f, 0.0f, 0.0f));
            }
        }
        for (const auto &[name, source] : geometry.color_attributes) {
            const auto id = contract::attribute_id(name);
            if (!geometry_attribute_residency.contains(id)) {
                continue;
            }
            auto &attribute = upload.attributes.emplace_back();
            attribute.id = id;
            attribute.domain = encode_attribute_domain(source.domain);
            attribute.values.reserve(source.values.size());
            for (const auto value : source.values) {
                attribute.values.emplace_back(luisa::make_float4(
                    value.x, value.y, value.z, value.w));
            }
        }
        std::map<std::string, std::size_t, std::less<>> uv_attribute_indices;
        for (const auto &[name, source] : geometry.uv_layers) {
            const auto id = contract::uv_attribute_id(name);
            if (!geometry_attribute_residency.contains(id)) {
                continue;
            }
            uv_attribute_indices.emplace(name, upload.attributes.size());
            auto &attribute = upload.attributes.emplace_back();
            attribute.id = id;
            attribute.domain = encode_attribute_domain(source.domain);
            attribute.values.reserve(source.values.size());
            for (const auto value : source.values) {
                attribute.values.emplace_back(
                    luisa::make_float4(value.x, value.y, 0.0f, 0.0f));
            }
        }
        for (const auto &[name, source] : geometry.uv_tangent_layers) {
            const auto tangent_id = contract::uv_tangent_attribute_id(name);
            const auto undisplaced_tangent_id =
                contract::uv_undisplaced_tangent_attribute_id(name);
            const auto retain_tangent =
                geometry_attribute_residency.contains(tangent_id);
            const auto retain_undisplaced =
                geometry_attribute_residency.contains(undisplaced_tangent_id);
            if (!retain_tangent && !retain_undisplaced) {
                continue;
            }
            const auto uv = uv_attribute_indices.find(name);
            if (uv == uv_attribute_indices.end()) {
                diags.emplace_back(
                    "Geometry '" + geometry.name +
                    "' has a named tangent layer without its UV layer.");
                break;
            }
            std::optional<std::size_t> tangent_index;
            if (retain_tangent) {
                tangent_index = upload.attributes.size();
                auto &attribute = upload.attributes.emplace_back();
                attribute.id = tangent_id;
                attribute.domain = encode_attribute_domain(source.domain);
            }
            std::optional<std::size_t> undisplaced_index;
            if (retain_undisplaced) {
                undisplaced_index = upload.attributes.size();
                auto &undisplaced = upload.attributes.emplace_back();
                undisplaced.id = undisplaced_tangent_id;
                undisplaced.domain = encode_attribute_domain(source.domain);
            }
            upload.uv_tangent_layers.emplace_back(UvTangentLayerUpload{
                .uv_attribute_index = uv->second,
                .tangent_attribute_index = tangent_index,
                .undisplaced_tangent_attribute_index = undisplaced_index});
        }
        upload.triangles.reserve(geometry.triangles.size());
        upload.triangle_material_slots.reserve(geometry.triangles.size());
        upload.triangle_random_per_island.reserve(geometry.triangles.size());
        upload.triangle_smooth.reserve(geometry.triangles.size());
        for (std::size_t tri_idx = 0u;
             tri_idx < geometry.triangles.size();
             ++tri_idx) {
            const auto triangle = geometry.triangles[tri_idx];
            if (triangle[0u] >= geometry.positions.size() ||
                triangle[1u] >= geometry.positions.size() ||
                triangle[2u] >= geometry.positions.size()) {
                diags.emplace_back(
                    "Geometry '" + geometry.name +
                    "' contains an out-of-range index.");
                break;
            }
            upload.triangles.emplace_back(
                Triangle{triangle[0u], triangle[1u], triangle[2u]});
            upload.triangle_material_slots.emplace_back(
                tri_idx < geometry.triangle_material_slots.size()
                    ? geometry.triangle_material_slots[tri_idx]
                    : 0u);
            upload.triangle_random_per_island.emplace_back(
                tri_idx < geometry.triangle_random_per_island.size()
                    ? geometry.triangle_random_per_island[tri_idx]
                    : 0.0f);
            upload.triangle_smooth.emplace_back(
                tri_idx < geometry.triangle_smooth.size() &&
                        geometry.triangle_smooth[tri_idx] != 0u
                    ? 1u
                    : 0u);
        }
        if (!diags.empty()) {
            return;
        }
        recompute_cycles_tangent_space(upload);
        initialize_cycles_undisplaced_tangent_space(upload);
    });
    for (auto &diags : mesh_diags) {
        for (auto &msg : diags) {
            diagnose(result.diagnostics, std::move(msg));
        }
    }
    if (!result.diagnostics.empty()) {
        return result;
    }

    for (std::size_t i = 0u; i < mesh_entries.size(); ++i) {
        const auto &entry = mesh_entries[i];
        const auto geometry_id = entry.id;
        const auto &geometry = *entry.geometry;
        auto &upload = uploads[i];
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
            geometry_materials.emplace_back(
                material_iter != data->material_bindings.end()
                    ? to_luisa(material_iter->second)
                    : inert_material_binding());
        }
        if (geometry.material_slots.empty()) {
            geometry_materials.emplace_back(inert_material_binding());
        }

        auto &resource =
            data->geometries.emplace_back();
        resource.positions =
            data->device.create_buffer<luisa::float3>(
                upload.positions.size());
        resource.triangles =
            data->device.create_buffer<Triangle>(
                upload.triangles.size());
        resource.triangle_material_slots =
            data->device.create_buffer<luisa::uint>(
                upload.triangle_material_slots.size());
        resource.triangle_smooth =
            data->device.create_buffer<luisa::uint>(
                upload.triangle_smooth.size());
        if (!upload.cycles_intersection_positions.empty()) {
            resource.cycles_intersection_positions.emplace(
                data->device.create_buffer<luisa::float3>(
                    upload.cycles_intersection_positions.size()));
        }
        resource.mesh = data->device.create_mesh(
            (apply_static_mesh_bvh && resource.cycles_intersection_positions)
                ? *resource.cycles_intersection_positions
                : resource.positions,
            resource.triangles,
            static_scene_accel_option());
        data->heap.emplace_on_update(
            bindless_base, resource.triangles);
        data->heap.emplace_on_update(
            bindless_base + 1u, resource.positions);
        data->heap.emplace_on_update(
            bindless_base + 4u,
            resource.triangle_material_slots);
        data->heap.emplace_on_update(
            bindless_base + 8u,
            resource.triangle_smooth);
        if (resource.cycles_intersection_positions) {
            data->heap.emplace_on_update(
                bindless_base + 9u,
                *resource.cycles_intersection_positions);
        } else {
            data->heap.emplace_on_update(
                bindless_base + 9u,
                resource.positions);
        }
        stream << resource.positions.copy_from(
                      luisa::span{upload.positions})
               << resource.triangles.copy_from(
                      luisa::span{upload.triangles})
               << resource.triangle_material_slots.copy_from(
                      luisa::span{
                          upload.triangle_material_slots})
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
            attribute_residency,
            next_attribute_slot);
    if (!curve_upload.ok()) {
        diagnose(result.diagnostics, curve_upload.diagnostic);
        return result;
    }

    const auto displacement = MeshDisplacementSceneComponent{}.build(
        data,
        stream,
        snapshot,
        cycles_instance_intersection_plan,
        geometry_indices,
        cycles_primitive_offsets,
        curve_upload.cycles_curve_offsets,
        uploads);
    if (!displacement.ok()) {
        diagnose(result.diagnostics, displacement.diagnostic);
        return result;
    }
    if (!data->geometries.empty() || !data->curve_geometries.empty()) {
        // Sync BLAS builds before TLAS emplace so backends compact in place and
        // release build scratch buffers prior to TLAS allocation.
        stream << synchronize();
    }
    std::string cycles_svm_geometry_diagnostic;
    if (!finalize_cycles_svm_scene_runtime(
            data,
            snapshot,
            cycles_instance_intersection_plan,
            uploads,
            geometry_indices,
            cycles_primitive_offsets,
            curve_upload.cycles_curve_offsets,
            cycles_svm_geometry_diagnostic)) {
        diagnose(
            result.diagnostics,
            std::move(cycles_svm_geometry_diagnostic));
        return result;
    }
    upload_cycles_svm_scene_runtime(
        stream, *data->cycles_svm);
    for (auto &upload : uploads) {
        auto release = [](auto &vec) {
            std::remove_reference_t<decltype(vec)>{}.swap(vec);
        };
        release(upload.normals);
        release(upload.uv);
        release(upload.uv_tangents);
        release(upload.undisplaced_positions);
        release(upload.undisplaced_normals);
        release(upload.undisplaced_uv_tangents);
        release(upload.generated);
        release(upload.triangle_random_per_island);
        release(upload.triangle_smooth);
        release(upload.attributes);
        release(upload.uv_tangent_layers);
    }
    const auto material_may_emit =
        collect_emission_sampling_materials(*data);

    luisa::vector<InstanceGpu> instances;
    luisa::vector<float> ambient_occlusion_object_distances;
    bool has_ambient_occlusion_object_distance = false;
    luisa::vector<MaterialBindingGpu> override_materials;
    luisa::vector<EmissiveTriangleGpu> emissive_triangles;
    std::vector<float> emissive_triangle_areas;
    data->accel = data->device.create_accel(static_scene_accel_option());
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
            override_materials.emplace_back(
                material_iter != data->material_bindings.end()
                    ? to_luisa(material_iter->second)
                    : inert_material_binding());
        }
        const auto instance_index =
            static_cast<std::uint32_t>(instances.size());
        const auto cycles_object_index =
            data->cycles_svm->object_identities.instance_indices.at(instance_id);
        if (ambient_occlusion_object_distances.size() <=
            cycles_object_index) {
            ambient_occlusion_object_distances.resize(
                static_cast<std::size_t>(cycles_object_index) + 1u,
                0.0f);
        }
        ambient_occlusion_object_distances[cycles_object_index] =
            std::max(instance.ambient_occlusion_distance, 0.0f);
        has_ambient_occlusion_object_distance |=
            instance.ambient_occlusion_distance > 0.0f;
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
                cycles_object_index,
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
                    !material_may_emit.contains(*material_id)) {
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
                            cycles_object_index,
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
        const auto accel_transform =
            (apply_static_mesh_bvh && intersection_plan.transform_applied)
                ? Mat4f{}
                : instance.transform;
        data->accel.emplace_back(
            data->geometries[geometry_iter->second].mesh,
            to_luisa(accel_transform),
            visibility,
            false,
            instance_index);
        subsurface_scene.append_triangle_instance(
            data, subsurface_scene_plan, instance_index,
            geometry_iter->second, instance.transform);
        ambient_occlusion_scene.append_triangle_instance(
            data, instance_index,
            geometry_iter->second, accel_transform);
        if (data->volume_boundary_accel &&
            volume_object_instances.contains(instance_id)) {
            const auto local_vol_inst = static_cast<luisa::uint>(
                data->volume_boundary_accel->size());
            if (instance_index >= data->primary_to_volume_local_host.size()) {
                data->primary_to_volume_local_host.resize(
                    static_cast<std::size_t>(instance_index) + 1u, ~luisa::uint{0u});
            }
            data->primary_to_volume_local_host[instance_index] = local_vol_inst;
            data->volume_boundary_accel->emplace_back(
                data->geometries[geometry_iter->second].mesh,
                to_luisa(accel_transform), visibility, false, instance_index);
        }
    }
    if (!result.diagnostics.empty()) {
        return result;
    }
    data->ambient_occlusion_object_distance_count =
        has_ambient_occlusion_object_distance
            ? static_cast<std::uint32_t>(
                  ambient_occlusion_object_distances.size())
            : 0u;

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
            dummy.positions, dummy.triangles, static_scene_accel_option());
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

    auto analytic_lights =
        AnalyticLightSceneComponent{}.build(snapshot, *data);
    if (!analytic_lights.ok()) {
        diagnose(result.diagnostics, analytic_lights.diagnostic);
        return result;
    }
    data->background = to_luisa(analytic_lights.background);
    data->light_count = analytic_lights.regular_count;
    {
        std::uint32_t light_type_mask = 0u;
        for (std::uint32_t i = 0u; i < analytic_lights.regular_count; ++i) {
            light_type_mask |=
                (1u << analytic_lights.device_lights[i].type);
        }
        data->light_type_mask = light_type_mask;
    }
    data->portal_count = analytic_lights.portal_count;
    data->emissive_triangle_count =
        static_cast<std::uint32_t>(
            emissive_triangles.size());
    if (data->cycles_background_shader_id != ~0u) {
        const auto &shader = data->cycles_svm->compilation.kernel_shaders.at(
            data->cycles_background_shader_id & cycles_svm::shader_mask);
        data->environment_emission_is_constant =
            (shader.flags & compiler::cycles_svm::SD_HAS_CONSTANT_EMISSION) != 0;
    } else {
        data->environment_emission_is_constant =
            !data->environment_texture_slot && data->environment_suns.empty() &&
            !data->nishita_environment;
    }

    bool world_is_spatially_varying = false;
    if (snapshot.world_shader) {
        world_is_spatially_varying =
            cycles_svm_material_metadata(*data, *snapshot.world_shader)
                .has_surface_spatial_varying;
    }
    // Cycles LightManager::test_enabled_lights enables the background when a
    // portal exists, independently of the world's MIS policy. Without a
    // portal it requires both MIS and a spatially varying raw surface graph.
    const auto include_environment =
        data->cycles_background_shader_id != ~0u &&
        (data->portal_count != 0u ||
         (snapshot.world_sampling != contract::WorldSampling::none &&
          world_is_spatially_varying));
    auto light_sampling = build_light_sampling_scene_upload(
        snapshot,
        *data,
        uploads,
        analytic_lights.regular_lights(),
        analytic_lights.regular_light_ids,
        analytic_lights.regular_shader_emission_estimates,
        emissive_triangles,
        emissive_triangle_areas,
        include_environment,
        _options.build_light_tree);
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
    data->light_tree_mesh_triangle_count = static_cast<std::uint32_t>(
        light_sampling.tree_mesh_triangles.size());
    LUISA_INFO(
        "Psycles light-sampling scene: emissive_triangles={}, "
        "analytic_lights={}, portals={}, environment={}, flat_emitters={}, "
        "light_tree_nodes={}, light_tree_emitters={}.",
        data->emissive_triangle_count,
        data->light_count,
        data->portal_count,
        data->environment_in_light_distribution,
        data->light_distribution_count,
        data->light_tree_node_count,
        data->light_tree_emitter_count);
    const auto table_upload = SceneTableUploadComponent{}.upload(
        data,
        stream,
        {.geometries = geometry_gpu,
         .instances = instances,
         .ambient_occlusion_object_distances =
             ambient_occlusion_object_distances,
         .geometry_materials = geometry_materials,
         .override_materials = override_materials,
         .lights = analytic_lights.device_lights,
         .emissive_triangles = emissive_triangles,
         .light_distribution = light_sampling.distribution,
         .light_tree_nodes = light_sampling.tree_nodes,
         .light_tree_emitters = light_sampling.tree_emitters,
         .light_tree_emitter_mappings =
             light_sampling.tree_emitter_mappings,
         .light_tree_triangle_emitter_mappings =
             light_sampling.tree_triangle_emitter_mappings,
         .light_tree_mesh_triangles =
             light_sampling.tree_mesh_triangles,
         .light_tree_triangle_lookup =
             light_sampling.tree_triangle_lookup,
         .geometry_uploads = std::span<const GeometryUpload>{
             uploads.data(), uploads.size()}});
    if (!table_upload.ok()) {
        diagnose(result.diagnostics, table_upload.diagnostic);
        return result;
    }

    const auto majorants =
        VolumeMajorantSceneComponent{}.plan(data, snapshot);
    if (!majorants.ok()) {
        diagnose(result.diagnostics, majorants.diagnostic);
        return result;
    }
    data->volume_majorant_plan =
        std::make_shared<VolumeMajorantScenePlan>(majorants);

    configure_background_sampling(
        *data,
        snapshot,
        include_environment);

    result.scene =
        std::make_unique<LuisaCompiledScene>(
            std::move(data));
    return result;
}
}// namespace psycles::luisa_backend
