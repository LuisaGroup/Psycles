#include "path_tracer_scene_upload.h"

#include "cycles_shader_identity.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <tuple>

namespace psycles::luisa_backend::detail {
namespace {

void provide_inert_storage(SceneTableUploadInput input) {
    if (input.light_distribution.empty()) {
        input.light_distribution.emplace_back(LightDistributionGpu{});
    }
    if (input.light_tree_nodes.empty()) {
        input.light_tree_nodes.emplace_back(LightTreeNodeGpu{});
    }
    if (input.light_tree_emitters.empty()) {
        input.light_tree_emitters.emplace_back(LightTreeEmitterGpu{});
    }
    if (input.light_tree_emitter_mappings.empty()) {
        input.light_tree_emitter_mappings.emplace_back(
            ~luisa::uint{0u}, ~luisa::uint{0u});
    }
    if (input.light_tree_triangle_emitter_mappings.empty()) {
        input.light_tree_triangle_emitter_mappings.emplace_back(
            ~luisa::uint{0u}, ~luisa::uint{0u});
    }
    if (input.light_tree_mesh_triangles.empty()) {
        input.light_tree_mesh_triangles.emplace_back(~luisa::uint{0u});
    }
    if (input.light_tree_triangle_lookup.empty()) {
        input.light_tree_triangle_lookup.emplace_back(
            ~luisa::uint{0u},
            ~luisa::uint{0u},
            ~luisa::uint{0u},
            0u);
    }
    if (input.lights.empty()) {
        input.lights.emplace_back(LightGpu{});
    }
    if (input.emissive_triangles.empty()) {
        input.emissive_triangles.emplace_back(EmissiveTriangleGpu{});
    }
    if (input.geometry_materials.empty()) {
        input.geometry_materials.emplace_back(inert_material_binding());
    }
    if (input.override_materials.empty()) {
        input.override_materials.emplace_back(inert_material_binding());
    }
    // Empty-world renders are valid Cycles scenes. Luisa buffers cannot be
    // zero-sized; these records stay unreachable without a committed hit.
    if (input.geometries.empty()) {
        input.geometries.emplace_back(GeometryGpu{});
    }
    if (input.instances.empty()) {
        input.instances.emplace_back(InstanceGpu{});
    }
    if (input.ambient_occlusion_object_distances.empty()) {
        input.ambient_occlusion_object_distances.emplace_back(0.0f);
    }
}

}// namespace

SceneTraversalTableBuildResult
build_scene_traversal_tables(SceneTraversalTableBuildInput input) {
    SceneTraversalTableBuildResult result;
    constexpr auto uint_maximum =
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
    if (input.geometry_materials.size() > uint_maximum ||
        input.override_materials.size() > uint_maximum ||
        input.instances.size() > uint_maximum) {
        result.diagnostic =
            "Scene traversal table exceeds the 32-bit device address space.";
        return result;
    }
    const auto material_flag_count =
        input.geometry_materials.size() + input.override_materials.size();
    if (material_flag_count > uint_maximum) {
        result.diagnostic =
            "Scene traversal table exceeds the 32-bit device address space.";
        return result;
    }

    result.material_flags.reserve(material_flag_count);
    for (const auto &binding : input.geometry_materials) {
        result.material_flags.emplace_back(binding.flags);
    }
    for (const auto &binding : input.override_materials) {
        result.material_flags.emplace_back(binding.flags);
    }

    const auto base_material_count =
        static_cast<std::uint32_t>(input.geometry_materials.size());
    result.instances.reserve(input.instances.size());
    for (auto instance_index = std::size_t{0u};
         instance_index < input.instances.size(); ++instance_index) {
        const auto &instance = input.instances[instance_index];
        if (instance.geometry_index >= input.geometries.size()) {
            result.diagnostic =
                "Scene traversal instance references an unavailable geometry.";
            return result;
        }
        const auto &geometry = input.geometries[instance.geometry_index];
        const auto geometry_material_count =
            std::max(geometry.material_count, 1u);
        if (geometry.material_offset > input.geometry_materials.size() ||
            geometry_material_count >
                input.geometry_materials.size() - geometry.material_offset) {
            result.diagnostic =
                "Scene traversal geometry material range is out of bounds.";
            return result;
        }
        if (instance.override_offset > input.override_materials.size() ||
            instance.override_count >
                input.override_materials.size() - instance.override_offset) {
            result.diagnostic =
                "Scene traversal instance override range is out of bounds.";
            return result;
        }
        if (instance.cycles_primitive_offset !=
            geometry.cycles_primitive_offset) {
            result.diagnostic =
                "Scene traversal instance and geometry primitive offsets "
                "disagree.";
            return result;
        }
        if (geometry.primitive_kind >
                scene_traversal_primitive_kind_mask ||
            geometry.curve_subdivision_level >
                scene_traversal_curve_subdivision_maximum) {
            result.diagnostic =
                "Scene traversal primitive metadata cannot be packed exactly.";
            return result;
        }

        const auto cycles_object_index =
            instance.cycles_object_index !=
                    cycles_shader_identity::invalid_index
                ? instance.cycles_object_index
                : static_cast<std::uint32_t>(instance_index);
        result.instances.emplace_back(SceneTraversalInstanceGpu{
            .bindless_base = geometry.bindless_base,
            .geometry_material_offset = geometry.material_offset,
            .geometry_material_count = geometry_material_count,
            .override_material_offset =
                base_material_count + instance.override_offset,
            .override_material_count = instance.override_count,
            .cycles_object_index = cycles_object_index,
            .cycles_primitive_offset = instance.cycles_primitive_offset,
            .primitive_kind_and_curve_subdivision =
                pack_scene_traversal_primitive(
                    geometry.primitive_kind,
                    geometry.curve_subdivision_level)});
    }

    if (!result.instances.empty()) {
        std::uint32_t max_object_index = 0u;
        bool within_object_table_limit = true;
        constexpr std::uint32_t max_dense_object_table_index = 16'777'215u;
        for (const auto &inst : result.instances) {
            if (inst.cycles_object_index > max_dense_object_table_index) {
                within_object_table_limit = false;
                break;
            }
            max_object_index =
                std::max(max_object_index, inst.cycles_object_index);
        }
        if (within_object_table_limit) {
            result.object_to_instance.assign(
                static_cast<std::size_t>(max_object_index) + 1u,
                luisa::make_uint2(~luisa::uint{0u}, 0u));
            bool injective = true;
            for (std::size_t i = 0u; i < result.instances.size(); ++i) {
                const auto obj = result.instances[i].cycles_object_index;
                if (result.object_to_instance[obj].x != ~luisa::uint{0u}) {
                    injective = false;
                    break;
                }
                result.object_to_instance[obj] = luisa::make_uint2(
                    static_cast<luisa::uint>(i),
                    result.instances[i].cycles_primitive_offset);
            }
            if (!injective) {
                result.object_to_instance.clear();
            }
        }
    }

    const auto eval_transparent = [&](const SceneTraversalInstanceGpu &inst_rec,
                                      std::uint32_t slot) noexcept {
        std::uint32_t flags = result.material_flags[
            inst_rec.geometry_material_offset +
            std::min(slot, inst_rec.geometry_material_count - 1u)];
        if (slot < inst_rec.override_material_count) {
            flags = result.material_flags[
                inst_rec.override_material_offset + slot];
        }
        return (flags & material_flag_has_transparent_shadow) != 0u;
    };

    bool has_any_transparent = false;
    bool can_build_triangle_opacity = !result.instances.empty();
    result.shadow_opacity_words.assign(result.instances.size(), 0u);
    std::map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>,
             luisa::uint>
        mixed_cache;
    for (std::size_t i = 0u; i < result.instances.size(); ++i) {
        const auto &src_inst = input.instances[i];
        const auto &geometry = input.geometries[src_inst.geometry_index];
        const auto &inst_rec = result.instances[i];
        const auto max_slot_probe = std::max(
            inst_rec.geometry_material_count, inst_rec.override_material_count);
        bool seen_transparent = false;
        bool seen_opaque = false;
        for (std::uint32_t s = 0u; s <= max_slot_probe; ++s) {
            if (eval_transparent(inst_rec, s)) {
                seen_transparent = true;
            } else {
                seen_opaque = true;
            }
            if (seen_transparent && seen_opaque) {
                break;
            }
        }
        if (!seen_transparent) {
            result.shadow_opacity_words[i] = 0u;
            continue;
        }
        if (!seen_opaque) {
            result.shadow_opacity_words[i] = ~luisa::uint{0u};
            has_any_transparent = true;
            continue;
        }
        if (geometry.primitive_kind == geometry_kind_triangle &&
            src_inst.geometry_index < input.geometry_uploads.size()) {
            const auto key = std::make_tuple(
                src_inst.geometry_index,
                src_inst.override_offset,
                src_inst.override_count);
            auto cached = mixed_cache.find(key);
            if (cached == mixed_cache.end()) {
                const auto &upload =
                    input.geometry_uploads[src_inst.geometry_index];
                const auto tri_count = upload.triangles.size();
                bool tri_transparent = false;
                bool tri_opaque = false;
                for (std::size_t prim = 0u; prim < tri_count; ++prim) {
                    const auto slot =
                        prim < upload.triangle_material_slots.size()
                            ? upload.triangle_material_slots[prim]
                            : 0u;
                    if (eval_transparent(inst_rec, slot)) {
                        tri_transparent = true;
                    } else {
                        tri_opaque = true;
                    }
                    if (tri_transparent && tri_opaque) {
                        break;
                    }
                }
                luisa::uint code = 0u;
                if (!tri_transparent) {
                    code = 0u;
                } else if (!tri_opaque) {
                    code = ~luisa::uint{0u};
                } else {
                    const auto word_count = (tri_count + 31u) / 32u;
                    if (result.shadow_opacity_words.size() + word_count >=
                        static_cast<std::size_t>(~luisa::uint{0u})) {
                        can_build_triangle_opacity = false;
                        has_any_transparent = true;
                        continue;
                    }
                    code = static_cast<luisa::uint>(
                        result.shadow_opacity_words.size());
                    result.shadow_opacity_words.resize(
                        result.shadow_opacity_words.size() + word_count, 0u);
                    for (std::size_t prim = 0u; prim < tri_count; ++prim) {
                        const auto slot =
                            prim < upload.triangle_material_slots.size()
                                ? upload.triangle_material_slots[prim]
                                : 0u;
                        if (eval_transparent(inst_rec, slot)) {
                            result.shadow_opacity_words[code + (prim >> 5u)] |=
                                (1u << (prim & 31u));
                        }
                    }
                }
                cached = mixed_cache.emplace(key, code).first;
            }
            const auto code = cached->second;
            result.shadow_opacity_words[i] = code;
            has_any_transparent |= (code != 0u);
        } else {
            // Non-triangle instances (curves) are visited only through
            // ProceduralCandidate, which resolves curve material flags directly.
            // Keep triangle shadow-opacity words valid for all surface instances.
            result.shadow_opacity_words[i] = ~luisa::uint{0u};
            has_any_transparent = true;
        }
    }
    if (!can_build_triangle_opacity) {
        result.shadow_opacity_words.clear();
    }
    result.has_transparent_shadow_materials = has_any_transparent;
    return result;
}

std::uint32_t encode_attribute_domain(
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

Vec3f from_luisa(luisa::float3 value) noexcept {
    return {value.x, value.y, value.z};
}

SceneTableUploadResult
SceneTableUploadComponent::upload(const std::shared_ptr<LuisaSceneData> &scene,
                                  Stream &stream,
                                  SceneTableUploadInput input) const {
    provide_inert_storage(input);
    auto traversal_tables = build_scene_traversal_tables(
        {.geometries = std::span<const GeometryGpu>{
             input.geometries.data(), input.geometries.size()},
         .instances = std::span<const InstanceGpu>{
             input.instances.data(), input.instances.size()},
         .geometry_materials = std::span<const MaterialBindingGpu>{
             input.geometry_materials.data(), input.geometry_materials.size()},
         .override_materials = std::span<const MaterialBindingGpu>{
             input.override_materials.data(), input.override_materials.size()},
         .geometry_uploads = input.geometry_uploads});
    if (!traversal_tables.ok()) {
        return {.diagnostic = std::move(traversal_tables.diagnostic)};
    }
    scene->has_transparent_shadow_materials =
        traversal_tables.has_transparent_shadow_materials;
    scene->has_mixed_shadow_opacity_instances =
        traversal_tables.shadow_opacity_words.size() >
        traversal_tables.instances.size();
    if (!traversal_tables.object_to_instance.empty()) {
        scene->traversal_object_instance_count =
            static_cast<std::uint32_t>(
                traversal_tables.object_to_instance.size());
        scene->traversal_object_instance_buffer.emplace(
            scene->device.create_buffer<luisa::uint2>(
                traversal_tables.object_to_instance.size()));
    } else {
        scene->traversal_object_instance_count = 0u;
        scene->traversal_object_instance_buffer.reset();
    }
    if (!traversal_tables.shadow_opacity_words.empty()) {
        scene->traversal_shadow_opacity_buffer.emplace(
            scene->device.create_buffer<luisa::uint>(
                traversal_tables.shadow_opacity_words.size()));
    } else {
        scene->traversal_shadow_opacity_buffer.reset();
    }
    if (scene->subsurface_accel &&
        !scene->primary_to_subsurface_local_host.empty()) {
        if (scene->primary_to_subsurface_local_host.size() <
            input.instances.size()) {
            scene->primary_to_subsurface_local_host.resize(
                input.instances.size(), ~luisa::uint{0u});
        }
        scene->primary_to_subsurface_local_buffer.emplace(
            scene->device.create_buffer<luisa::uint>(
                scene->primary_to_subsurface_local_host.size()));
    } else {
        scene->primary_to_subsurface_local_buffer.reset();
    }
    if (scene->volume_boundary_accel &&
        scene->volume_boundary_accel->size() > 0u &&
        !scene->primary_to_volume_local_host.empty()) {
        if (scene->primary_to_volume_local_host.size() <
            input.instances.size()) {
            scene->primary_to_volume_local_host.resize(
                input.instances.size(), ~luisa::uint{0u});
        }
        bool all_vol = true;
        for (std::size_t i = 0u; i < traversal_tables.instances.size(); ++i) {
            if (i >= scene->primary_to_volume_local_host.size() ||
                scene->primary_to_volume_local_host[i] == ~luisa::uint{0u}) {
                continue;
            }
            const auto &inst_rec = traversal_tables.instances[i];
            const auto slot_count = std::max(
                inst_rec.geometry_material_count,
                inst_rec.override_material_count);
            for (std::uint32_t s = 0u; s < slot_count; ++s) {
                std::uint32_t eff_flags = traversal_tables.material_flags[
                    inst_rec.geometry_material_offset +
                    std::min(s, inst_rec.geometry_material_count - 1u)];
                if (s < inst_rec.override_material_count) {
                    eff_flags = traversal_tables.material_flags[
                        inst_rec.override_material_offset + s];
                }
                if ((eff_flags & material_flag_has_volume) == 0u) {
                    all_vol = false;
                    break;
                }
            }
            if (!all_vol) { break; }
        }
        scene->all_volume_boundary_primitives_have_volume = all_vol;
        scene->primary_to_volume_local_buffer.emplace(
            scene->device.create_buffer<luisa::uint>(
                scene->primary_to_volume_local_host.size()));
    } else {
        scene->volume_boundary_accel.reset();
        scene->primary_to_volume_local_buffer.reset();
        scene->all_volume_boundary_primitives_have_volume = false;
    }
    scene->geometry_buffer =
        scene->device.create_buffer<GeometryGpu>(input.geometries.size());
    scene->instance_buffer =
        scene->device.create_buffer<InstanceGpu>(input.instances.size());
    scene->ambient_occlusion_object_distance_buffer =
        scene->device.create_buffer<float>(
            input.ambient_occlusion_object_distances.size());
    scene->geometry_material_buffer =
        scene->device.create_buffer<MaterialBindingGpu>(
            input.geometry_materials.size());
    scene->override_material_buffer =
        scene->device.create_buffer<MaterialBindingGpu>(
            input.override_materials.size());
    scene->traversal_instance_buffer =
        scene->device.create_buffer<SceneTraversalInstanceGpu>(
            traversal_tables.instances.size());
    scene->traversal_material_flags_buffer =
        scene->device.create_buffer<luisa::uint>(
            traversal_tables.material_flags.size());
    scene->light_buffer =
        scene->device.create_buffer<LightGpu>(input.lights.size());
    luisa::vector<luisa::float4> intersectable_light_bounds;
    luisa::vector<luisa::uint> intersectable_light_indices;
    std::uint32_t intersectable_light_type_mask = 0u;
    bool any_intersectable_light_camera_visible = false;
    bool any_intersectable_light_indirect_mis = false;
    {
        const std::size_t regular_light_count = std::min<std::size_t>(
            scene->light_count, input.lights.size());
        intersectable_light_bounds.reserve(regular_light_count);
        intersectable_light_indices.reserve(regular_light_count);
        constexpr auto area_type =
            static_cast<std::uint32_t>(LightType::area);
        constexpr auto point_type =
            static_cast<std::uint32_t>(LightType::point);
        constexpr auto spot_type =
            static_cast<std::uint32_t>(LightType::spot);
        for (std::size_t light_index = 0u;
             light_index < regular_light_count; ++light_index) {
            const auto &light = input.lights[light_index];
            const bool use_mis = (light.flags & light_flag_use_mis) != 0u;
            if (!use_mis && light.visibility_mask == 0u) {
                continue;
            }
            float bound_w = 0.0f;
            if (light.type == area_type) {
                if (!(light.size_u > 0.0f && light.size_v > 0.0f)) {
                    continue;
                }
                const auto az = light.axis_z;
                const auto dot3 = [](luisa::float3 a, luisa::float3 b) noexcept {
                    return a.x * b.x + a.y * b.y + a.z * b.z;
                };
                const auto ax_z = dot3(light.axis_x, az);
                const auto ay_z = dot3(light.axis_y, az);
                const auto u_proj = luisa::make_float3(
                    light.axis_x.x - ax_z * az.x,
                    light.axis_x.y - ax_z * az.y,
                    light.axis_x.z - ax_z * az.z);
                const auto v_proj = luisa::make_float3(
                    light.axis_y.x - ay_z * az.x,
                    light.axis_y.y - ay_z * az.y,
                    light.axis_y.z - ay_z * az.z);
                const double a = static_cast<double>(dot3(u_proj, u_proj));
                const double b = static_cast<double>(dot3(u_proj, v_proj));
                const double d = static_cast<double>(dot3(v_proj, v_proj));
                const double det = a * d - b * b;
                if (det > 1.0e-12) {
                    const double inv_det = 1.0 / det;
                    const double su = 0.5 * static_cast<double>(light.size_u);
                    const double sv = 0.5 * static_cast<double>(light.size_v);
                    const double ex = (d * static_cast<double>(u_proj.x) -
                                       b * static_cast<double>(v_proj.x)) *
                                      inv_det;
                    const double ey = (d * static_cast<double>(u_proj.y) -
                                       b * static_cast<double>(v_proj.y)) *
                                      inv_det;
                    const double ez = (d * static_cast<double>(u_proj.z) -
                                       b * static_cast<double>(v_proj.z)) *
                                      inv_det;
                    const double fx = (a * static_cast<double>(v_proj.x) -
                                       b * static_cast<double>(u_proj.x)) *
                                      inv_det;
                    const double fy = (a * static_cast<double>(v_proj.y) -
                                       b * static_cast<double>(u_proj.y)) *
                                      inv_det;
                    const double fz = (a * static_cast<double>(v_proj.z) -
                                       b * static_cast<double>(u_proj.z)) *
                                      inv_det;
                    const double hux = su * ex;
                    const double huy = su * ey;
                    const double huz = su * ez;
                    const double hvx = sv * fx;
                    const double hvy = sv * fy;
                    const double hvz = sv * fz;
                    const double px = hux + hvx;
                    const double py = huy + hvy;
                    const double pz = huz + hvz;
                    const double mx = hux - hvx;
                    const double my = huy - hvy;
                    const double mz = huz - hvz;
                    const double r_sq = std::max(
                        px * px + py * py + pz * pz,
                        mx * mx + my * my + mz * mz);
                    const double r = std::sqrt(std::max(r_sq, 0.0));
                    const double r_pad = r * 1.01 + 1.0e-2;
                    const double bound_w_d = std::min(r_pad * r_pad, 1.0e36);
                    bound_w = std::isfinite(bound_w_d)
                                  ? static_cast<float>(bound_w_d)
                                  : 1.0e38f;
                } else {
                    bound_w = 1.0e38f;
                }
            } else if (light.type == point_type || light.type == spot_type) {
                if (!(light.radius > 0.0f)) {
                    continue;
                }
                const double r_pad =
                    static_cast<double>(light.radius) * 1.01 + 1.0e-2;
                const double bound_w_d = std::min(r_pad * r_pad, 1.0e36);
                bound_w = std::isfinite(bound_w_d)
                              ? static_cast<float>(bound_w_d)
                              : 1.0e38f;
            } else {
                continue;
            }
            intersectable_light_bounds.emplace_back(
                luisa::make_float4(light.position.x, light.position.y,
                                   light.position.z, bound_w));
            intersectable_light_indices.emplace_back(
                static_cast<luisa::uint>(light_index));
            intersectable_light_type_mask |= (1u << light.type);
            any_intersectable_light_camera_visible |=
                ((light.visibility_mask & camera_visibility) != 0u);
            any_intersectable_light_indirect_mis |= use_mis;
        }
    }
    scene->intersectable_light_count =
        static_cast<std::uint32_t>(intersectable_light_indices.size());
    scene->intersectable_light_type_mask = intersectable_light_type_mask;
    scene->any_intersectable_light_camera_visible =
        any_intersectable_light_camera_visible;
    scene->any_intersectable_light_indirect_mis =
        any_intersectable_light_indirect_mis;
    if (intersectable_light_bounds.empty()) {
        intersectable_light_bounds.emplace_back(luisa::make_float4(0.0f));
        intersectable_light_indices.emplace_back(0u);
    }
    scene->intersectable_light_bounds =
        scene->device.create_buffer<luisa::float4>(
            intersectable_light_bounds.size());
    scene->intersectable_light_indices =
        scene->device.create_buffer<luisa::uint>(
            intersectable_light_indices.size());
    scene->emissive_triangle_buffer =
        scene->device.create_buffer<EmissiveTriangleGpu>(
            input.emissive_triangles.size());
    luisa::vector<float> light_distribution_cdf;
    light_distribution_cdf.reserve(input.light_distribution.size());
    for (const auto &entry : input.light_distribution) {
        light_distribution_cdf.emplace_back(entry.cumulative);
    }
    constexpr std::uint32_t k_light_dist_guide_bins = 8192u;
    luisa::vector<luisa::uint> light_distribution_cdf_guide;
    std::uint32_t guide_bins = 0u;
    if (scene->light_distribution_count > 64u &&
        light_distribution_cdf.size() >
            static_cast<std::size_t>(scene->light_distribution_count)) {
        const auto cdf_len =
            static_cast<std::size_t>(scene->light_distribution_count) + 1u;
        bool monotonic = std::isfinite(light_distribution_cdf[0u]);
        for (std::size_t i = 1u; monotonic && i < cdf_len; ++i) {
            if (!(light_distribution_cdf[i] >=
                  light_distribution_cdf[i - 1u])) {
                monotonic = false;
            }
        }
        if (monotonic) {
            guide_bins = k_light_dist_guide_bins;
            light_distribution_cdf_guide.resize(
                static_cast<std::size_t>(guide_bins) + 1u);
            light_distribution_cdf_guide[0u] = 0u;
            light_distribution_cdf_guide[guide_bins] =
                scene->light_distribution_count + 1u;
            std::size_t cursor = 0u;
            for (std::uint32_t b = 1u; b < guide_bins; ++b) {
                const float t = static_cast<float>(b) /
                                static_cast<float>(guide_bins);
                while (cursor < cdf_len &&
                       !(t < light_distribution_cdf[cursor])) {
                    ++cursor;
                }
                light_distribution_cdf_guide[b] =
                    static_cast<luisa::uint>(cursor);
            }
        }
    }
    if (light_distribution_cdf_guide.empty()) {
        light_distribution_cdf_guide.emplace_back(0u);
    }
    scene->light_distribution_cdf_guide_bins = guide_bins;

    scene->light_distribution_buffer =
        scene->device.create_buffer<LightDistributionGpu>(
            input.light_distribution.size());
    scene->light_distribution_cdf_buffer =
        scene->device.create_buffer<float>(light_distribution_cdf.size());
    scene->light_distribution_cdf_guide_buffer =
        scene->device.create_buffer<luisa::uint>(
            light_distribution_cdf_guide.size());
    scene->light_tree_node_buffer =
        scene->device.create_buffer<LightTreeNodeGpu>(
            input.light_tree_nodes.size());
    scene->light_tree_emitter_buffer =
        scene->device.create_buffer<LightTreeEmitterGpu>(
            input.light_tree_emitters.size());
    scene->light_tree_emitter_mapping_buffer =
        scene->device.create_buffer<luisa::uint2>(
            input.light_tree_emitter_mappings.size());
    scene->light_tree_triangle_emitter_mapping_buffer =
        scene->device.create_buffer<luisa::uint2>(
            input.light_tree_triangle_emitter_mappings.size());
    scene->light_tree_mesh_triangle_buffer =
        scene->device.create_buffer<luisa::uint>(
            input.light_tree_mesh_triangles.size());
    scene->light_tree_triangle_lookup_buffer =
        scene->device.create_buffer<luisa::uint4>(
            input.light_tree_triangle_lookup.size());

    stream << scene->geometry_buffer.copy_from(luisa::span{input.geometries})
           << scene->instance_buffer.copy_from(luisa::span{input.instances})
           << scene->ambient_occlusion_object_distance_buffer.copy_from(
                  luisa::span{input.ambient_occlusion_object_distances})
           << scene->geometry_material_buffer.copy_from(
                  luisa::span{input.geometry_materials})
           << scene->override_material_buffer.copy_from(
                  luisa::span{input.override_materials})
           << scene->traversal_instance_buffer.copy_from(
                  luisa::span{traversal_tables.instances})
           << scene->traversal_material_flags_buffer.copy_from(
                  luisa::span{traversal_tables.material_flags})
           << scene->light_buffer.copy_from(luisa::span{input.lights})
           << scene->intersectable_light_bounds.copy_from(
                  luisa::span{intersectable_light_bounds})
           << scene->intersectable_light_indices.copy_from(
                  luisa::span{intersectable_light_indices})
           << scene->emissive_triangle_buffer.copy_from(
                  luisa::span{input.emissive_triangles})
           << scene->light_distribution_buffer.copy_from(
                  luisa::span{input.light_distribution})
           << scene->light_distribution_cdf_buffer.copy_from(
                  luisa::span{light_distribution_cdf})
           << scene->light_distribution_cdf_guide_buffer.copy_from(
                  luisa::span{light_distribution_cdf_guide})
           << scene->light_tree_node_buffer.copy_from(
                  luisa::span{input.light_tree_nodes})
           << scene->light_tree_emitter_buffer.copy_from(
                  luisa::span{input.light_tree_emitters})
           << scene->light_tree_emitter_mapping_buffer.copy_from(
                  luisa::span{input.light_tree_emitter_mappings})
           << scene->light_tree_triangle_emitter_mapping_buffer.copy_from(
                  luisa::span{input.light_tree_triangle_emitter_mappings})
           << scene->light_tree_mesh_triangle_buffer.copy_from(
                  luisa::span{input.light_tree_mesh_triangles})
           << scene->light_tree_triangle_lookup_buffer.copy_from(
                  luisa::span{input.light_tree_triangle_lookup});
    if (scene->traversal_object_instance_buffer) {
        stream << scene->traversal_object_instance_buffer->copy_from(
            luisa::span{traversal_tables.object_to_instance});
    }
    if (scene->traversal_shadow_opacity_buffer) {
        stream << scene->traversal_shadow_opacity_buffer->copy_from(
            luisa::span{traversal_tables.shadow_opacity_words});
    }
    if (scene->primary_to_subsurface_local_buffer) {
        stream << scene->primary_to_subsurface_local_buffer->copy_from(
            luisa::span{scene->primary_to_subsurface_local_host});
    }
    if (scene->primary_to_volume_local_buffer) {
        stream << scene->primary_to_volume_local_buffer->copy_from(
            luisa::span{scene->primary_to_volume_local_host});
    }
    stream << scene->texture_heap.update() << scene->heap.update()
           << scene->accel.build();
    if (scene->subsurface_accel) {
        stream << scene->subsurface_accel->build();
    }
    if (scene->ambient_occlusion_local_accel) {
        stream << scene->ambient_occlusion_local_accel->build();
    }
    if (scene->volume_boundary_accel) {
        stream << scene->volume_boundary_accel->build();
    }
    stream << synchronize();
    return {};
}

}// namespace psycles::luisa_backend::detail
