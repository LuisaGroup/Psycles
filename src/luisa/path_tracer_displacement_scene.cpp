#include "path_tracer_displacement_scene.h"

#include "path_tracer_shader_services.h"
#include "path_tracer_tangent_space.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

struct DisplacementObjectContext {
    const contract::InstanceDesc *instance{};
    std::uint32_t instance_index{};
};

[[nodiscard]] luisa::float3 host_subtract(
    luisa::float3 a,
    luisa::float3 b) noexcept {
    return luisa::make_float3(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z);
}

[[nodiscard]] luisa::float3 host_add(
    luisa::float3 a,
    luisa::float3 b) noexcept {
    return luisa::make_float3(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z);
}

[[nodiscard]] luisa::float3 host_cross(
    luisa::float3 a,
    luisa::float3 b) noexcept {
    return luisa::make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

[[nodiscard]] luisa::float3 host_safe_normalize(
    luisa::float3 value,
    luisa::float3 fallback = {}) noexcept {
    const auto length_squared =
        value.x * value.x +
        value.y * value.y +
        value.z * value.z;
    if (!(length_squared > 0.0f) ||
        !std::isfinite(length_squared)) {
        return fallback;
    }
    const auto inverse_length =
        1.0f / std::sqrt(length_squared);
    return luisa::make_float3(
        value.x * inverse_length,
        value.y * inverse_length,
        value.z * inverse_length);
}

[[nodiscard]] luisa::float3 host_face_normal(
    const GeometryUpload &upload,
    const Triangle &triangle) noexcept {
    return host_safe_normalize(
        host_cross(
            host_subtract(
                upload.positions[triangle.i1],
                upload.positions[triangle.i0]),
            host_subtract(
                upload.positions[triangle.i2],
                upload.positions[triangle.i0])),
        luisa::make_float3(1.0f, 0.0f, 0.0f));
}

[[nodiscard]] std::vector<luisa::float3>
compute_cycles_vertex_normals(
    const GeometryUpload &upload,
    const std::vector<bool> &triangles) {
    std::vector<luisa::float3> normals(
        upload.positions.size(), luisa::make_float3(0.0f));
    for (std::size_t primitive = 0u;
         primitive < upload.triangles.size();
         ++primitive) {
        if (primitive >= triangles.size() || !triangles[primitive]) {
            continue;
        }
        const auto &triangle = upload.triangles[primitive];
        const auto normal = host_face_normal(upload, triangle);
        normals[triangle.i0] = host_add(normals[triangle.i0], normal);
        normals[triangle.i1] = host_add(normals[triangle.i1], normal);
        normals[triangle.i2] = host_add(normals[triangle.i2], normal);
    }
    return normals;
}

void recompute_cycles_displaced_normals(
    GeometryUpload &upload,
    const CyclesMeshDisplacementPlan &plan,
    const std::vector<luisa::float3> &pre_displacement_vertex_normals) {
    const auto corner_normals =
        (upload.attribute_domains & geometry_normal_corner) != 0u;
    auto recompute = plan.true_displacement_triangles;
    if (corner_normals) {
        recompute.assign(upload.triangles.size(), true);
    }
    if (!std::any_of(recompute.begin(), recompute.end(),
                     [](bool value) noexcept { return value; })) {
        return;
    }
    const auto post = compute_cycles_vertex_normals(upload, recompute);
    if (!corner_normals) {
        std::vector<bool> done(upload.positions.size(), false);
        for (std::size_t primitive = 0u;
             primitive < upload.triangles.size();
             ++primitive) {
            if (primitive >= recompute.size() || !recompute[primitive]) {
                continue;
            }
            const auto &triangle = upload.triangles[primitive];
            const std::array vertices{
                triangle.i0, triangle.i1, triangle.i2};
            for (const auto vertex : vertices) {
                if (vertex >= upload.normals.size() || done[vertex]) {
                    continue;
                }
                upload.normals[vertex] =
                    host_safe_normalize(post[vertex]);
                done[vertex] = true;
            }
        }
        return;
    }

    for (std::size_t primitive = 0u;
         primitive < upload.triangles.size();
         ++primitive) {
        const auto &triangle = upload.triangles[primitive];
        const auto corner = primitive * 3u;
        const auto smooth =
            primitive < upload.triangle_smooth.size() &&
            upload.triangle_smooth[primitive] != 0u;
        if (!smooth) {
            const auto normal = host_face_normal(upload, triangle);
            for (std::size_t offset = 0u; offset < 3u; ++offset) {
                upload.normals[corner + offset] = normal;
            }
            continue;
        }
        const std::array vertices{
            triangle.i0, triangle.i1, triangle.i2};
        for (std::size_t offset = 0u; offset < 3u; ++offset) {
            const auto vertex = vertices[offset];
            const auto before = host_safe_normalize(
                pre_displacement_vertex_normals[vertex]);
            const auto after = host_safe_normalize(post[vertex]);
            const auto delta = host_subtract(after, before);
            upload.normals[corner + offset] = host_safe_normalize(
                host_add(upload.normals[corner + offset], delta));
        }
    }
}

[[nodiscard]] std::map<contract::GeometryId,
                       DisplacementObjectContext>
first_objects_by_geometry(
    const contract::SceneSnapshot &snapshot) {
    std::map<contract::GeometryId, DisplacementObjectContext> result;
    std::uint32_t index = 0u;
    for (const auto &[id, instance] : snapshot.instances) {
        static_cast<void>(id);
        result.try_emplace(
            instance.geometry,
            DisplacementObjectContext{
                .instance = &instance,
                .instance_index = index});
        ++index;
    }
    return result;
}

[[nodiscard]] Float3 safe_normalize_device(
    Float3 value,
    Float3 fallback) noexcept {
    const auto length_squared = dot(value, value);
    return select(
        fallback,
        value * rsqrt(length_squared),
        length_squared > 1.0e-20f);
}

[[nodiscard]] Float3 ensure_finite_device(
    Float3 value) noexcept {
    using namespace luisa::compute;
    return make_float3(
        select(0.0f, value.x, !isnan(value.x) & !isinf(value.x)),
        select(0.0f, value.y, !isnan(value.y) & !isinf(value.y)),
        select(0.0f, value.z, !isnan(value.z) & !isinf(value.z)));
}

}// namespace

MeshDisplacementSceneBuildResult
MeshDisplacementSceneComponent::build(
    const std::shared_ptr<LuisaSceneData> &scene,
    Stream &stream,
    const contract::SceneSnapshot &snapshot,
    const std::map<contract::GeometryId, std::uint32_t> &geometry_indices,
    std::vector<GeometryUpload> &uploads,
    luisa::vector<AttributeBindingGpu> &attribute_bindings,
    luisa::vector<AttributeRangeGpu> &attribute_ranges) const {
    MeshDisplacementSceneBuildResult result;
    if (!scene) {
        result.diagnostic = "mesh displacement scene resource is null";
        return result;
    }
    const auto objects = first_objects_by_geometry(snapshot);
    const auto has_mesh_displacement = std::any_of(
        snapshot.geometries.begin(),
        snapshot.geometries.end(),
        [&](const auto &item) {
            const auto object = objects.find(item.first);
            const auto overrides =
                object != objects.end() && object->second.instance != nullptr
                    ? std::span<const contract::MaterialId>{
                          object->second.instance->material_overrides}
                    : std::span<const contract::MaterialId>{};
            return !make_cycles_mesh_displacement_plan(
                        item.second, snapshot.materials, overrides)
                        .empty();
        });
    if (has_mesh_displacement) {
        if (attribute_bindings.empty()) {
            attribute_bindings.emplace_back(AttributeBindingGpu{});
        }
        if (attribute_ranges.empty()) {
            attribute_ranges.emplace_back(AttributeRangeGpu{});
        }
        scene->attribute_binding_buffer =
            scene->device.create_buffer<AttributeBindingGpu>(
                attribute_bindings.size());
        scene->attribute_range_buffer =
            scene->device.create_buffer<AttributeRangeGpu>(
                attribute_ranges.size());
        scene->heap.emplace_on_update(
            scene->attribute_binding_slot,
            scene->attribute_binding_buffer);
        scene->heap.emplace_on_update(
            scene->attribute_range_slot,
            scene->attribute_range_buffer);
        stream
            << scene->attribute_binding_buffer.copy_from(
                   luisa::span{attribute_bindings})
            << scene->attribute_range_buffer.copy_from(
                   luisa::span{attribute_ranges})
            << scene->texture_heap.update()
            << scene->heap.update()
            << synchronize();
    }

    for (const auto &[geometry_id, geometry] : snapshot.geometries) {
        const auto object_iter = objects.find(geometry_id);
        const auto overrides =
            object_iter != objects.end() &&
                    object_iter->second.instance != nullptr
                ? std::span<const contract::MaterialId>{
                      object_iter->second.instance->material_overrides}
                : std::span<const contract::MaterialId>{};
        const auto plan = make_cycles_mesh_displacement_plan(
            geometry, snapshot.materials, overrides);
        if (plan.empty()) {
            continue;
        }
        const auto geometry_iter = geometry_indices.find(geometry_id);
        if (geometry_iter == geometry_indices.end() ||
            object_iter == objects.end() ||
            object_iter->second.instance == nullptr) {
            result.diagnostic =
                "geometry '" + geometry.name +
                "' has displacement but no Cycles object context";
            return result;
        }
        const auto geometry_index = geometry_iter->second;
        if (geometry_index >= uploads.size() ||
            geometry_index >= scene->geometries.size()) {
            result.diagnostic =
                "geometry '" + geometry.name +
                "' has an invalid displacement resource index";
            return result;
        }
        auto &upload = uploads[geometry_index];
        auto &resource = scene->geometries[geometry_index];
        // Cycles Mesh::add_undisplaced snapshots these attributes before
        // any vertex is moved. BOTH's bump-eval state and Normal Map's
        // ORIGINAL base must read the immutable representation even when a
        // neighboring TRUE triangle later changes shared normals.
        upload.undisplaced_positions = upload.positions;
        upload.undisplaced_normals = upload.normals;
        upload.undisplaced_uv_tangents = upload.uv_tangents;
        resource.undisplaced_positions.emplace(
            scene->device.create_buffer<luisa::float3>(
                upload.undisplaced_positions.size()));
        resource.undisplaced_normals.emplace(
            scene->device.create_buffer<luisa::float3>(
                upload.undisplaced_normals.size()));
        resource.undisplaced_uv_tangents.emplace(
            scene->device.create_buffer<luisa::float4>(
                upload.undisplaced_uv_tangents.size()));
        const auto bindless_base =
            geometry_index * geometry_bindless_stride;
        scene->heap.emplace_on_update(
            bindless_base + 10u,
            *resource.undisplaced_positions);
        scene->heap.emplace_on_update(
            bindless_base + 11u,
            *resource.undisplaced_normals);
        scene->heap.emplace_on_update(
            bindless_base + 12u,
            *resource.undisplaced_uv_tangents);
        stream
            << resource.undisplaced_positions->copy_from(
                   luisa::span{upload.undisplaced_positions})
            << resource.undisplaced_normals->copy_from(
                   luisa::span{upload.undisplaced_normals})
            << resource.undisplaced_uv_tangents->copy_from(
                   luisa::span{upload.undisplaced_uv_tangents})
            << scene->heap.update();
        const auto corner_normals =
            (upload.attribute_domains & geometry_normal_corner) != 0u;
        std::vector<luisa::float3> pre_displacement_vertex_normals;
        if (corner_normals) {
            pre_displacement_vertex_normals =
                compute_cycles_vertex_normals(
                    upload,
                    std::vector<bool>(upload.triangles.size(), true));
        }

        std::map<std::uint32_t, luisa::vector<luisa::uint4>> batches;
        for (const auto &evaluation : plan.evaluations) {
            const auto binding = scene->material_bindings.find(
                evaluation.material);
            if (binding == scene->material_bindings.end()) {
                result.diagnostic =
                    "geometry '" + geometry.name +
                    "' displacement references an unavailable material";
                return result;
            }
            const auto packed_corner =
                evaluation.corner_index |
                (((binding->second.flags &
                   material_flag_use_bump_map_correction) != 0u
                      ? 1u
                      : 0u)
                 << 8u);
            batches[binding->second.surface_tag].emplace_back(
                luisa::make_uint4(
                    evaluation.vertex_index,
                    evaluation.primitive_index,
                    packed_corner,
                    binding->second.parameter_block));
        }

        const auto &object = *object_iter->second.instance;
        for (const auto &[surface_tag, evaluations] : batches) {
            const auto *surface =
                scene->surfaces.implementation(surface_tag);
            if (surface == nullptr || evaluations.empty()) {
                result.diagnostic =
                    "geometry '" + geometry.name +
                    "' has an unavailable displacement surface";
                return result;
            }
            auto evaluation_buffer =
                scene->device.create_buffer<luisa::uint4>(
                    evaluations.size());
            auto output_buffer =
                scene->device.create_buffer<luisa::float3>(
                    evaluations.size());
            luisa::vector<luisa::float3> output(evaluations.size());

            Kernel1D evaluate =
                [scene, surface](
                    luisa::compute::BufferVar<luisa::uint4> records,
                    luisa::compute::BufferVar<luisa::float3> output_values,
                    UInt geometry_index_value,
                    UInt attribute_domains,
                    luisa::compute::Float4x4 object_to_world,
                    Float object_random,
                    UInt particle_index,
                    UInt instance_index) noexcept {
                    set_block_size(64u, 1u, 1u);
                    const auto record = records.read(dispatch_x());
                    const auto primitive = record.y;
                    const auto corner_index = record.z & 0xffu;
                    const auto bindless_base =
                        geometry_index_value * geometry_bindless_stride;
                    const auto triangle =
                        scene->heap->buffer<Triangle>(bindless_base)
                            .read(primitive);
                    const auto positions =
                        scene->heap->buffer<luisa::float3>(
                            bindless_base + 1u);
                    const auto normals =
                        scene->heap->buffer<luisa::float3>(
                            bindless_base + 2u);
                    const auto uvs =
                        scene->heap->buffer<luisa::float2>(
                            bindless_base + 3u);
                    const auto generated_values =
                        scene->heap->buffer<luisa::float3>(
                            bindless_base + 5u);
                    const auto island_random_values =
                        scene->heap->buffer<float>(
                            bindless_base + 6u);
                    const auto tangents =
                        scene->heap->buffer<luisa::float4>(
                            bindless_base + 7u);
                    const auto triangle_smooth =
                        scene->heap->buffer<luisa::uint>(
                            bindless_base + 8u)
                            .read(primitive) != 0u;
                    const auto corner = primitive * 3u;
                    const auto attribute_index =
                        [&](UInt point,
                            std::uint32_t offset,
                            std::uint32_t flag) noexcept {
                            return select(
                                point,
                                corner + offset,
                                (attribute_domains & flag) != 0u);
                        };
                    const auto p0 = positions.read(triangle.i0);
                    const auto p1 = positions.read(triangle.i1);
                    const auto p2 = positions.read(triangle.i2);
                    const auto n0 = normals.read(attribute_index(
                        triangle.i0, 0u, geometry_normal_corner));
                    const auto n1 = normals.read(attribute_index(
                        triangle.i1, 1u, geometry_normal_corner));
                    const auto n2 = normals.read(attribute_index(
                        triangle.i2, 2u, geometry_normal_corner));
                    const auto uv0 = uvs.read(attribute_index(
                        triangle.i0, 0u, geometry_uv_corner));
                    const auto uv1 = uvs.read(attribute_index(
                        triangle.i1, 1u, geometry_uv_corner));
                    const auto uv2 = uvs.read(attribute_index(
                        triangle.i2, 2u, geometry_uv_corner));
                    const auto generated0 =
                        generated_values.read(attribute_index(
                            triangle.i0, 0u, geometry_generated_corner));
                    const auto generated1 =
                        generated_values.read(attribute_index(
                            triangle.i1, 1u, geometry_generated_corner));
                    const auto generated2 =
                        generated_values.read(attribute_index(
                            triangle.i2, 2u, geometry_generated_corner));
                    const auto tangent0 = tangents.read(attribute_index(
                        triangle.i0, 0u, geometry_uv_tangent_corner));
                    const auto tangent1 = tangents.read(attribute_index(
                        triangle.i1, 1u, geometry_uv_tangent_corner));
                    const auto tangent2 = tangents.read(attribute_index(
                        triangle.i2, 2u, geometry_uv_tangent_corner));

                    Float2 barycentric = make_float2(0.0f);
                    barycentric = select(
                        barycentric,
                        make_float2(1.0f, 0.0f),
                        corner_index == 1u);
                    barycentric = select(
                        barycentric,
                        make_float2(0.0f, 1.0f),
                        corner_index == 2u);
                    const auto object_position = triangle_interpolate(
                        barycentric, p0, p1, p2);
                    const auto object_geometric_normal =
                        safe_normalize_device(
                            cross(p1 - p0, p2 - p0),
                            make_float3(0.0f, 0.0f, 1.0f));
                    const auto object_shading_normal =
                        safe_normalize_device(
                            select(
                                object_geometric_normal,
                                triangle_interpolate(
                                    barycentric, n0, n1, n2),
                                triangle_smooth),
                            object_geometric_normal);
                    const auto packed_object_tangent =
                        triangle_interpolate(
                            barycentric,
                            tangent0,
                            tangent1,
                            tangent2);
                    const auto world_to_object = inverse(object_to_world);
                    const auto normal_to_world = transpose(world_to_object);
                    const auto position =
                        (object_to_world *
                         make_float4(object_position, 1.0f))
                            .xyz();
                    const auto geometric_normal = safe_normalize_device(
                        (normal_to_world *
                         make_float4(object_geometric_normal, 0.0f))
                            .xyz(),
                        make_float3(0.0f, 0.0f, 1.0f));
                    const auto shading_normal = safe_normalize_device(
                        (normal_to_world *
                         make_float4(object_shading_normal, 0.0f))
                            .xyz(),
                        geometric_normal);
                    const auto world_p0 =
                        (object_to_world * make_float4(p0, 1.0f)).xyz();
                    const auto world_p1 =
                        (object_to_world * make_float4(p1, 1.0f)).xyz();
                    const auto world_p2 =
                        (object_to_world * make_float4(p2, 1.0f)).xyz();
                    const auto world_dpdu = world_p1 - world_p0;
                    const auto world_dpdv = world_p2 - world_p0;
                    const auto world_tangent = safe_normalize_device(
                        (object_to_world *
                         make_float4(packed_object_tangent.xyz(), 0.0f))
                            .xyz(),
                        safe_normalize_device(
                            world_dpdu,
                            make_float3(1.0f, 0.0f, 0.0f)));

                    BufferShaderServices services{
                        scene->scalar_parameter_buffer,
                        scene->vector_parameter_buffer,
                        scene->cycles_bsdf_table_buffer,
                        scene->texture_heap,
                        scene->heap,
                        scene->attribute_binding_slot,
                        scene->attribute_range_slot,
                        scene->nishita_texture_bindings,
                        scene->shader_color_space};
                    const SurfacePoint point{
                        .position = position,
                        .object_position = object_position,
                        .object_location =
                            (object_to_world *
                             make_float4(0.0f, 0.0f, 0.0f, 1.0f))
                                .xyz(),
                        .generated = triangle_interpolate(
                            barycentric,
                            generated0,
                            generated1,
                            generated2),
                        .geometric_normal = geometric_normal,
                        .shading_normal = shading_normal,
                        .object_shading_normal = object_shading_normal,
                        .object_tangent = packed_object_tangent.xyz(),
                        .tangent_sign = packed_object_tangent.w,
                        // This kernel runs before any vertex is displaced,
                        // so both geometry states are exactly identical.
                        .undisplaced_position = position,
                        .undisplaced_object_position = object_position,
                        .undisplaced_shading_normal = shading_normal,
                        .undisplaced_object_shading_normal =
                            object_shading_normal,
                        .undisplaced_object_tangent =
                            packed_object_tangent.xyz(),
                        .undisplaced_tangent_sign =
                            packed_object_tangent.w,
                        .normal_to_world_x =
                            (normal_to_world *
                             make_float4(1.0f, 0.0f, 0.0f, 0.0f))
                                .xyz(),
                        .normal_to_world_y =
                            (normal_to_world *
                             make_float4(0.0f, 1.0f, 0.0f, 0.0f))
                                .xyz(),
                        .normal_to_world_z =
                            (normal_to_world *
                             make_float4(0.0f, 0.0f, 1.0f, 0.0f))
                                .xyz(),
                        .dpdu = world_dpdu,
                        .dpdv = world_dpdv,
                        .dPdx = world_dpdu,
                        .dPdy = world_dpdv,
                        .object_dPdx = p1 - p0,
                        .object_dPdy = p2 - p0,
                        .undisplaced_dPdx = world_dpdu,
                        .undisplaced_dPdy = world_dpdv,
                        .undisplaced_object_dPdx = p1 - p0,
                        .undisplaced_object_dPdy = p2 - p0,
                        .generated_dx = generated1 - generated0,
                        .generated_dy = generated2 - generated0,
                        .incoming = shading_normal,
                        .uv = triangle_interpolate(
                            barycentric, uv0, uv1, uv2),
                        .uv_dx = uv1 - uv0,
                        .uv_dy = uv2 - uv0,
                        .geometry_index = geometry_index_value,
                        .barycentric = barycentric,
                        .barycentric_dx = make_float2(1.0f, 0.0f),
                        .barycentric_dy = make_float2(0.0f, 1.0f),
                        .instance_id = instance_index,
                        .primitive_id = primitive,
                        .parameter_block = record.w,
                        .object_random = object_random,
                        .particle_index = particle_index,
                        .random_per_island =
                            island_random_values.read(primitive),
                        .triangle_smooth = triangle_smooth,
                        .is_curve = false,
                        .curve_intercept = 0.0f,
                        .curve_length = 0.0f,
                        .curve_thickness = 0.0f,
                        .curve_tangent_normal = make_float3(0.0f),
                        .curve_random = 0.0f,
                        .ray_visibility = 0u,
                        .ray_events = 0u,
                        .ray_depth = 0u,
                        .diffuse_depth = 0u,
                        .glossy_depth = 0u,
                        .transparent_depth = 0u,
                        .transmission_depth = 0u,
                        .ray_length = 0.0f,
                        .time = 0.5f,
                        .use_bump_map_correction =
                            ((record.z >> 8u) & 1u) != 0u,
                        .back_facing = false};
                    const auto world_displacement =
                        surface->displacement(services, point);
                    const auto object_displacement =
                        (world_to_object *
                         make_float4(world_displacement, 0.0f))
                            .xyz();
                    output_values.write(
                        dispatch_x(),
                        ensure_finite_device(object_displacement));
                };
            luisa::compute::ShaderOption shader_options;
            shader_options.enable_cache = true;
            shader_options.enable_fast_math = false;
            auto shader = scene->device.compile(
                evaluate, shader_options);
            stream
                << evaluation_buffer.copy_from(
                       luisa::span{evaluations})
                << shader(
                       evaluation_buffer,
                       output_buffer,
                       geometry_index,
                       upload.attribute_domains,
                       to_luisa(object.transform),
                       std::clamp(object.random, 0.0f, 1.0f),
                       object.particle_index,
                       object_iter->second.instance_index)
                       .dispatch(evaluations.size())
                << output_buffer.copy_to(luisa::span{output})
                << synchronize();
            for (std::size_t index = 0u;
                 index < evaluations.size();
                 ++index) {
                const auto vertex = evaluations[index].x;
                upload.positions[vertex] = host_add(
                    upload.positions[vertex], output[index]);
            }
        }
        recompute_cycles_displaced_normals(
            upload, plan, pre_displacement_vertex_normals);
        // Cycles regenerates current MikkTSpace after positions and any
        // affected normals change. ORIGINAL-base bindings remain on the
        // immutable pre-displacement frames captured above.
        recompute_cycles_tangent_space(upload);
        result.displaced_geometry_indices.emplace_back(geometry_index);
    }
    const std::set<std::uint32_t> displaced{
        result.displaced_geometry_indices.begin(),
        result.displaced_geometry_indices.end()};
    for (std::size_t index = 0u;
         index < scene->geometries.size();
         ++index) {
        auto &resource = scene->geometries[index];
        if (displaced.contains(static_cast<std::uint32_t>(index))) {
            auto &upload = uploads[index];
            stream
                << resource.positions.copy_from(
                       luisa::span{upload.positions})
                << resource.normals.copy_from(
                       luisa::span{upload.normals})
                << resource.uv_tangents.copy_from(
                       luisa::span{upload.uv_tangents});
            for (const auto &layer : upload.uv_tangent_layers) {
                if (layer.tangent_attribute_index >=
                        upload.attributes.size() ||
                    layer.tangent_attribute_index >=
                        resource.attributes.size()) {
                    continue;
                }
                stream << resource.attributes[
                              layer.tangent_attribute_index]
                              .copy_from(luisa::span{
                                  upload.attributes[
                                      layer.tangent_attribute_index]
                                      .values});
            }
        }
        stream << resource.mesh.build();
    }
    return result;
}

}// namespace psycles::luisa_backend::detail
