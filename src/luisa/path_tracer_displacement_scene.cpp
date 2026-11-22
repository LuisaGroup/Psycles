#include "path_tracer_displacement_scene.h"

#include "cycles_svm_internal.h"
#include "path_tracer_cycles_svm_kernel_globals.h"
#include "path_tracer_cycles_svm_scene.h"
#include "path_tracer_cycles_svm_shader_data.h"
#include "path_tracer_tangent_space.h"

#include <psycles/compiler/cycles_camera.h>
#include <psycles/compiler/cycles_transform.h>
#include <psycles/luisa/cycles_svm.h>
#include <psycles/luisa/cycles_svm_translator.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <utility>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

namespace svm = cycles_svm;
namespace abi = compiler::cycles_svm;

struct DisplacementObjectContext {
    contract::InstanceId instance_id{};
    const contract::InstanceDesc *instance{};
    std::uint32_t instance_index{};
};

[[nodiscard]] luisa::float3 host_add(
    luisa::float3 a,
    luisa::float3 b) noexcept {
    return luisa::make_float3(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z);
}

[[nodiscard]] luisa::float3 host_subtract(
    luisa::float3 a,
    luisa::float3 b) noexcept {
    return luisa::make_float3(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z);
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
        result.try_emplace(
            instance.geometry,
            DisplacementObjectContext{
                .instance_id = id,
                .instance = &instance,
                .instance_index = index});
        ++index;
    }
    return result;
}

[[nodiscard]] Float3 ensure_finite_device(
    Float3 value) noexcept {
    using namespace luisa::compute;
    return make_float3(
        select(0.0f, value.x, !luisa::compute::dsl::isnan(value.x) & !luisa::compute::dsl::isinf(value.x)),
        select(0.0f, value.y, !luisa::compute::dsl::isnan(value.y) & !luisa::compute::dsl::isinf(value.y)),
        select(0.0f, value.z, !luisa::compute::dsl::isnan(value.z) & !luisa::compute::dsl::isinf(value.z)));
}

}// namespace

MeshDisplacementSceneBuildResult
MeshDisplacementSceneComponent::build(
    const std::shared_ptr<LuisaSceneData> &scene,
    Stream &stream,
    const contract::SceneSnapshot &snapshot,
    std::span<const CyclesInstanceIntersectionPlan> intersection_plans,
    const std::map<contract::GeometryId, std::uint32_t> &geometry_indices,
    const std::map<contract::GeometryId, std::uint32_t> &triangle_primitive_offsets,
    const std::map<contract::GeometryId, std::uint32_t> &curve_primitive_offsets,
    std::vector<GeometryUpload> &uploads) const {
    MeshDisplacementSceneBuildResult result;
    if (!scene || !scene->cycles_svm) {
        result.diagnostic =
            "mesh displacement requires an active Cycles SVM scene runtime";
        return result;
    }

    struct DisplacedGeometryWork {
        contract::GeometryId geometry_id{};
        const contract::TriangleMeshDesc *geometry{};
        DisplacementObjectContext object;
        std::uint32_t geometry_index{};
        std::uint32_t primitive_offset{};
        std::uint32_t cycles_object_index{};
        CyclesMeshDisplacementPlan plan;
        std::vector<luisa::float3> pre_displacement_vertex_normals;
    };

    const auto objects = first_objects_by_geometry(snapshot);
    std::vector<DisplacedGeometryWork> work_items;
    for (const auto &[geometry_id, geometry] : snapshot.geometries) {
        const auto object_iter = objects.find(geometry_id);
        const auto overrides =
            object_iter != objects.end() &&
                    object_iter->second.instance != nullptr
                ? std::span<const contract::MaterialId>{
                      object_iter->second.instance->material_overrides}
                : std::span<const contract::MaterialId>{};
        auto plan = make_cycles_mesh_displacement_plan(
            geometry, snapshot.materials, overrides);
        if (plan.empty()) {
            continue;
        }
        const auto geometry_iter = geometry_indices.find(geometry_id);
        const auto prim_iter = triangle_primitive_offsets.find(geometry_id);
        if (geometry_iter == geometry_indices.end() ||
            prim_iter == triangle_primitive_offsets.end() ||
            object_iter == objects.end() ||
            object_iter->second.instance == nullptr) {
            result.diagnostic =
                "geometry '" + geometry.name +
                "' has displacement but no Cycles object context";
            return result;
        }
        const auto cycles_obj_iter =
            scene->cycles_svm->object_identities.instance_indices.find(
                object_iter->second.instance_id);
        if (cycles_obj_iter ==
            scene->cycles_svm->object_identities.instance_indices.end()) {
            result.diagnostic =
                "geometry '" + geometry.name +
                "' has displacement without a native Cycles object index";
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
                   luisa::span{upload.undisplaced_uv_tangents});

        std::vector<luisa::float3> pre_displacement_vertex_normals;
        if ((upload.attribute_domains & geometry_normal_corner) != 0u) {
            pre_displacement_vertex_normals = compute_cycles_vertex_normals(
                upload, std::vector<bool>(upload.triangles.size(), true));
        }

        work_items.emplace_back(DisplacedGeometryWork{
            .geometry_id = geometry_id,
            .geometry = &geometry,
            .object = object_iter->second,
            .geometry_index = geometry_index,
            .primitive_offset = prim_iter->second,
            .cycles_object_index = cycles_obj_iter->second,
            .plan = std::move(plan),
            .pre_displacement_vertex_normals =
                std::move(pre_displacement_vertex_normals)});
    }

    if (!work_items.empty()) {
        stream << scene->texture_heap.update() << scene->heap.update();
        if (!finalize_cycles_svm_scene_runtime(
                scene,
                snapshot,
                intersection_plans,
                uploads,
                geometry_indices,
                triangle_primitive_offsets,
                curve_primitive_offsets,
                result.diagnostic)) {
            return result;
        }
        upload_cycles_svm_scene_runtime(stream, *scene->cycles_svm);
        stream << synchronize();

        const auto shader_count = static_cast<std::size_t>(
            scene->cycles_svm->compilation.table.shader_count);
        const auto camera_transform = to_luisa(scene->camera.transform);
        const auto camera_inverse_transform = to_luisa(
            compiler::cycles_inverse_affine_transform(scene->camera.transform));
        const auto camera_setup = compiler::make_cycles_camera_projection(
            scene->camera, 1u, 1u);
        const auto camera_world_to_ndc = to_luisa(camera_setup.world_to_ndc);
        const auto &backend_name = scene->device.backend_name();
        const bool cpu_backend =
            backend_name == "fallback" || backend_name == "simd";

        for (auto &work : work_items) {
            auto &upload = uploads[work.geometry_index];
            luisa::vector<luisa::uint4> evaluations;
            evaluations.reserve(work.plan.evaluations.size());
            std::vector<std::uint8_t> active_shader_mask(shader_count, 0u);
            for (const auto &evaluation : work.plan.evaluations) {
                const auto shader_iter =
                    scene->cycles_svm->material_shader_indices.find(
                        evaluation.material);
                if (shader_iter ==
                        scene->cycles_svm->material_shader_indices.end() ||
                    shader_iter->second >= shader_count) {
                    result.diagnostic =
                        "geometry '" + work.geometry->name +
                        "' displacement references an unavailable material";
                    reset_cycles_svm_scene_runtime(*scene->cycles_svm);
                    return result;
                }
                const auto shader_index = shader_iter->second;
                active_shader_mask[shader_index] = 1u;
                evaluations.emplace_back(luisa::make_uint4(
                    evaluation.vertex_index,
                    evaluation.primitive_index,
                    evaluation.corner_index,
                    shader_index));
            }

            auto evaluation_buffer =
                scene->device.create_buffer<luisa::uint4>(evaluations.size());
            auto output_buffer =
                scene->device.create_buffer<luisa::float3>(evaluations.size());
            luisa::vector<luisa::float3> output(evaluations.size());

            Kernel1D evaluate =
                [scene,
                 &active_shader_mask,
                 camera_transform,
                 camera_inverse_transform,
                 camera_world_to_ndc,
                 cpu_backend](
                    luisa::compute::BufferVar<luisa::uint4> records,
                    luisa::compute::BufferVar<luisa::float3> output_values,
                    UInt cycles_object_index,
                    UInt primitive_offset,
                    luisa::compute::Float4x4 object_to_world) noexcept {
                    set_block_size(64u, 1u, 1u);
                    const auto record = records.read(dispatch_x());
                    const auto primitive = record.y;
                    const auto corner_index = record.z;
                    const auto shader_index = record.w;

                    Var<RenderKernelParameters> params;
                    params.full_width = 1u;
                    params.full_height = 1u;
                    params.camera_transform = camera_transform;
                    params.camera_inverse_transform = camera_inverse_transform;
                    params.camera_world_to_ndc = camera_world_to_ndc;

                    const PathCyclesSvmKernelGlobals kg{
                        scene,
                        params,
                        scene->camera.projection,
                        Bool{true},
                        Bool{true},
                        PathCyclesSvmShaderDomain::all_shaders,
                        active_shader_mask};
                    const auto world_to_object = inverse(object_to_world);
                    const svm::TransformState transforms{
                        camera_transform,
                        camera_inverse_transform,
                        object_to_world,
                        world_to_object};

                    const UInt cycles_prim = primitive_offset + primitive;
                    const UInt object_flags =
                        scene->cycles_svm->objects->object_flag_buffer->read(
                            cycles_object_index);
                    const Expr<Buffer<abi::KernelShader>> shaders{
                        *scene->cycles_svm->kernel_shader_buffer};
                    const UInt shader_flags =
                        shaders->read(shader_index).flags.cast<std::uint32_t>();

                    Float2 barycentric = make_float2(0.0f);
                    barycentric = select(
                        barycentric,
                        make_float2(1.0f, 0.0f),
                        corner_index == 1u);
                    barycentric = select(
                        barycentric,
                        make_float2(0.0f, 1.0f),
                        corner_index == 2u);

                    const auto zero = make_float3(0.0f);
                    const auto identity = make_float4x4(1.0f);
                    svm::ShaderData sd{
                        zero,
                        zero,
                        zero,
                        zero,
                        svm::primitive_triangle,
                        shader_index | svm::shader_smooth_normal,
                        shader_flags,
                        object_flags,
                        cycles_prim,
                        barycentric.x,
                        barycentric.y,
                        cycles_object_index,
                        0.5f,
                        0.0f,
                        0.0f,
                        0.0f,
                        1.0f,
                        0.0f,
                        0.0f,
                        1.0f,
                        zero,
                        zero,
                        identity,
                        identity,
                        0u,
                        nullptr};

                    cycles_svm_triangle_shader_setup(
                        kg,
                        transforms,
                        kg.triangle_vertices(cycles_object_index, cycles_prim),
                        sd);
                    sd.wi = sd.N;
                    sd.ray_P = sd.P;

                    const Float3 initial_P = sd.P;
                    const auto usage =
                        scene->cycles_svm->compilation.table.usage_for(
                            abi::SHADER_TYPE_DISPLACEMENT);
                    const svm::PathState path_state{0u, 0u};
                    svm::translate_svm_nodes_assume_valid(
                        svm::SvmTranslationRequest{
                            .kernel_globals = kg,
                            .table = scene->cycles_svm->compilation.table,
                            .words = *scene->cycles_svm->word_buffer,
                            .shader_type = abi::SHADER_TYPE_DISPLACEMENT,
                            .kernel_features =
                                scene->cycles_svm->kernel_features |
                                svm::kernel_feature_node_bump,
                            .node_feature_mask =
                                svm::kernel_feature_node_mask_displacement,
                            .node_types_used = usage.node_types_used,
                            .transform_state = transforms,
                            .shader_data = sd,
                            .path_state = path_state,
                            .stack_size =
                                std::max<std::size_t>(1u, usage.peak_stack_usage),
                            .noise_usage = usage.noise_usage,
                            .active_shader_mask = active_shader_mask},
                        cpu_backend ? svm::SvmBackendStrategy::direct_jit
                                    : svm::SvmBackendStrategy::auto_select);

                    const Float3 world_displacement = sd.P - initial_P;
                    const Float3 object_displacement =
                        (world_to_object *
                         make_float4(world_displacement, 0.0f))
                            .xyz();
                    output_values.write(
                        dispatch_x(),
                        ensure_finite_device(object_displacement));
                };

            luisa::compute::ShaderOption shader_options;
            shader_options.enable_cache = true;
            // Cycles builds device displacement with the same fast-math
            // policy as its render kernels. Displacement has a structural
            // finite-value contract, but no cross-backend last-bit contract;
            // do not serialize arithmetic merely to freeze an oracle hash.
            shader_options.enable_fast_math = true;
            auto shader = scene->device.compile(evaluate, shader_options);
            stream
                << evaluation_buffer.copy_from(luisa::span{evaluations})
                << shader(
                       evaluation_buffer,
                       output_buffer,
                       work.cycles_object_index,
                       work.primitive_offset,
                       to_luisa(work.object.instance->transform))
                       .dispatch(evaluations.size())
                << output_buffer.copy_to(luisa::span{output})
                << synchronize();

            for (std::size_t index = 0u; index < evaluations.size(); ++index) {
                const auto vertex = evaluations[index].x;
                upload.positions[vertex] = host_add(
                    upload.positions[vertex], output[index]);
            }

            recompute_cycles_displaced_normals(
                upload, work.plan, work.pre_displacement_vertex_normals);
            // Cycles regenerates current MikkTSpace after positions and any
            // affected normals change. ORIGINAL-base bindings remain on the
            // immutable pre-displacement frames captured above.
            recompute_cycles_tangent_space(upload);
            result.displaced_geometry_indices.emplace_back(work.geometry_index);
        }

        // Clear the pre-displacement DeviceScene geometry/object upload so
        // the main finalize_cycles_svm_scene_runtime call packs the displaced
        // positions, recomputed normals/tangents, and undisplaced snapshots.
        reset_cycles_svm_scene_runtime(*scene->cycles_svm);
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
                if (!layer.tangent_attribute_index ||
                    *layer.tangent_attribute_index >=
                        upload.attributes.size() ||
                    *layer.tangent_attribute_index >=
                        resource.attributes.size()) {
                    continue;
                }
                stream << resource.attributes[
                              *layer.tangent_attribute_index]
                              .copy_from(luisa::span{
                                  upload.attributes[
                                      *layer.tangent_attribute_index]
                                      .values});
            }
        }
        stream << resource.mesh.build();
    }
    return result;
}

}// namespace psycles::luisa_backend::detail
