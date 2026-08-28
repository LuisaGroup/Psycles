#include "path_tracer_ambient_occlusion.h"

#include <psycles/luisa/cycles_sample_mapping.h>
#include <psycles/luisa/cycles_sampler.h>
#include <psycles/luisa/surface_ray.h>
#include <psycles/sampling/tabulated_sobol.h>

namespace psycles::luisa_backend::detail {

PathSurfaceAmbientOcclusionProvider::PathSurfaceAmbientOcclusionProvider(
    std::shared_ptr<LuisaSceneData> scene,
    std::shared_ptr<const SceneTraversalComponent> traversal,
    const PathSurfaceAmbientOcclusionContext &context) noexcept
    : _scene{std::move(scene)},
      _traversal{std::move(traversal)},
      _context{context} {}

Float PathSurfaceAmbientOcclusionProvider::evaluate(
    const SurfacePoint &point,
    const SurfaceAmbientOcclusionInput &input) const noexcept {
    LUISA_ASSERT(
        _scene->ambient_occlusion_distance_buffer,
        "AO provider requires the scene distance buffer.");
    const Expr<Buffer<float>> ambient_occlusion_distance{
        *_scene->ambient_occlusion_distance_buffer};
    const auto maximum_distance = select(
        input.distance,
        ambient_occlusion_distance->read(0u),
        input.global_radius);
    Float ao = 1.0f;
    const auto valid =
        (maximum_distance > 0.0f) &
        (input.samples > 0u) &
        (_context.source_object != surface_ray::invalid_primitive);
    $if(valid) {
        const auto normal = select(input.normal, -input.normal, input.inside);
        const auto basis = cycles_sample_mapping::make_orthonormals(normal);
        UInt unoccluded = 0u;
        UInt branch = 0u;
        $while(branch < input.samples) {
            const auto random = cycles_sampler::sample_branched_2d(
                _context.sobol_table,
                _context.sobol_sequence_size,
                _context.sample_index,
                _context.rng_hash,
                _context.rng_offset,
                branch,
                input.samples,
                sampling::tabulated_sobol::surface_ao_dimension);
            const auto disk =
                cycles_sample_mapping::sample_uniform_disk(random);
            const auto local = make_float3(
                disk.x,
                disk.y,
                sqrt(max(1.0f - dot(disk, disk), 0.0f)));
            const auto direction =
                local.x * basis.tangent +
                local.y * basis.bitangent +
                local.z * normal;
            Var<luisa::compute::Ray> ray = make_ray(
                point.position, direction, 0.0f, maximum_distance);
            const auto blocked = _traversal->ambient_occluded(
                _scene,
                ray,
                {.object = _context.source_object,
                 .primitive = _context.source_primitive},
                input.only_local);
            unoccluded += select(0u, 1u, !blocked);
            branch += 1u;
        };
        ao = cast<float>(unoccluded) / cast<float>(input.samples);
    };
    return ao;
}

}// namespace psycles::luisa_backend::detail
