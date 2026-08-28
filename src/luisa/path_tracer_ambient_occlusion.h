#pragma once

#include "path_kernel_scene_traversal.h"

namespace psycles::luisa_backend::detail {

// Per-path device expressions required by shader AO. This is host/JIT state:
// it is passed only to AO-aware surface programs, and none of its fields enter
// a no-AO callable signature or shader cache identity. Expressions are owned
// by value so a populated surface cannot retain a reference to a temporary
// context assembled by its enclosing shading stage.
struct PathSurfaceAmbientOcclusionContext {
    luisa::compute::Expr<
        luisa::compute::Buffer<luisa::float4>> sobol_table;
    UInt sobol_sequence_size;
    UInt sample_index;
    UInt rng_hash;
    UInt rng_offset;
    UInt source_object;
    UInt source_primitive;
};

class PathSurfaceAmbientOcclusionProvider final
    : public SurfaceAmbientOcclusionProvider {

private:
    std::shared_ptr<LuisaSceneData> _scene;
    std::shared_ptr<const SceneTraversalComponent> _traversal;
    PathSurfaceAmbientOcclusionContext _context;

public:
    PathSurfaceAmbientOcclusionProvider(
        std::shared_ptr<LuisaSceneData> scene,
        std::shared_ptr<const SceneTraversalComponent> traversal,
        const PathSurfaceAmbientOcclusionContext &context) noexcept;

    [[nodiscard]] Float evaluate(
        const SurfacePoint &point,
        const SurfaceAmbientOcclusionInput &input) const noexcept override;
};

}// namespace psycles::luisa_backend::detail
