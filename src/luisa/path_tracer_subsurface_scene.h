#pragma once

#include "path_tracer_internal.h"

#include <set>
#include <vector>

namespace psycles::luisa_backend::detail {

// Host/JIT plan for the conservative BSSRDF local-intersection domain. The
// mask is indexed by the same stable primary-instance ordinal used by
// InstanceGpu and the primary TLAS; it therefore provides an explicit
// bijection instead of relying on secondary-TLAS insertion order.
struct SubsurfaceScenePlan {
    std::vector<std::uint8_t> triangle_instance_mask;
    std::uint32_t triangle_instance_count{};

    [[nodiscard]] bool contains(
        std::uint32_t primary_instance) const noexcept;
};

class SubsurfaceSceneComponent final {

public:
    [[nodiscard]] SubsurfaceScenePlan plan(
        const contract::SceneSnapshot &snapshot,
        const std::set<contract::MaterialId> &
            bssrdf_materials) const;

    void initialize_accel(
        const std::shared_ptr<LuisaSceneData> &scene,
        const SubsurfaceScenePlan &plan) const;

    void append_triangle_instance(
        const std::shared_ptr<LuisaSceneData> &scene,
        const SubsurfaceScenePlan &plan,
        std::uint32_t primary_instance,
        std::uint32_t geometry_index,
        const Mat4f &transform) const;
};

}// namespace psycles::luisa_backend::detail
