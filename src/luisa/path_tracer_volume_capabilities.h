#pragma once

#include <psycles/compiler/surface_program.h>

#include <cstdint>
#include <vector>

namespace psycles::luisa_backend::detail {

inline constexpr std::uint32_t
    volume_surface_flag_heterogeneous = 1u << 0u;
inline constexpr std::uint32_t
    volume_surface_flag_light_path = 1u << 1u;

struct VolumeProgramCapabilities {
    bool has_spatial_values{};
    bool homogeneous{};
    // Current Cycles sets SD_HAS_LIGHT_PATH_NODE for any Light Path node in
    // the finalized shader, including a surface-only dependency. Preserve
    // that deliberately broad runtime-majorant trigger.
    bool has_light_path{};
};

// Conservative, structural capability analysis. Homogeneous integration is
// enabled only when every value reachable from the raw volume-closure tree is
// independent of the shading position. This keeps unsupported heterogeneous
// graphs out of the production path without flattening or pre-baking them.
class VolumeProgramCapabilityComponent {

  public:
    [[nodiscard]] VolumeProgramCapabilities
    analyze(const compiler::SurfaceProgram &program) const noexcept;

    // Merge one structural surface program into its device-dispatch slot.
    // Multiple materials may share a tag, so flags are deliberately ORed.
    void merge_surface_flags(
        std::vector<std::uint32_t> &flags,
        std::uint32_t surface_tag,
        const compiler::SurfaceProgram &program) const;
};

}// namespace psycles::luisa_backend::detail
