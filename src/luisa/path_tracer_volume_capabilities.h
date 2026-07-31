#pragma once

#include <psycles/compiler/surface_program.h>

namespace psycles::luisa_backend::detail {

struct VolumeProgramCapabilities {
    bool has_spatial_values{};
    bool homogeneous{};
};

// Conservative, structural capability analysis. Homogeneous integration is
// enabled only when every value reachable from the raw volume-closure tree is
// independent of the shading position. This keeps unsupported heterogeneous
// graphs out of the production path without flattening or pre-baking them.
class VolumeProgramCapabilityComponent {

  public:
    [[nodiscard]] VolumeProgramCapabilities
    analyze(const compiler::SurfaceProgram &program) const noexcept;
};

}// namespace psycles::luisa_backend::detail
