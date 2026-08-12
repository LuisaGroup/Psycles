#pragma once

#include "path_kernel_builder.h"

namespace psycles::luisa_backend::detail {

// Native Luisa lowering of Cycles' volumetric BSSRDF random walk. The class is
// a host-side AST component: every input and result below remains a device
// expression, while C++ methods keep coefficient mapping, entry sampling, and
// spatial transport as independently testable semantic stages.
struct SubsurfaceRandomWalkCoefficients {
    Float3 sigma_t;
    Float3 alpha;
    Float3 sigma_s;
    Float3 throughput;
    Bool valid;
};

struct SubsurfaceRandomWalkEntry {
    Float3 direction;
    Bool valid;
};

class SubsurfaceRandomWalkComponent {

public:
    [[nodiscard]] SubsurfaceRandomWalkCoefficients coefficients(
        UInt method,
        Float3 albedo,
        Float3 radius,
        Float anisotropy,
        Float3 throughput) const noexcept;

    [[nodiscard]] SubsurfaceRandomWalkEntry sample_entry(
        const SurfacePoint &point,
        const SurfaceSample &closure,
        Float2 random) const noexcept;

    // Mutates only the path state owned by Cycles' INTERSECT_SUBSURFACE stage:
    // throughput, exact pending exit, synthetic exit ray, and the outer RNG
    // offset. A false result terminates the selected BSSRDF path.
    [[nodiscard]] Bool transport(
        PathSampleContext &path,
        const SubsurfaceTransportState &state) const noexcept;
};

}// namespace psycles::luisa_backend::detail
