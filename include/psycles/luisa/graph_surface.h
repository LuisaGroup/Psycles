#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/graph_surface.h> through the Psycles::luisa target."
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_color_nodes.h>
#include <psycles/luisa/cycles_noise.h>
#include <psycles/luisa/cycles_sample_mapping.h>
#include <psycles/luisa/cycles_volume.h>
#include <psycles/luisa/surface.h>

#include <luisa/core/stl/vector.h>

namespace psycles::luisa_backend {

class GraphSurface final : public Surface {

// The header-only Luisa DSL implementation is partitioned by renderer
// semantics. Each fragment is included in this class context so template
// callables remain visible without merging the responsibilities back into
// one monolithic source file.
#include <psycles/luisa/detail/graph_surface_state.inl>
#include <psycles/luisa/detail/graph_surface_scattering.inl>
#include <psycles/luisa/detail/graph_surface_values.inl>
#include <psycles/luisa/detail/graph_surface_closure_walk.inl>
#include <psycles/luisa/detail/graph_surface_api.inl>
};

}// namespace psycles::luisa_backend
