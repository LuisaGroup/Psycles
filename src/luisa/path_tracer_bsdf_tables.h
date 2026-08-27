#pragma once

#include <psycles/contract/scene.h>

#include <luisa/core/stl/vector.h>

namespace psycles::luisa_backend::detail {

// Builds the exact, versioned Cycles lookup-table buffer in device ABI order.
// Keeping payload ownership here prevents scene orchestration from depending
// on the generated table symbols or their textual implementation file.
[[nodiscard]] luisa::vector<float> make_cycles_bsdf_table_values(
    const contract::ShaderColorSpace &color_space);

}// namespace psycles::luisa_backend::detail
