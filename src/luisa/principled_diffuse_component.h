#pragma once

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

// Canonical inline transfer function shared by standalone GraphSurface and
// the path tracer's cross-topology typed callable.
[[nodiscard]] PrincipledDiffuseSetupResult
setup_principled_diffuse(
    const PrincipledDiffuseSetupInput &input) noexcept;

// Host-stage component selecting the production shared callable when the
// shader-service boundary supplies one. Authored values remain Luisa device
// expressions in either route.
class PrincipledDiffuseComponent final {

private:
    const ShaderServices &_services;

public:
    explicit PrincipledDiffuseComponent(
        const ShaderServices &services) noexcept;

    [[nodiscard]] PrincipledDiffuseSetupResult
    setup(const PrincipledDiffuseSetupInput &input) const noexcept;
};

}// namespace psycles::luisa_backend::detail
