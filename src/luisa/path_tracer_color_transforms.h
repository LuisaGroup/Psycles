#pragma once

#include "path_tracer_internal.h"

#include <optional>

namespace psycles::luisa_backend::detail {

using ShaderColorTransformCallable =
    Callable<luisa::float3(luisa::float3)>;

class CallableSurfaceColorTransformProvider final
    : public SurfaceColorTransformProvider {

private:
    // Callable ASTs are host-stage objects. Laziness ensures that an unused
    // transform is neither recorded nor attached before backend lowering.
    mutable std::optional<ShaderColorTransformCallable> _rgb_to_hsv;
    mutable std::optional<ShaderColorTransformCallable> _hsv_to_rgb;
    mutable std::optional<ShaderColorTransformCallable> _rgb_to_hsl;
    mutable std::optional<ShaderColorTransformCallable> _hsl_to_rgb;

public:
    CallableSurfaceColorTransformProvider() noexcept = default;

    [[nodiscard]] Float3 rgb_to_hsv(
        Float3 rgb) const noexcept override;

    [[nodiscard]] Float3 hsv_to_rgb(
        Float3 hsv) const noexcept override;

    [[nodiscard]] Float3 rgb_to_hsl(
        Float3 rgb) const noexcept override;

    [[nodiscard]] Float3 hsl_to_rgb(
        Float3 hsl) const noexcept override;
};

}// namespace psycles::luisa_backend::detail
