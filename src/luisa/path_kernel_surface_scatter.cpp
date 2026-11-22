#include "path_kernel_builder.h"
#include "path_kernel_cycles_svm_surface_scatter.h"

namespace psycles::luisa_backend::detail {
namespace {

class SurfaceScatterStageImpl final : public SurfaceScatterStage {

  public:
    Result emit(DirectLightingContext &context) const noexcept override {
        LUISA_ASSERT(
            context.shading.populated_surface != nullptr,
            "Native continuation requires the retained surface population.");
        const auto native =
            context.shading.populated_surface->native_surface_state();
        LUISA_ASSERT(native.has_value(),
                     "Missing native ShaderData consumer.");
        return emit_cycles_svm_surface_scatter(context, *native);
    }
};

} // namespace

std::unique_ptr<SurfaceScatterStage> make_surface_scatter_stage() {
    return std::make_unique<SurfaceScatterStageImpl>();
}

} // namespace psycles::luisa_backend::detail
