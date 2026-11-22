#include "path_tracer_surfaces.h"

#include "path_tracer_cycles_svm_surface.h"

namespace psycles::luisa_backend::detail {

SurfaceCallables
make_surface_callables(const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    return {make_cycles_svm_surface_population_component(scene)};
}

}// namespace psycles::luisa_backend::detail
