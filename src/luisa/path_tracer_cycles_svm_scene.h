#pragma once

#include "path_tracer_internal.h"

#include <memory>
#include <string>

namespace psycles::luisa_backend::detail {

[[nodiscard]] std::unique_ptr<CyclesSvmRuntime>
build_cycles_svm_runtime(const std::shared_ptr<LuisaSceneData> &scene,
                         const contract::SceneSnapshot &snapshot,
                         std::string &diagnostic);

void upload_cycles_svm_runtime(Stream &stream,
                               CyclesSvmRuntime &runtime) noexcept;

} // namespace psycles::luisa_backend::detail
