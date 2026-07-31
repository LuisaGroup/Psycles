#pragma once

#include "path_kernel_volume_boundary.h"

#include <cstddef>
#include <memory>

namespace psycles::luisa_backend::detail {

// Per-sample volume state. VolumeStack owns Luisa Local arrays and therefore
// cannot be copied or moved directly; the host pointer gives PathSampleContext
// ordinary move semantics while keeping the device-local storage alive for
// every stage that records operations against it.
struct PathVolumeState {
    std::unique_ptr<VolumeStack> stack;
    CameraVolumeStackInitialization camera_initialization;

    [[nodiscard]] bool enabled() const noexcept {
        return stack != nullptr;
    }
};

class PathVolumeStateComponent {

  public:
    virtual ~PathVolumeStateComponent() noexcept =
        default;

    [[nodiscard]] virtual PathVolumeState
    initialize(
        const std::shared_ptr<LuisaSceneData> &scene,
        Expr<luisa::float3> camera_origin,
        Expr<std::uint32_t> visibility,
        std::size_t stack_size,
        bool probe_camera_enclosures) const noexcept = 0;
};

// Host-specialized disabled state. It emits no Local arrays and no ray query,
// so volume-free kernels retain their previous device program.
[[nodiscard]]
PathVolumeState make_disabled_path_volume_state() noexcept;

[[nodiscard]]
std::shared_ptr<const PathVolumeStateComponent>
make_path_volume_state_component();

}// namespace psycles::luisa_backend::detail
