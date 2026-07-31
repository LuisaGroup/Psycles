#include "path_kernel_volume_state.h"

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class PathVolumeStateComponentImpl final
    : public PathVolumeStateComponent {

  private:
    std::shared_ptr<const CameraVolumeStackComponent>
        _camera_stack;

  public:
    explicit PathVolumeStateComponentImpl(
        std::shared_ptr<
            const CameraVolumeStackComponent>
            camera_stack) noexcept
        : _camera_stack{std::move(camera_stack)} {}

    PathVolumeState initialize(
        const std::shared_ptr<LuisaSceneData> &scene,
        Expr<luisa::float3> camera_origin,
        Expr<std::uint32_t> visibility,
        std::size_t stack_size,
        bool probe_camera_enclosures)
        const noexcept override {
        auto stack =
            std::make_unique<VolumeStack>(
                stack_size);
        CameraVolumeStackInitialization
            camera_initialization{
                .intersection_count = 0u,
                .enclosed_count = 0u};
        if (probe_camera_enclosures) {
            camera_initialization =
                _camera_stack->initialize(
                    scene,
                    camera_origin,
                    visibility,
                    *stack);
        } else {
            stack->clear();
            _camera_stack
                ->initialize_background(
                    scene, *stack);
        }
        return {
            .stack = std::move(stack),
            .camera_initialization =
                std::move(
                    camera_initialization)};
    }
};

}// namespace

PathVolumeState
make_disabled_path_volume_state() noexcept {
    return {
        .stack = {},
        .camera_initialization = {
            .intersection_count = 0u,
            .enclosed_count = 0u}};
}

std::shared_ptr<const PathVolumeStateComponent>
make_path_volume_state_component() {
    return std::make_shared<
        PathVolumeStateComponentImpl>(
        make_camera_volume_stack_component());
}

}// namespace psycles::luisa_backend::detail
