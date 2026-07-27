#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <psycles/contract/render.h>
#include <psycles/contract/scene.h>

namespace psycles::adapter {

enum class BlenderSceneDiagnosticSeverity : std::uint8_t {
    warning,
    error
};

struct BlenderSceneDiagnostic {
    BlenderSceneDiagnosticSeverity severity{
        BlenderSceneDiagnosticSeverity::error};
    std::string message;
};

struct BlenderSceneImport {
    std::optional<contract::SceneSnapshot> scene;
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t samples{};
    bool transparent_background{};
    contract::PixelFilter pixel_filter{
        contract::PixelFilter::box};
    float filter_width{1.0f};
    std::vector<BlenderSceneDiagnostic> diagnostics;

    [[nodiscard]] bool ok() const noexcept {
        if (!scene) {
            return false;
        }
        for (const auto &diagnostic : diagnostics) {
            if (diagnostic.severity ==
                BlenderSceneDiagnosticSeverity::error) {
                return false;
            }
        }
        return true;
    }
};

// Loads the deterministic bundle emitted by tools/export_psycles_scene.py.
// Shader nodes are normalized into Psycles' typed graph contract; no shader
// value is evaluated on the host.
[[nodiscard]] BlenderSceneImport load_blender_scene_bundle(
    const std::filesystem::path &directory);

}// namespace psycles::adapter
