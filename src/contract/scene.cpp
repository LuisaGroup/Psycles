#include <psycles/contract/scene.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace psycles::contract {

namespace {

template<typename Id>
[[nodiscard]] bool valid_id(Id id) noexcept {
    return id.valid();
}

[[nodiscard]] std::vector<SceneDiagnostic> validate_scene(
    const SceneSnapshot &scene) {
    std::vector<SceneDiagnostic> diagnostics;
    auto diagnose = [&](SceneDiagnosticCode code, std::string message) {
        diagnostics.emplace_back(SceneDiagnostic{
            .code = code,
            .message = std::move(message)});
    };

    for (const auto &[id, _] : scene.materials) {
        if (!valid_id(id)) {
            diagnose(
                SceneDiagnosticCode::invalid_id,
                "scene contains an invalid material identifier");
        }
    }

    for (const auto &[id, image] : scene.images) {
        if (!valid_id(id)) {
            diagnose(
                SceneDiagnosticCode::invalid_id,
                "scene contains an invalid image identifier");
        }
        if (image.width == 0u || image.height == 0u ||
            image.encoded_data.empty()) {
            diagnose(
                SceneDiagnosticCode::invalid_reference,
                "image '" + image.name +
                    "' has no encoded pixels or dimensions");
        }
    }

    for (const auto &[id, mesh] : scene.geometries) {
        if (!valid_id(id)) {
            diagnose(
                SceneDiagnosticCode::invalid_id,
                "scene contains an invalid geometry identifier");
        }
        if (mesh.positions.empty() || mesh.triangles.empty()) {
            diagnose(
                SceneDiagnosticCode::invalid_mesh,
                "mesh '" + mesh.name + "' has no positions or triangles");
        }
        if (!mesh.normals.empty() &&
            mesh.normals.size() != mesh.positions.size()) {
            diagnose(
                SceneDiagnosticCode::invalid_mesh,
                "mesh '" + mesh.name + "' has a mismatched normal count");
        }
        if (!mesh.uv.empty() && mesh.uv.size() != mesh.positions.size()) {
            diagnose(
                SceneDiagnosticCode::invalid_mesh,
                "mesh '" + mesh.name + "' has a mismatched UV count");
        }
        if (!mesh.generated.empty() &&
            mesh.generated.size() != mesh.positions.size()) {
            diagnose(
                SceneDiagnosticCode::invalid_mesh,
                "mesh '" + mesh.name +
                    "' has a mismatched Generated attribute count");
        }
        for (const auto &[name, values] :
             mesh.color_attributes) {
            if (name.empty() ||
                values.size() != mesh.positions.size()) {
                diagnose(
                    SceneDiagnosticCode::invalid_mesh,
                    "mesh '" + mesh.name +
                        "' has an invalid color attribute '" +
                        name + "'");
            }
        }
        for (const auto &triangle : mesh.triangles) {
            if (std::any_of(
                    triangle.begin(), triangle.end(),
                    [&](auto index) {
                        return static_cast<std::size_t>(index) >=
                               mesh.positions.size();
                    })) {
                diagnose(
                    SceneDiagnosticCode::invalid_mesh,
                    "mesh '" + mesh.name +
                        "' contains an out-of-range triangle index");
                break;
            }
        }
        for (auto material : mesh.material_slots) {
            if (!scene.materials.contains(material)) {
                diagnose(
                    SceneDiagnosticCode::invalid_reference,
                    "mesh '" + mesh.name +
                        "' references a missing material");
            }
        }
        if (!mesh.triangle_material_slots.empty()) {
            if (mesh.triangle_material_slots.size() != mesh.triangles.size()) {
                diagnose(
                    SceneDiagnosticCode::invalid_mesh,
                    "mesh '" + mesh.name +
                        "' has a mismatched triangle material count");
            } else if (std::any_of(
                           mesh.triangle_material_slots.begin(),
                           mesh.triangle_material_slots.end(),
                           [&](auto slot) {
                               return static_cast<std::size_t>(slot) >=
                                      mesh.material_slots.size();
                           })) {
                diagnose(
                    SceneDiagnosticCode::invalid_mesh,
                    "mesh '" + mesh.name +
                        "' contains an out-of-range material slot");
            }
        }
        if (!mesh.triangle_random_per_island.empty() &&
            mesh.triangle_random_per_island.size() !=
                mesh.triangles.size()) {
            diagnose(
                SceneDiagnosticCode::invalid_mesh,
                "mesh '" + mesh.name +
                    "' has a mismatched Random Per Island count");
        }
    }

    for (const auto &[id, instance] : scene.instances) {
        if (!valid_id(id)) {
            diagnose(
                SceneDiagnosticCode::invalid_id,
                "scene contains an invalid instance identifier");
        }
        if (!scene.geometries.contains(instance.geometry)) {
            diagnose(
                SceneDiagnosticCode::invalid_reference,
                "instance '" + instance.name +
                    "' references missing geometry");
        }
        for (auto material : instance.material_overrides) {
            if (!scene.materials.contains(material)) {
                diagnose(
                    SceneDiagnosticCode::invalid_reference,
                    "instance '" + instance.name +
                        "' references a missing material override");
            }
        }
    }

    for (const auto &[id, camera] : scene.cameras) {
        if (!valid_id(id)) {
            diagnose(
                SceneDiagnosticCode::invalid_id,
                "scene contains an invalid camera identifier");
        }
        if (camera.near_clip <= 0.0f ||
            camera.far_clip <= camera.near_clip) {
            diagnose(
                SceneDiagnosticCode::invalid_camera,
                "camera '" + camera.name + "' has an invalid clip range");
        }
    }

    if (scene.active_camera &&
        !scene.cameras.contains(*scene.active_camera)) {
        diagnose(
            SceneDiagnosticCode::invalid_reference,
            "active camera references a missing camera");
    }

    for (const auto &[id, light] : scene.lights) {
        if (!valid_id(id)) {
            diagnose(
                SceneDiagnosticCode::invalid_id,
                "scene contains an invalid light identifier");
        }
        if (light.shader && !scene.materials.contains(*light.shader)) {
            diagnose(
                SceneDiagnosticCode::invalid_reference,
                "light '" + light.name +
                    "' references a missing shader");
        }
        if (!std::isfinite(light.color.x) ||
            !std::isfinite(light.color.y) ||
            !std::isfinite(light.color.z) ||
            !std::isfinite(light.power) ||
            !std::isfinite(light.size) ||
            !std::isfinite(light.size_y) ||
            !std::isfinite(light.spread) ||
            !std::isfinite(light.spot_angle) ||
            !std::isfinite(light.spot_smooth) ||
            !std::isfinite(light.angle) ||
            light.size < 0.0f ||
            light.size_y < 0.0f ||
            light.spread < 0.0f ||
            light.spot_angle < 0.0f ||
            light.spot_smooth < 0.0f ||
            light.angle < 0.0f) {
            diagnose(
                SceneDiagnosticCode::invalid_reference,
                "light '" + light.name +
                    "' contains invalid radiometric or shape parameters");
        }
    }

    if (scene.world_shader &&
        !scene.materials.contains(*scene.world_shader)) {
        diagnose(
            SceneDiagnosticCode::invalid_reference,
            "world references a missing shader");
    }
    if (scene.environment) {
        const auto &environment = *scene.environment;
        if (environment.nishita) {
            const auto &sky = *environment.nishita;
            if (!std::isfinite(sky.sun_elevation) ||
                !std::isfinite(sky.sun_rotation) ||
                !std::isfinite(sky.angular_diameter) ||
                !std::isfinite(sky.sun_intensity) ||
                !std::isfinite(sky.altitude) ||
                !std::isfinite(sky.air_density) ||
                !std::isfinite(sky.dust_density) ||
                !std::isfinite(sky.ozone_density) ||
                !std::isfinite(sky.background_strength)) {
                diagnose(
                    SceneDiagnosticCode::invalid_reference,
                    "world Nishita environment contains non-finite parameters");
            }
        } else {
            const auto expected =
                static_cast<std::size_t>(environment.width) *
                static_cast<std::size_t>(environment.height);
            if (environment.width == 0u ||
                environment.height == 0u ||
                environment.pixels.size() != expected) {
                diagnose(
                    SceneDiagnosticCode::invalid_reference,
                    "world environment has invalid dimensions or pixel count");
            }
        }
        for (const auto &sun : environment.suns) {
            if (sun.angular_radius <= 0.0f) {
                diagnose(
                    SceneDiagnosticCode::invalid_reference,
                    "world environment contains an invalid sun cone");
            }
        }
    }

    return diagnostics;
}

}// namespace

SceneApplyResult SceneDatabase::apply(const SceneDelta &delta) {
    if (delta.base_revision != _snapshot.revision) {
        return {
            .committed = false,
            .revision = _snapshot.revision,
            .changes = 0u,
            .diagnostics = {{
                .code = SceneDiagnosticCode::stale_revision,
                .message = "scene delta base revision is stale"}}};
    }

    auto candidate = _snapshot;
    std::uint64_t changes{};

    for (const auto &command : delta.commands) {
        std::visit(
            [&](const auto &operation) {
                using T = std::decay_t<decltype(operation)>;
                if constexpr (std::is_same_v<T, UpsertMaterial>) {
                    candidate.materials.insert_or_assign(
                        operation.id, operation.value);
                    changes |= change_bit(SceneChange::materials);
                } else if constexpr (std::is_same_v<T, RemoveMaterial>) {
                    candidate.materials.erase(operation.id);
                    changes |= change_bit(SceneChange::materials);
                } else if constexpr (std::is_same_v<T, UpsertImage>) {
                    candidate.images.insert_or_assign(
                        operation.id, operation.value);
                    changes |= change_bit(SceneChange::images);
                } else if constexpr (std::is_same_v<T, RemoveImage>) {
                    candidate.images.erase(operation.id);
                    changes |= change_bit(SceneChange::images);
                } else if constexpr (std::is_same_v<T, UpsertGeometry>) {
                    candidate.geometries.insert_or_assign(
                        operation.id, operation.value);
                    changes |= change_bit(SceneChange::geometry);
                } else if constexpr (std::is_same_v<T, RemoveGeometry>) {
                    candidate.geometries.erase(operation.id);
                    changes |= change_bit(SceneChange::geometry);
                } else if constexpr (std::is_same_v<T, UpsertInstance>) {
                    candidate.instances.insert_or_assign(
                        operation.id, operation.value);
                    changes |= change_bit(SceneChange::instances);
                } else if constexpr (std::is_same_v<T, RemoveInstance>) {
                    candidate.instances.erase(operation.id);
                    changes |= change_bit(SceneChange::instances);
                } else if constexpr (std::is_same_v<T, UpsertCamera>) {
                    candidate.cameras.insert_or_assign(
                        operation.id, operation.value);
                    changes |= change_bit(SceneChange::cameras);
                } else if constexpr (std::is_same_v<T, RemoveCamera>) {
                    candidate.cameras.erase(operation.id);
                    changes |= change_bit(SceneChange::cameras);
                } else if constexpr (std::is_same_v<T, SetActiveCamera>) {
                    candidate.active_camera = operation.id;
                    changes |= change_bit(SceneChange::cameras);
                } else if constexpr (std::is_same_v<T, UpsertLight>) {
                    candidate.lights.insert_or_assign(
                        operation.id, operation.value);
                    changes |= change_bit(SceneChange::lights);
                } else if constexpr (std::is_same_v<T, RemoveLight>) {
                    candidate.lights.erase(operation.id);
                    changes |= change_bit(SceneChange::lights);
                } else if constexpr (std::is_same_v<T, SetWorldShader>) {
                    candidate.world_shader = operation.id;
                    changes |= change_bit(SceneChange::world);
                } else if constexpr (
                    std::is_same_v<T, SetEnvironment>) {
                    candidate.environment = operation.value;
                    changes |= change_bit(SceneChange::world);
                }
            },
            command);
    }

    auto diagnostics = validate_scene(candidate);
    if (!diagnostics.empty()) {
        return {
            .committed = false,
            .revision = _snapshot.revision,
            .changes = 0u,
            .diagnostics = std::move(diagnostics)};
    }

    ++candidate.revision;
    _snapshot = std::move(candidate);
    return {
        .committed = true,
        .revision = _snapshot.revision,
        .changes = changes,
        .diagnostics = {}};
}

}// namespace psycles::contract
