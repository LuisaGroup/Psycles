#include "path_tracer_volume_metadata.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace psycles;
using namespace psycles::contract;
using namespace psycles::luisa_backend::detail;

[[nodiscard]] Mat4f transform(
    Vec3f translation,
    Vec3f scale = {1.0f, 1.0f, 1.0f})
    noexcept {
    Mat4f result;
    result.elements = {
        scale.x, 0.0f, 0.0f, 0.0f,
        0.0f, scale.y, 0.0f, 0.0f,
        0.0f, 0.0f, scale.z, 0.0f,
        translation.x,
        translation.y,
        translation.z,
        1.0f};
    return result;
}

[[nodiscard]] TriangleMeshDesc box(
    std::string_view name,
    MaterialId material,
    Vec3f minimum = {-1.0f, -1.0f, -1.0f},
    Vec3f maximum = {1.0f, 1.0f, 1.0f}) {
    return {
        .name = std::string{name},
        .positions = {
            minimum,
            maximum},
        .material_slots = {material}};
}

void add_instance(
    SceneSnapshot &scene,
    std::uint64_t id,
    GeometryId geometry,
    Mat4f matrix,
    std::vector<MaterialId> overrides = {}) {
    scene.instances.emplace(
        InstanceId{id},
        InstanceDesc{
            .name = "instance-" +
                    std::to_string(id),
            .geometry = geometry,
            .transform = matrix,
            .material_overrides =
                std::move(overrides)});
}

[[nodiscard]] bool close(
    float actual,
    float expected) noexcept {
    return std::abs(actual - expected) <=
           1.0e-6f;
}

}// namespace

int main() {
    constexpr MaterialId volume{1u};
    constexpr MaterialId surface{2u};
    constexpr MaterialId world{3u};
    const std::set volume_materials{
        volume, world};
    VolumeSceneMetadataComponent component;
    auto failed = false;
    const auto require =
        [&](bool condition,
            std::string_view message) {
            if (!condition) {
                failed = true;
                std::cerr << message << '\n';
            }
        };

    SceneSnapshot scene;
    scene.world_shader = world;
    scene.geometries.emplace(
        GeometryId{1u},
        box("volume", volume));
    scene.geometries.emplace(
        GeometryId{2u},
        box("surface", surface));
    add_instance(
        scene,
        1u,
        GeometryId{1u},
        transform({0.0f, 0.0f, 0.0f}));
    add_instance(
        scene,
        2u,
        GeometryId{1u},
        transform({1.5f, 0.0f, 0.0f}));
    add_instance(
        scene,
        3u,
        GeometryId{1u},
        transform({10.0f, 0.0f, 0.0f}));
    // A non-volume override must remove the geometry's volume capability.
    add_instance(
        scene,
        4u,
        GeometryId{1u},
        transform({20.0f, 0.0f, 0.0f}),
        {surface});
    // Conversely, a volume override makes a surface geometry a boundary.
    add_instance(
        scene,
        5u,
        GeometryId{2u},
        transform(
            {10.5f, 0.0f, 0.0f},
            {-2.0f, 1.0f, 0.5f}),
        {volume});

    const auto metadata =
        component.analyze(
            scene, volume_materials);
    require(
        metadata.has_world_volume,
        "world volume was not retained");
    require(
        metadata.objects.size() == 4u,
        "effective instance overrides were not respected");
    require(
        metadata.stack_size == 6u,
        "Cycles intersecting-volume stack estimate changed");
    require(
        metadata.objects[0u].intersects_volume &&
            metadata.objects[1u].intersects_volume &&
            metadata.objects[2u].intersects_volume &&
            metadata.objects[3u].intersects_volume,
        "pairwise volume overlap was not recorded");
    const auto &reflected =
        metadata.objects[3u].bounds;
    require(
        close(reflected.minimum.x, 8.5f) &&
            close(reflected.maximum.x, 12.5f) &&
            close(reflected.minimum.z, -0.5f) &&
            close(reflected.maximum.z, 0.5f),
        "negative-scale world bounds changed");

    SceneSnapshot disjoint;
    disjoint.geometries.emplace(
        GeometryId{1u},
        box("volume", volume));
    add_instance(
        disjoint,
        1u,
        GeometryId{1u},
        transform({-10.0f, 0.0f, 0.0f}));
    add_instance(
        disjoint,
        2u,
        GeometryId{1u},
        transform({10.0f, 0.0f, 0.0f}));
    const auto disjoint_metadata =
        component.analyze(
            disjoint, volume_materials);
    require(
        disjoint_metadata.stack_size == 3u,
        "disjoint volumes should share one conservative object slot");
    require(
        !disjoint_metadata.objects[0u]
             .intersects_volume &&
            !disjoint_metadata.objects[1u]
                 .intersects_volume,
        "disjoint volumes were classified as intersecting");

    SceneSnapshot world_only;
    world_only.world_shader = world;
    const auto world_metadata =
        component.analyze(
            world_only, volume_materials);
    require(
        world_metadata.stack_size == 2u &&
            world_metadata.objects.empty(),
        "world-only stack must contain background and terminator");
    require(
        component
                .analyze(
                    SceneSnapshot{},
                    volume_materials)
                .stack_size == 0u,
        "volume-free scene did not select the host specialization");

    SceneSnapshot saturated;
    saturated.geometries.emplace(
        GeometryId{1u},
        box("volume", volume));
    for (auto index = std::uint64_t{0u};
         index < 40u;
         ++index) {
        add_instance(
            saturated,
            index + 1u,
            GeometryId{1u},
            transform({0.0f, 0.0f, 0.0f}));
    }
    require(
        component
                .analyze(
                    saturated,
                    volume_materials)
                .stack_size == 32u,
        "volume stack did not clamp to Cycles' 32 slots");

    VolumeSceneMetadata camera_metadata;
    camera_metadata.objects.emplace_back(
        VolumeObjectMetadata{
            .instance = InstanceId{1u},
            .bounds = {
                .minimum =
                    {0.19f, -0.01f, -0.11f},
                .maximum =
                    {0.21f, 0.01f, -0.09f},
                .valid = true}});
    CameraVolumeBoundsQuery camera{
        .projection =
            CameraProjection::perspective,
        .transform = transform(
            {0.0f, 0.0f, 0.0f}),
        .aspect = 1.0f,
        .horizontal_tangent = 1.0f,
        .vertical_tangent = 1.0f,
        .orthographic_scale = 2.0f,
        .near_clip = 0.1f,
        .focal_distance = 1.0f,
        .aperture_ratio = 1.0f};
    require(
        component.camera_may_be_inside_volume(
            camera_metadata, camera),
        "perspective near-plane extent missed a volume");
    camera.transform =
        transform({100.0f, 0.0f, 0.0f});
    require(
        !component.camera_may_be_inside_volume(
            camera_metadata, camera),
        "translated camera produced a false overlap");

    camera.transform =
        transform({0.0f, 0.0f, 0.0f});
    camera.projection =
        CameraProjection::orthographic;
    camera.aspect = 2.0f;
    require(
        component.camera_may_be_inside_volume(
            camera_metadata, camera),
        "orthographic view-plane extent missed a volume");

    camera.projection =
        CameraProjection::panorama;
    camera.near_clip = 0.2f;
    require(
        component.camera_may_be_inside_volume(
            camera_metadata, camera),
        "panorama near-clip sphere missed a volume");
    camera.transform =
        transform({100.0f, 0.0f, 0.0f});
    require(
        !component.camera_may_be_inside_volume(
            camera_metadata, camera),
        "translated panorama produced a false overlap");

    VolumeSceneMetadata aperture_metadata;
    aperture_metadata.objects.emplace_back(
        VolumeObjectMetadata{
            .instance = InstanceId{1u},
            .bounds = {
                .minimum =
                    {0.55f, -0.01f, -0.11f},
                .maximum =
                    {0.56f, 0.01f, -0.09f},
                .valid = true}});
    camera = CameraVolumeBoundsQuery{
        .projection =
            CameraProjection::perspective,
        .transform = transform(
            {0.0f, 0.0f, 0.0f}),
        .aspect = 1.0f,
        .horizontal_tangent = 1.0f,
        .vertical_tangent = 1.0f,
        .near_clip = 0.1f,
        .aperture_radius = 0.2f,
        .focal_distance = 1.0f,
        .aperture_ratio = 0.5f};
    require(
        component.camera_may_be_inside_volume(
            aperture_metadata, camera),
        "Cycles aperture extension was not applied");
    camera.aperture_radius = 0.0f;
    require(
        !component.camera_may_be_inside_volume(
            aperture_metadata, camera),
        "zero aperture retained a stale camera overlap");

    return failed ? EXIT_FAILURE :
                    EXIT_SUCCESS;
}
