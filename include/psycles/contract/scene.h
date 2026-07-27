#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <psycles/contract/shader_graph.h>
#include <psycles/core/id.h>
#include <psycles/core/math.h>

namespace psycles::contract {

[[nodiscard]] constexpr std::uint64_t attribute_id(
    std::string_view name) noexcept {
    std::uint64_t hash = 14695981039346656037ull;
    for (const auto character : name) {
        hash ^= static_cast<std::uint8_t>(character);
        hash *= 1099511628211ull;
    }
    return hash;
}

struct MaterialTag;
struct ImageTag;
struct GeometryTag;
struct InstanceTag;
struct CameraTag;
struct LightTag;

using MaterialId = Id<MaterialTag>;
using ImageId = Id<ImageTag>;
using GeometryId = Id<GeometryTag>;
using InstanceId = Id<InstanceTag>;
using CameraId = Id<CameraTag>;
using LightId = Id<LightTag>;

struct MaterialDesc {
    std::string name;
    ShaderGraph shader;
};

enum class ImageColorSpace : std::uint8_t {
    data,
    linear,
    srgb
};

struct ImageDesc {
    std::string name;
    std::string source_format;
    ImageColorSpace color_space{ImageColorSpace::data};
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<std::uint8_t> encoded_data;
};

struct TriangleMeshDesc {
    std::string name;
    std::vector<Vec3f> positions;
    std::vector<Vec3f> normals;
    std::vector<Vec2f> uv;
    // Blender's evaluated MikkTSpace tangent attribute, flattened to the
    // same triangle-corner indexing as positions. xyz is the unnormalized
    // tangent and w is Cycles' tangent sign. A zero entry means the mesh has
    // no usable UV tangent attribute; tangent-space Normal Map then falls
    // back to the unperturbed shading normal exactly as Cycles does.
    std::vector<Vec4f> uv_tangents;
    // Cycles' Generated attribute is evaluated from the undeformed position
    // in Blender texture space. Keeping it explicit avoids reconstructing a
    // subtly different bounding-box mapping in a device backend.
    std::vector<Vec3f> generated;
    // Named vertex/corner attributes are flattened to the same triangle-
    // corner indexing as positions. Values are stored in scene-linear space;
    // BYTE_COLOR conversion therefore matches Cycles before device upload.
    std::map<std::string, std::vector<Vec4f>, std::less<>>
        color_attributes;
    std::vector<std::array<std::uint32_t, 3u>> triangles;
    std::vector<MaterialId> material_slots;
    std::vector<std::uint32_t> triangle_material_slots;
    // Face-domain ATTR_STD_RANDOM_PER_ISLAND values, already hashed with the
    // Cycles hash. This must not be interpolated across a triangle.
    std::vector<float> triangle_random_per_island;
};

struct MotionTransform {
    float time{};
    Mat4f transform;
};

enum class RayVisibility : std::uint32_t {
    camera = 1u << 0u,
    diffuse = 1u << 1u,
    glossy = 1u << 2u,
    transmission = 1u << 3u,
    shadow = 1u << 4u,
    volume_scatter = 1u << 5u
};

[[nodiscard]] constexpr std::uint32_t visibility_bit(
    RayVisibility visibility) noexcept {
    return static_cast<std::uint32_t>(visibility);
}

inline constexpr auto all_ray_visibility =
    visibility_bit(RayVisibility::camera) |
    visibility_bit(RayVisibility::diffuse) |
    visibility_bit(RayVisibility::glossy) |
    visibility_bit(RayVisibility::transmission) |
    visibility_bit(RayVisibility::shadow) |
    visibility_bit(RayVisibility::volume_scatter);

struct InstanceDesc {
    std::string name;
    GeometryId geometry;
    Mat4f transform;
    std::vector<MotionTransform> motion;
    std::vector<MaterialId> material_overrides;
    // Cycles Object Info.Random in [0, 1], computed from Object::random_id.
    float random{};
    // Cycles Particle Info indexes its particle table separately from object
    // identity. Zero is the non-particle sentinel used by ordinary objects.
    std::uint32_t particle_index{};
    std::uint32_t visibility_mask{all_ray_visibility};
};

enum class CameraProjection : std::uint8_t {
    perspective,
    orthographic,
    panorama
};

enum class CameraSensorFit : std::uint8_t {
    horizontal,
    vertical,
    automatic
};

struct CameraDesc {
    std::string name;
    CameraProjection projection{CameraProjection::perspective};
    Mat4f transform;
    // field_of_view is the vertical sensor angle. Blender also exposes a
    // horizontal sensor angle; keeping both avoids baking the source
    // render aspect ratio into the scene contract.
    float field_of_view{0.78539816339f};
    float horizontal_field_of_view{0.78539816339f};
    CameraSensorFit sensor_fit{CameraSensorFit::vertical};
    float orthographic_scale{1.0f};
    float lens_shift_x{};
    float lens_shift_y{};
    float near_clip{1.0e-4f};
    float far_clip{1.0e5f};
    float aperture_radius{};
    float focal_distance{};
    std::uint32_t aperture_blades{};
    float aperture_rotation{};
    float aperture_ratio{1.0f};
};

enum class LightType : std::uint8_t {
    point,
    spot,
    area,
    distant,
    background
};

struct LightDesc {
    std::string name;
    LightType type{LightType::point};
    Mat4f transform;
    Vec3f color{1.0f, 1.0f, 1.0f};
    float power{1.0f};
    float size{};
    float spread{3.14159265359f};
    std::optional<MaterialId> shader;
};

struct EnvironmentSunDesc {
    Vec3f direction{0.0f, 0.0f, 1.0f};
    // Disk-average radiance. The Cycles Nishita sun uses the radial profile
    // 0.4 + 0.6 * sqrt(1 - r^2), whose unit-disk average is 0.8.
    Vec3f radiance{};
    float angular_radius{};
};

struct EnvironmentDesc {
    std::string name;
    std::uint32_t width{};
    std::uint32_t height{};
    // Linear RGB, top-to-bottom equirectangular rows. The sun discs are
    // excluded and represented by `suns` so they can be sampled explicitly.
    std::vector<Vec3f> pixels;
    std::vector<EnvironmentSunDesc> suns;
};

struct SceneSnapshot {
    std::uint64_t revision{};
    std::map<MaterialId, MaterialDesc> materials;
    std::map<ImageId, ImageDesc> images;
    std::map<GeometryId, TriangleMeshDesc> geometries;
    std::map<InstanceId, InstanceDesc> instances;
    std::map<CameraId, CameraDesc> cameras;
    std::map<LightId, LightDesc> lights;
    std::optional<CameraId> active_camera;
    std::optional<MaterialId> world_shader;
    std::optional<EnvironmentDesc> environment;
};

struct UpsertMaterial {
    MaterialId id;
    MaterialDesc value;
};
struct RemoveMaterial {
    MaterialId id;
};
struct UpsertImage {
    ImageId id;
    ImageDesc value;
};
struct RemoveImage {
    ImageId id;
};
struct UpsertGeometry {
    GeometryId id;
    TriangleMeshDesc value;
};
struct RemoveGeometry {
    GeometryId id;
};
struct UpsertInstance {
    InstanceId id;
    InstanceDesc value;
};
struct RemoveInstance {
    InstanceId id;
};
struct UpsertCamera {
    CameraId id;
    CameraDesc value;
};
struct RemoveCamera {
    CameraId id;
};
struct SetActiveCamera {
    std::optional<CameraId> id;
};
struct UpsertLight {
    LightId id;
    LightDesc value;
};
struct RemoveLight {
    LightId id;
};
struct SetWorldShader {
    std::optional<MaterialId> id;
};
struct SetEnvironment {
    std::optional<EnvironmentDesc> value;
};

using SceneCommand = std::variant<
    UpsertMaterial,
    RemoveMaterial,
    UpsertImage,
    RemoveImage,
    UpsertGeometry,
    RemoveGeometry,
    UpsertInstance,
    RemoveInstance,
    UpsertCamera,
    RemoveCamera,
    SetActiveCamera,
    UpsertLight,
    RemoveLight,
    SetWorldShader,
    SetEnvironment>;

struct SceneDelta {
    std::uint64_t base_revision{};
    std::vector<SceneCommand> commands;

    template<typename Command, typename... Args>
    void emplace(Args &&...args) {
        commands.emplace_back(Command{std::forward<Args>(args)...});
    }
};

enum class SceneChange : std::uint64_t {
    none = 0u,
    materials = 1ull << 0u,
    geometry = 1ull << 1u,
    instances = 1ull << 2u,
    cameras = 1ull << 3u,
    lights = 1ull << 4u,
    world = 1ull << 5u,
    images = 1ull << 6u
};

[[nodiscard]] constexpr std::uint64_t change_bit(SceneChange change) noexcept {
    return static_cast<std::uint64_t>(change);
}

enum class SceneDiagnosticCode : std::uint8_t {
    stale_revision,
    invalid_id,
    invalid_mesh,
    invalid_reference,
    invalid_camera
};

struct SceneDiagnostic {
    SceneDiagnosticCode code{};
    std::string message;
};

struct SceneApplyResult {
    bool committed{false};
    std::uint64_t revision{};
    std::uint64_t changes{};
    std::vector<SceneDiagnostic> diagnostics;
};

class SceneDatabase {

private:
    SceneSnapshot _snapshot;

public:
    [[nodiscard]] const SceneSnapshot &snapshot() const noexcept { return _snapshot; }
    [[nodiscard]] SceneApplyResult apply(const SceneDelta &delta);
};

}// namespace psycles::contract
