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

// Cycles exposes Pointiness as a standard geometry attribute rather than a
// user-named attribute. Keep its binding in a reserved namespace so a custom
// attribute called "pointiness" cannot alias it.
inline constexpr auto cycles_pointiness_attribute_id =
    attribute_id("geom:pointiness");

[[nodiscard]] inline std::uint64_t uv_attribute_id(
    std::string_view name) {
    std::string qualified{"geom:uv:"};
    qualified.append(name);
    return attribute_id(qualified);
}

[[nodiscard]] inline std::uint64_t uv_tangent_attribute_id(
    std::string_view name) {
    std::string qualified{"geom:tangent:"};
    qualified.append(name);
    return attribute_id(qualified);
}

[[nodiscard]] inline std::uint64_t
uv_undisplaced_tangent_attribute_id(std::string_view name) {
    std::string qualified{"geom:undisplaced_tangent:"};
    qualified.append(name);
    return attribute_id(qualified);
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

enum class EmissionSampling : std::uint8_t {
    none,
    automatic,
    front,
    back,
    front_back
};

// Cycles represents volume direct-light sampling as two independent
// techniques. Keeping the same bit algebra makes a stack of authored media
// composable: mixing distance and equiangular necessarily yields MIS.
enum class VolumeSampling : std::uint8_t {
    distance = 1u << 0u,
    equiangular = 1u << 1u,
    multiple_importance =
        (1u << 0u) | (1u << 1u)
};

// Blender/Cycles material displacement policy. This is deliberately an enum
// rather than a pair of booleans: the three authored states have distinct
// shader-evaluation and geometry-rebuild semantics.
enum class DisplacementMethod : std::uint8_t {
    bump,
    displacement,
    both
};

[[nodiscard]] constexpr bool uses_true_displacement(
    DisplacementMethod method) noexcept {
    return method == DisplacementMethod::displacement ||
           method == DisplacementMethod::both;
}

[[nodiscard]] constexpr bool uses_displacement_bump(
    DisplacementMethod method) noexcept {
    return method == DisplacementMethod::bump ||
           method == DisplacementMethod::both;
}

struct MaterialDesc {
    std::string name;
    ShaderGraph shader;
    // Cycles Shader::use_bump_map_correction. This remains material
    // metadata beside the raw closure graph because structurally identical
    // graphs may use different correction policies at runtime.
    bool use_bump_map_correction{true};
    // This is the original Cycles material setting. It controls whether and
    // from which side an emissive closure participates in light sampling; it
    // never replaces or pre-evaluates the closure graph above.
    EmissionSampling emission_sampling{
        EmissionSampling::automatic};
    // The original Cycles volume-sampling policy. Blender's current default
    // is Multiple Importance; this remains metadata beside the raw closure
    // graph and is interpreted only when the material is active in a volume
    // stack.
    VolumeSampling volume_sampling{
        VolumeSampling::multiple_importance};
    // Cycles keeps geometry with true displacement in object space even when
    // it has a single user. The raw graph is retained above; this enum owns
    // the exact authored BUMP/DISPLACEMENT/BOTH policy.
    DisplacementMethod displacement_method{
        DisplacementMethod::bump};
    // Index in Cycles' Scene::shaders vector, before per-hit shader flags
    // such as SHADER_CAST_SHADOW and SHADER_SMOOTH_NORMAL are applied.
    // This source identity is optional for programmatically built scenes;
    // Blender bundles exported for differential validation always carry it.
    std::optional<std::uint32_t> cycles_shader_index;
};

enum class ImageColorSpace : std::uint8_t {
    data,
    linear,
    srgb
};

enum class ImageAlphaType : std::uint8_t {
    straight,
    premultiplied,
    channel_packed,
    ignore
};

struct ImageDesc {
    std::string name;
    std::string source_format;
    ImageColorSpace color_space{ImageColorSpace::data};
    ImageAlphaType alpha_type{ImageAlphaType::straight};
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<std::uint8_t> encoded_data;
};

enum class MeshAttributeDomain : std::uint8_t {
    point,
    corner,
    face
};

template<typename T>
struct MeshAttribute {
    MeshAttributeDomain domain{MeshAttributeDomain::point};
    std::vector<T> values;
};

// Source topology retained for Cycles' mesh-sync Pointiness construction.
// These are the evaluated Blender point normals and original (pre-triangle-
// tessellation) edges. Pointiness is derived from them by Psycles at scene
// synchronization time; this is geometry preprocessing, never shader baking.
struct MeshPointinessSource {
    std::vector<Vec3f> point_normals;
    std::vector<std::array<std::uint32_t, 2u>> edges;
};

struct TriangleMeshDesc {
    std::string name;
    std::vector<Vec3f> positions;
    MeshAttribute<Vec3f> normals;
    MeshAttribute<Vec2f> uv;
    // `uv` remains a total shader-coordinate field and may therefore contain
    // zero placeholders. This optional distinguishes those placeholders from
    // a real standard UV layer. Programmatic scenes that omit it preserve the
    // historical inference from whether `uv` contains values.
    std::optional<bool> default_uv_available;
    // Serialized Blender MikkTSpace tangent input. The Luisa scene stage
    // regenerates corner frames with the aligned Cycles Mikk implementation
    // so true displacement can maintain separate current and ORIGINAL
    // attributes. When no real default UV exists, Cycles' spherical-position
    // fallback is used instead of mistaking the total zero UV buffer for a
    // genuine map. xyz is the tangent and w is the orientation sign.
    MeshAttribute<Vec4f> uv_tangents;
    // Every evaluated Blender UV layer is retained by name. The active layer
    // is also present in `uv`/`uv_tangents` for the unnamed Cycles contract.
    // Named UV Map and Normal Map nodes select these immutable attributes by
    // their structure-time layer name.
    std::map<std::string, MeshAttribute<Vec2f>, std::less<>>
        uv_layers;
    std::map<std::string, MeshAttribute<Vec4f>, std::less<>>
        uv_tangent_layers;
    // Cycles' Generated attribute is evaluated from the undeformed position
    // in Blender texture space. Keeping it explicit avoids reconstructing a
    // subtly different bounding-box mapping in a device backend.
    MeshAttribute<Vec3f> generated;
    // Volume shaders do not have a surface primitive from which Generated
    // can be interpolated. Cycles instead transforms the object-space
    // shading position with ATTR_STD_GENERATED_TRANSFORM. Blender bundles
    // preserve that affine transform explicitly; programmatic scenes may
    // omit it and let the runtime derive the ordinary bounds mapping.
    std::optional<Mat4f> generated_transform;
    // Named vertex/corner values are stored in scene-linear space;
    // BYTE_COLOR conversion therefore matches Cycles before device upload.
    std::map<std::string, MeshAttribute<Vec4f>, std::less<>>
        color_attributes;
    std::optional<MeshPointinessSource> pointiness_source;
    std::vector<std::array<std::uint32_t, 3u>> triangles;
    std::vector<MaterialId> material_slots;
    std::vector<std::uint32_t> triangle_material_slots;
    // Cycles' SHADER_SMOOTH_NORMAL flag is a per-triangle topology
    // property. It gates shadow-terminator geometry offset and cannot be
    // reconstructed reliably from corner-normal values.
    std::vector<std::uint8_t> triangle_smooth;
    // Face-domain ATTR_STD_RANDOM_PER_ISLAND values, already hashed with the
    // Cycles hash. This must not be interpolated across a triangle.
    std::vector<float> triangle_random_per_island;
    // First triangle in Cycles' global primitive array. Blender bundles carry
    // the exact GeometryManager prefix; renderer-authored scenes may omit it
    // and let the backend assign a deterministic geometry-order prefix.
    std::optional<std::uint32_t> cycles_primitive_offset;
    // Adaptive subdivision meshes are never eligible for Cycles' static
    // object-transform application. This is source geometry metadata, not a
    // tessellation or baking result.
    bool uses_adaptive_subdivision{};
};

// Cycles keeps legacy particle hair as an independent Geometry rather than
// tessellating it into the emitter mesh. The control points and radii below
// are therefore source geometry, not a render-time mesh approximation. A
// renderer backend is responsible for applying the selected Cycles curve
// intersection model to every adjacent key pair.
enum class CurveShape : std::uint8_t {
    ribbon,
    thick,
    thick_linear
};

struct CurveGeometryDesc {
    std::string name;
    CurveShape shape{CurveShape::ribbon};
    // Cycles subdivides each Catmull-Rom key interval into 2^subdivisions
    // intersection intervals, clamped by its scene parameter range.
    std::uint32_t subdivisions{2u};
    // xyz is the object-space key position and w is the object-space radius.
    std::vector<Vec4f> keys;
    // One entry per curve. The end of a curve is the next entry, or
    // keys.size() for the final curve.
    std::vector<std::uint32_t> curve_first_key;
    std::vector<MaterialId> material_slots;
    std::vector<std::uint32_t> curve_material_slots;
    // Cycles Hair Info attributes retain their native domains: Intercept is
    // per key, while Length and Random are per curve.
    std::vector<float> intercept;
    std::vector<float> length;
    std::vector<float> random;
    // Curves and curve segments occupy independent Cycles primitive spaces;
    // neither offset may be reconstructed from triangle primitive offsets.
    std::optional<std::uint32_t> cycles_curve_offset;
    std::optional<std::uint32_t> cycles_segment_offset;
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
    // Cycles Object::shadow_terminator_geometry_offset. This is evaluated per
    // instance because objects sharing a mesh may author different values.
    float shadow_terminator_geometry_offset{};
    std::uint32_t visibility_mask{all_ray_visibility};
    // Stable indices produced by BlenderSync/ObjectManager and Film's
    // light-group map. They are source-scene identities, not Psycles map or
    // TLAS indices.
    std::optional<std::uint32_t> cycles_object_index;
    std::int32_t cycles_light_group{-1};
    // Cycles folds shadow-catcher membership into sampled-emitter shader
    // identity. Keep it per instance because shared geometry may be instanced
    // both as a catcher and as an ordinary object.
    bool is_shadow_catcher{};
    // Preserve whether this entry came from a dependency-graph dupli. Cycles'
    // static-transform decision is based on geometry users rather than this
    // bit, but keeping the source representation prevents later adapters from
    // having to infer it from names or persistent IDs.
    bool is_blender_instance{};
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
    float size_y{};
    float spread{3.14159265359f};
    float spot_angle{0.78539816339f};
    float spot_smooth{0.15f};
    float angle{};
    bool normalize{true};
    bool ellipse{};
    bool is_sphere{true};
    // Cycles area-light portal. Portals are environment sampling geometry,
    // never ordinary analytic emitters.
    bool is_portal{};
    std::optional<MaterialId> shader;
    bool use_mis{true};
    bool cast_shadow{true};
    std::uint32_t visibility_mask{all_ray_visibility};
    bool is_shadow_catcher{};
    std::optional<std::uint32_t> cycles_shader_index;
    std::optional<std::uint32_t> cycles_object_index;
    std::int32_t cycles_light_group{-1};
    // CyclesLightSettings::max_bounces. This limits NEE selection of lamp
    // emitters; it does not hide a lamp reached by forward path tracing.
    std::uint32_t max_bounces{1024u};
};

struct EnvironmentSunDesc {
    Vec3f direction{0.0f, 0.0f, 1.0f};
    // Disk-average radiance. The Cycles Nishita sun uses the radial profile
    // 0.4 + 0.6 * sqrt(1 - r^2), whose unit-disk average is 0.8.
    Vec3f radiance{};
    float angular_radius{};
};

struct NishitaSkyDesc {
    // Cycles' simplified node parameters. `sun_rotation` already follows the
    // Cycles device convention after Blender's node-space wrap.
    float sun_elevation{};
    float sun_rotation{};
    float angular_diameter{-1.0f};
    float sun_intensity{1.0f};
    float altitude{};
    float air_density{1.0f};
    float dust_density{1.0f};
    float ozone_density{1.0f};
    float background_strength{1.0f};
};

struct EnvironmentDesc {
    std::string name;
    std::uint32_t width{};
    std::uint32_t height{};
    // Linear RGB, top-to-bottom equirectangular rows. The sun discs are
    // excluded and represented by `suns` so they can be sampled explicitly.
    std::vector<Vec3f> pixels;
    std::vector<EnvironmentSunDesc> suns;
    // Procedural Cycles Nishita data is compiled into a Luisa precompute
    // kernel. It is intentionally not a Blender/Cycles-baked image.
    std::optional<NishitaSkyDesc> nishita;
};

enum class WorldSampling : std::uint8_t {
    none,
    automatic,
    manual
};

// Cycles derives these transforms from the active OCIO configuration and
// uploads them with film data. Shader nodes such as Blackbody and Wavelength
// must use the same scene-linear working space as the rest of the graph.
struct ShaderColorSpace {
    Vec3f xyz_to_r{3.2404542f, -1.5371385f, -0.4985314f};
    Vec3f xyz_to_g{-0.9692660f, 1.8760108f, 0.0415560f};
    Vec3f xyz_to_b{0.0556434f, -0.2040259f, 1.0572252f};
    Vec3f rec709_to_r{1.0f, 0.0f, 0.0f};
    Vec3f rec709_to_g{0.0f, 1.0f, 0.0f};
    Vec3f rec709_to_b{0.0f, 0.0f, 1.0f};
};

struct SceneSnapshot {
    std::uint64_t revision{};
    std::map<MaterialId, MaterialDesc> materials;
    std::map<ImageId, ImageDesc> images;
    std::map<GeometryId, TriangleMeshDesc> geometries;
    std::map<GeometryId, CurveGeometryDesc> curve_geometries;
    std::map<InstanceId, InstanceDesc> instances;
    std::map<CameraId, CameraDesc> cameras;
    std::map<LightId, LightDesc> lights;
    std::optional<CameraId> active_camera;
    std::optional<MaterialId> world_shader;
    // Cycles represents the world as a background-light object. Volume-stack
    // identity needs that exact object index independently of the world
    // shader index retained by MaterialDesc.
    std::optional<std::uint32_t>
        cycles_background_object_index;
    std::optional<EnvironmentDesc> environment;
    ShaderColorSpace shader_color_space;
    // Preserve Cycles' world-light policy independently from the raw world
    // closure graph and procedural environment resources.
    WorldSampling world_sampling{WorldSampling::automatic};
    std::uint32_t world_sample_map_resolution{1024u};
    // Cycles Background::visibility. Unlike an ordinary object this belongs
    // to the world shader/background-light pair, but uses the same public ray
    // categories.
    std::uint32_t world_visibility_mask{all_ray_visibility};
    bool world_cast_shadow{true};
    std::int32_t cycles_background_light_group{-1};
    // CyclesWorldSettings::max_bounces follows the same inclusive NEE limit
    // as analytic lamps.
    std::uint32_t world_max_bounces{1024u};
    // Blender World.light_settings.distance. Cycles mirrors this scene value
    // into kernel_data.integrator.ao_bounces_distance; shader AO nodes with
    // Global Radius enabled read it instead of their authored Distance.
    float ambient_occlusion_distance{10.0f};
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
struct UpsertCurveGeometry {
    GeometryId id;
    CurveGeometryDesc value;
};
struct RemoveCurveGeometry {
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
    UpsertCurveGeometry,
    RemoveCurveGeometry,
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
