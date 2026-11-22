#pragma once

#include <psycles/luisa/path_tracer.h>

#include <psycles/compiler/cycles_svm_scene.h>
#include <psycles/compiler/cycles_svm_object_scene.h>
#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/cycles_sampler.h>
#include <psycles/luisa/volume_majorant_hierarchy.h>

#include "path_kernel_executor.h"
#include "cycles_svm_scene_image.h"
#include "path_tracer_types.h"
#include "path_tracer_volume_metadata.h"
#include "surface_math_constants.h"
#include "volume_guiding_filter.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/vector.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/rtx/accel.h>
#include <luisa/dsl/rtx/hit.h>
#include <luisa/dsl/rtx/ray.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>
#include <luisa/luisa-compute.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/mesh.h>
#include <luisa/runtime/rtx/procedural_primitive.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

namespace psycles::luisa_backend::detail {

struct PathDiagnosticBufferLayout {
    std::size_t path_trace_slot_count{};
    std::size_t surface_closure_count_histogram_base{};
    std::size_t surface_closure_count_histogram_slot_count{};
    std::size_t allocation_slot_count{};
};

[[nodiscard]] PathDiagnosticBufferLayout path_diagnostic_buffer_layout(
    const LuisaPathTracerOptions &options) noexcept;

namespace cycles_sampler =
    ::psycles::luisa_backend::cycles_sampler;
namespace tabulated_sobol =
    ::psycles::sampling::tabulated_sobol;

using contract::CameraDesc;
using contract::CameraProjection;
using contract::CameraSensorFit;
using contract::ImageExtent;
using contract::LightType;
using contract::PassKind;
using contract::PassRequest;
using contract::PassTile;
using contract::PixelWindow;
using contract::RayVisibility;
using contract::RenderDiagnostic;
using contract::RenderSettings;
using contract::SampleRange;
using contract::SceneSnapshot;
using luisa::compute::Accel;
using luisa::compute::AccelVar;
using luisa::compute::BindlessArray;
using luisa::compute::BindlessVar;
using luisa::compute::Bool;
using luisa::compute::Buffer;
using luisa::compute::BufferFloat;
using luisa::compute::BufferFloat3;
using luisa::compute::BufferFloat4;
using luisa::compute::BufferUInt;
using luisa::compute::Callable;
using luisa::compute::cast;
using luisa::compute::clamp;
using luisa::compute::cross;
using luisa::compute::def;
using luisa::compute::dispatch_id;
using luisa::compute::dispatch_x;
using luisa::compute::dot;
using luisa::compute::Expr;
using luisa::compute::Float;
using luisa::compute::Float2;
using luisa::compute::Float3;
using luisa::compute::Float4;
using luisa::compute::Image;
using luisa::compute::ImageFloat;
using luisa::compute::Int;
using luisa::compute::inverse;
using luisa::compute::Kernel1D;
using luisa::compute::Kernel2D;
using luisa::compute::length_squared;
using luisa::compute::make_float2;
using luisa::compute::make_float3;
using luisa::compute::make_float4;
using luisa::compute::make_float4x4;
using luisa::compute::make_ray;
using luisa::compute::make_uint2;
using luisa::compute::max;
using luisa::compute::Mesh;
using luisa::compute::min;
using luisa::compute::normalize;
using luisa::compute::offset_ray_origin;
using luisa::compute::ProceduralPrimitive;
using luisa::compute::select;
using luisa::compute::set_block_size;
using luisa::compute::Shader1D;
using luisa::compute::Stream;
using luisa::compute::synchronize;
using luisa::compute::transpose;
using luisa::compute::Triangle;
using luisa::compute::triangle_interpolate;
using luisa::compute::UInt;
using luisa::compute::UInt2;
using luisa::compute::Var;

// Cycles uses FLT_MAX both as the unbounded ray interval and as the observable
// distant/background shader distance. A smaller finite sentinel changes the
// SHADE_LIGHT_NEE sun setup branch and Light Path's Ray Length.
constexpr auto ray_maximum = std::numeric_limits<float>::max();
// Slot 9 stores the Cycles intersection representation of positions. It
// aliases slot 1 for object-space geometry and points at host-transformed
// vertices for single-user static meshes.
// Slots 10-12 retain the position, normal, and tangent state from before
// true displacement. Non-displaced geometry aliases these slots to the
// ordinary buffers, so shader code has one invariant representation.
constexpr std::uint32_t geometry_bindless_stride = 13u;
constexpr std::uint32_t geometry_normal_corner = 1u << 0u;
constexpr std::uint32_t geometry_uv_corner = 1u << 1u;
constexpr std::uint32_t geometry_uv_tangent_corner = 1u << 2u;
constexpr std::uint32_t geometry_generated_corner = 1u << 3u;
constexpr std::uint32_t geometry_curve_default_uv = 1u << 4u;
constexpr std::uint32_t attribute_domain_point = 0u;
constexpr std::uint32_t attribute_domain_corner = 1u;
constexpr std::uint32_t attribute_domain_face = 2u;
constexpr std::uint32_t attribute_domain_curve = 3u;
constexpr std::uint32_t attribute_domain_mask = 0xffu;
constexpr std::uint32_t attribute_format_shift = 8u;
constexpr std::uint32_t attribute_format_float4 = 0u;
constexpr std::uint32_t attribute_format_float2 = 1u;

[[nodiscard]] constexpr std::uint32_t pack_attribute_layout(
    std::uint32_t domain,
    std::uint32_t format = attribute_format_float4) noexcept {
    return (domain & attribute_domain_mask) |
           (format << attribute_format_shift);
}
constexpr auto camera_visibility =
    contract::visibility_bit(RayVisibility::camera);
constexpr auto diffuse_visibility =
    contract::visibility_bit(RayVisibility::diffuse);
constexpr auto glossy_visibility =
    contract::visibility_bit(RayVisibility::glossy);
constexpr auto transmission_visibility =
    contract::visibility_bit(RayVisibility::transmission);
constexpr auto shadow_visibility =
    contract::visibility_bit(RayVisibility::shadow);
constexpr auto volume_scatter_visibility =
    contract::visibility_bit(RayVisibility::volume_scatter);
// Cycles considers a mesh usable as a sampled light only when at least one
// non-camera transport class can reach it. Keep this mask shared by scene
// construction and JIT-stage primitive resolution so forward-hit MIS and the
// emitter distribution cannot disagree about membership.
constexpr auto mesh_light_sampling_visibility =
    diffuse_visibility | glossy_visibility | transmission_visibility |
    volume_scatter_visibility;
static_assert(
    static_cast<std::uint32_t>(
        contract::EmissionSampling::front_back) <=
    (material_emission_sampling_mask >>
     material_emission_sampling_shift));
constexpr std::uint32_t light_flag_normalize = 1u << 0u;
constexpr std::uint32_t light_flag_ellipse = 1u << 1u;
constexpr std::uint32_t light_flag_sphere = 1u << 2u;
constexpr std::uint32_t light_flag_use_mis = 1u << 3u;
constexpr std::uint32_t light_flag_forward_intersectable =
    1u << 5u;
constexpr std::uint32_t light_flag_constant_emission =
    1u << 6u;


struct MaterialBinding {
    std::uint32_t surface_tag{};
    std::uint32_t parameter_block{};
    std::uint32_t cycles_shader_index{
        ~std::uint32_t{0u}};
    // Stable path-stack fallback for renderer-neutral scenes which do not
    // carry Blender's exact Cycles shader index.
    std::uint32_t material_identity{
        ~std::uint32_t{0u}};
    std::uint32_t flags{};
    // Authored Cycles sampling policy. This is intentionally independent of
    // material_flag_may_emit: an emissive closure remains visible when light
    // sampling is disabled, but it must not compete in forward-hit MIS.
    contract::EmissionSampling emission_sampling{
        contract::EmissionSampling::automatic};
    contract::VolumeSampling volume_sampling{
        contract::VolumeSampling::
            multiple_importance};
};

struct GeometryResource {
    Buffer<luisa::float3> positions;
    Buffer<luisa::float3> normals;
    Buffer<luisa::float2> uv;
    Buffer<luisa::float4> uv_tangents;
    Buffer<luisa::float3> generated;
    Buffer<Triangle> triangles;
    Buffer<luisa::uint> triangle_material_slots;
    Buffer<float> triangle_random_per_island;
    Buffer<luisa::uint> triangle_smooth;
    std::vector<Buffer<luisa::float4>> attributes;
    std::optional<Buffer<luisa::float3>>
        cycles_intersection_positions;
    std::optional<Buffer<luisa::float3>>
        undisplaced_positions;
    std::optional<Buffer<luisa::float3>>
        undisplaced_normals;
    std::optional<Buffer<luisa::float4>>
        undisplaced_uv_tangents;
    Mesh mesh;
};

struct CurveGeometryResource {
    Buffer<luisa::compute::AABB> bounds;
    Buffer<CurveSegmentGpu> segments;
    Buffer<luisa::float4> keys;
    std::vector<Buffer<luisa::float2>> uv_layers;
    std::vector<Buffer<luisa::float4>> attributes;
    Buffer<luisa::uint> material_slots;
    Buffer<float> intercept;
    Buffer<float> length;
    Buffer<float> random;
    ProceduralPrimitive primitive;
};

struct AttributeUpload {
    std::uint64_t id{};
    std::uint32_t domain{};
    luisa::vector<luisa::float4> values;
};

struct UvTangentLayerUpload {
    std::size_t uv_attribute_index{};
    std::optional<std::size_t> tangent_attribute_index;
    std::optional<std::size_t> undisplaced_tangent_attribute_index;
};

struct GeometryUpload {
    std::uint32_t attribute_domains{};
    bool default_uv_available{};
    luisa::float4x4 generated_transform{};
    luisa::vector<luisa::float3> positions;
    luisa::vector<luisa::float3> cycles_intersection_positions;
    luisa::vector<luisa::float3> normals;
    luisa::vector<luisa::float2> uv;
    luisa::vector<luisa::float4> uv_tangents;
    luisa::vector<luisa::float3> undisplaced_positions;
    luisa::vector<luisa::float3> undisplaced_normals;
    luisa::vector<luisa::float4> undisplaced_uv_tangents;
    luisa::vector<luisa::float3> generated;
    luisa::vector<Triangle> triangles;
    luisa::vector<luisa::uint> triangle_material_slots;
    luisa::vector<float> triangle_random_per_island;
    luisa::vector<luisa::uint> triangle_smooth;
    std::vector<AttributeUpload> attributes;
    std::vector<UvTangentLayerUpload> uv_tangent_layers;
};

struct NishitaEnvironmentRuntime {
    contract::NishitaSkyDesc parameters;
    luisa::float3 pixel_bottom_xyz{};
    luisa::float3 pixel_top_xyz{};
    luisa::float3 sun_direction{};
    float angular_radius{};
};

// Transactional host/device image of Cycles' native scene-wide SVM program.
// `material_shader_indices` is required only for renderer-authored scenes that
// lack Blender's source index; Blender bundles preserve their exact
// Scene::shaders identity. Resources remain typed and separately indexed just
// as in Cycles DeviceScene.
struct CyclesSvmGeometryRuntime {
    CyclesSvmGeometrySceneImage image;
    Buffer<compiler::cycles_svm::AttributeMap> attribute_map_buffer;
    Buffer<float> attribute_float_buffer;
    Buffer<compiler::cycles_svm::packed_float2> attribute_float2_buffer;
    Buffer<compiler::cycles_svm::packed_float3> attribute_float3_buffer;
    Buffer<compiler::cycles_svm::packed_float4> attribute_float4_buffer;
    Buffer<compiler::cycles_svm::uchar4> attribute_uchar4_buffer;
    Buffer<compiler::cycles_svm::packed_normal> attribute_normal_buffer;
    Buffer<compiler::cycles_svm::packed_float3> triangle_vertex_buffer;
    Buffer<compiler::cycles_svm::packed_float4> curve_key_buffer;
    Buffer<compiler::cycles_svm::packed_float4> point_buffer;
    Buffer<compiler::cycles_svm::packed_uint3> triangle_index_buffer;
    Buffer<luisa::uint> triangle_shader_buffer;
    Buffer<compiler::cycles_svm::KernelCurve> curve_buffer;
};

struct CyclesSvmObjectRuntime {
    CyclesSvmObjectSceneImage image;
    Buffer<compiler::cycles_svm::KernelObject> object_buffer;
    Buffer<luisa::uint> object_flag_buffer;
};

struct CyclesSvmNishitaImageRuntime {
    compiler::cycles_svm::NishitaImageBinding parameters;
    std::uint32_t texture_slot{};
};

struct CyclesSvmRuntime {
    compiler::cycles_svm::CompiledShaderTable compilation;
    // Exact Cycles Scene::dscene.data.kernel_features projection consumed as
    // a host/JIT specialization constant. Shader features are known at
    // compilation; geometry features are added transactionally at finalization.
    std::uint32_t kernel_features{};
    compiler::cycles_svm::ObjectIdentityPlan object_identities;
    compiler::cycles_svm::ParticleTableImage particles;
    std::map<contract::MaterialId, std::uint32_t>
        material_shader_indices;
    luisa::vector<CyclesSvmImageBindingGpu> image_bindings;
    // Cycles SkyLoader resources occupy the same logical image-handle table
    // as exported images, but are generated from immutable node parameters.
    // Their bindless slots are assigned once during scene construction and
    // never alias a scene ImageId.
    std::vector<CyclesSvmNishitaImageRuntime> nishita_images;
    luisa::vector<CyclesSvmParticleGpu> particle_records;
    std::optional<Buffer<luisa::uint>> word_buffer;
    std::optional<Buffer<compiler::cycles_svm::KernelShader>>
        kernel_shader_buffer;
    std::optional<Buffer<float>> ies_buffer;
    std::optional<Buffer<CyclesSvmImageBindingGpu>>
        image_binding_buffer;
    std::optional<Buffer<CyclesSvmParticleGpu>> particle_buffer;
    // Null is the only pre-finalization state. The complete typed image is
    // installed atomically after displacement has fixed the final geometry.
    std::unique_ptr<CyclesSvmGeometryRuntime> geometry;
    std::unique_ptr<CyclesSvmObjectRuntime> objects;
};

struct VolumeMajorantScenePlan;
struct VolumeMajorantRuntime {
    Buffer<VolumeMajorantNodeGpu> node_buffer;
    Buffer<VolumeMajorantRootGpu> root_buffer;
    Buffer<VolumeMajorantRootRangeGpu> range_buffer;
    std::uint32_t root_count{};
    std::uint32_t node_count{};
    std::uint32_t range_count{};
    std::uint32_t world_range{~0u};
};

struct LuisaSceneData {
    luisa::compute::Device device;
    std::uint64_t revision{};
    std::unique_ptr<CyclesSvmRuntime> cycles_svm;
    bool has_subsurface{};
    std::uint32_t subsurface_instance_count{};
    bool has_ambient_occlusion{};
    bool has_local_ambient_occlusion{};
    std::optional<Buffer<float>> ambient_occlusion_distance_buffer;
    std::map<contract::MaterialId, MaterialBinding>
        material_bindings;
    std::optional<MaterialBinding> world_surface;
    std::uint32_t cycles_background_object_index{
        ~std::uint32_t{0u}};
    std::uint32_t cycles_background_shader_id{
        ~std::uint32_t{0u}};
    std::uint32_t cycles_background_shader_flags{};
    std::int32_t cycles_background_light_group{-1};
    std::uint32_t world_visibility_mask{
        contract::all_ray_visibility};
    std::uint32_t world_max_bounces{1024u};
    Buffer<float> cycles_bsdf_table_buffer;
    std::vector<GeometryResource> geometries;
    std::vector<CurveGeometryResource> curve_geometries;
    std::vector<Image<float>> images;
    BindlessArray texture_heap;
    // The scene image upload below has one closed set of storage choices:
    // decoded UNORM8 images become BYTE4 and wider/HDR images become FLOAT4.
    // Keep the default conservative so any future/imported storage path still
    // retains HIP's generic R10G10B10A2 sampling support.
    bool may_have_packed_textures{true};
    std::optional<std::uint32_t> environment_texture_slot;
    std::uint32_t environment_width{};
    std::uint32_t environment_height{};
    std::vector<contract::EnvironmentSunDesc> environment_suns;
    std::optional<NishitaEnvironmentRuntime>
        nishita_environment;
    bool environment_emission_is_constant{true};
    std::uint32_t background_map_width{1u};
    std::uint32_t background_map_height{1u};
    float background_portal_weight{};
    float background_map_weight{};
    float background_guided_sun_weight{};
    luisa::float3 background_guided_sun_axis{
        0.0f, 0.0f, 1.0f};
    float background_guided_sun_radius{};
    BindlessArray heap;
    Buffer<GeometryGpu> geometry_buffer;
    Buffer<InstanceGpu> instance_buffer;
    // Sparse-in-use, dense-by-Cycles-object table for Fast GI ray-length
    // overrides. It stays separate from hot InstanceGpu/traversal records so
    // ordinary closest-hit traffic pays no stride increase.
    Buffer<float> ambient_occlusion_object_distance_buffer;
    std::uint32_t ambient_occlusion_object_distance_count{};
    Buffer<MaterialBindingGpu> geometry_material_buffer;
    Buffer<MaterialBindingGpu> override_material_buffer;
    // Read-only traversal quotient. These tables project the immutable scene
    // records onto exactly the identity/material-capability fields observed by
    // candidate callbacks; full material bindings remain the shading source.
    Buffer<SceneTraversalInstanceGpu> traversal_instance_buffer;
    Buffer<luisa::uint> traversal_material_flags_buffer;
    std::optional<Buffer<luisa::uint2>> traversal_object_instance_buffer;
    std::uint32_t traversal_object_instance_count{};
    std::optional<Buffer<luisa::uint>> traversal_shadow_opacity_buffer;
    bool has_transparent_shadow_materials{true};
    bool has_mixed_shadow_opacity_instances{true};
    std::vector<luisa::uint> primary_to_subsurface_local_host;
    std::optional<Buffer<luisa::uint>> primary_to_subsurface_local_buffer;
    std::vector<luisa::uint> primary_to_volume_local_host;
    std::optional<Buffer<luisa::uint>> primary_to_volume_local_buffer;
    bool all_volume_boundary_primitives_have_volume{false};
    Buffer<LightGpu> light_buffer;
    std::uint32_t light_count{};
    std::uint32_t light_type_mask{0xfu};
    // Portal records follow regular lights in light_buffer and never enter
    // the direct-light distribution. Their offset is exactly light_count.
    std::uint32_t portal_count{};
    Buffer<EmissiveTriangleGpu> emissive_triangle_buffer;
    std::uint32_t emissive_triangle_count{};
    Buffer<LightDistributionGpu> light_distribution_buffer;
    std::uint32_t light_distribution_count{};
    Buffer<LightTreeNodeGpu> light_tree_node_buffer;
    Buffer<LightTreeEmitterGpu> light_tree_emitter_buffer;
    // Indexed by flat distribution emitter id: top-level emitter and leaf.
    Buffer<luisa::uint2> light_tree_emitter_mapping_buffer;
    // Indexed by emissive-triangle emitter id: mesh-local emitter and leaf.
    Buffer<luisa::uint2> light_tree_triangle_emitter_mapping_buffer;
    // Concatenated per-mesh-proxy maps from local emitter id to the actual
    // instance-specific emissive-triangle distribution id.
    Buffer<luisa::uint> light_tree_mesh_triangle_buffer;
    Buffer<luisa::uint4> light_tree_triangle_lookup_buffer;
    std::uint32_t light_tree_node_count{};
    std::uint32_t light_tree_emitter_count{};
    std::uint32_t light_tree_triangle_lookup_count{};
    std::uint32_t light_tree_mesh_triangle_count{};
    std::uint32_t light_tree_root{~std::uint32_t{0u}};
    float triangle_area_pdf{};
    float light_selection_pdf{};
    bool environment_in_light_distribution{};
    luisa::float3 background{};
    contract::ShaderColorSpace shader_color_space;
    // Cycles local intersection traverses only the current object's triangle
    // BLAS. Luisa's public traversal abstraction is TLAS-based, so the first
    // exact domain reduction is a secondary TLAS containing every complete
    // triangle object that can originate BSSRDF transport. Candidate user ids
    // map bijectively back to the primary TLAS/InstanceGpu index. The callback
    // still filters the current Cycles object; this structure only removes
    // objects that can be proved unreachable at the host/JIT stage. Keep both
    // TLAS resources after their referenced meshes so reverse member
    // destruction releases acceleration structures first.
    std::optional<Accel> subsurface_accel;
    // Cycles Only Local AO traverses the complete current triangle object and
    // ignores ordinary ray visibility. This secondary TLAS preserves primary
    // instance ids as user ids, while its all-visible masks make that semantic
    // distinction explicit. Curves are intentionally absent.
    std::optional<Accel> ambient_occlusion_local_accel;
    std::optional<Accel> volume_boundary_accel;
    Accel accel;
    std::shared_ptr<const struct FilmPassAccessorGpu> film_pass_accessor;
    CameraDesc camera;
    VolumeSceneMetadata volume_metadata;
    std::shared_ptr<const VolumeMajorantScenePlan> volume_majorant_plan;
};

class LuisaCompiledScene final : public contract::CompiledScene {

private:
    std::shared_ptr<LuisaSceneData> _data;

public:
    explicit LuisaCompiledScene(
        std::shared_ptr<LuisaSceneData> data) noexcept
        : _data{std::move(data)} {}

    [[nodiscard]] std::uint64_t source_revision()
        const noexcept override {
        return _data->revision;
    }

    [[nodiscard]] const std::shared_ptr<LuisaSceneData> &
    data() const noexcept {
        return _data;
    }
};

enum class LightPassBuffer : std::uint32_t {
    diffuse_direct,
    diffuse_indirect,
    glossy_direct,
    glossy_indirect,
    transmission_direct,
    transmission_indirect,
    volume_direct,
    volume_indirect,
    emission,
    environment,
    glossy_color,
    transmission_color,
    count
};

[[nodiscard]] constexpr std::uint32_t light_pass_index(
    LightPassBuffer pass) noexcept {
    return static_cast<std::uint32_t>(pass);
}

inline constexpr auto light_pass_buffer_count =
    light_pass_index(LightPassBuffer::count);


struct BackgroundSamplingDistribution;

using FilmSampleVerifyShader = luisa::compute::Shader1D<
    luisa::compute::Buffer<luisa::uint>,
    luisa::compute::Buffer<luisa::uint>,
    std::uint32_t,
    std::uint32_t>;

using FilmPassResolveShader = luisa::compute::Shader1D<
    luisa::compute::Buffer<luisa::float4>,
    luisa::compute::Buffer<luisa::float4>,
    luisa::compute::Buffer<luisa::float4>,
    luisa::compute::Buffer<luisa::float4>,
    luisa::compute::Buffer<luisa::uint>,
    luisa::compute::Buffer<float>,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    float>;

struct FilmPassAccessorGpu {
    FilmSampleVerifyShader verify_sample_count;
    FilmPassResolveShader resolve_pass;
};

class LuisaRenderSession final : public contract::RenderSession {

private:
    std::shared_ptr<LuisaSceneData> _scene;
    LuisaPathTracerOptions _options;
    RenderSettings _settings;
    PixelWindow _window;
    Stream _stream;
    Buffer<luisa::float4> _combined;
    Buffer<luisa::float4> _normal;
    Buffer<luisa::float4> _albedo;
    Buffer<luisa::float4> _light_passes;
    Buffer<luisa::uint> _sample_count;
    Buffer<luisa::float4> _volume_guiding_raw;
    Buffer<luisa::uint> _volume_guiding_denoised;
    Buffer<luisa::uint> _volume_guiding_intermediate;
    Buffer<luisa::float4> _path_trace;
    Buffer<luisa::float4> _sobol_table;
    Buffer<float> _pixel_filter_table;
    Buffer<luisa::uint> _sample_mismatch_flag;
    Buffer<float> _resolved_pass_staging;
    std::vector<float> _resolved_pass_host;
    std::shared_ptr<const BackgroundSamplingDistribution> _background_sampling;
    std::shared_ptr<const VolumeMajorantRuntime> _volume_majorants;
    std::uint32_t _sobol_sequence_size{};
    std::uint32_t _total_aa_samples{};
    std::uint32_t _rendered_samples{};
    RenderKernelParameters _kernel_parameters{};
    PathKernelExecutor _render_executor;
    std::unique_ptr<VolumeGuidingFilter>
        _volume_guiding_filter;
    bool _path_trace_delivered{false};
    std::optional<contract::RenderSampleTimings> _last_render_timings;
    std::atomic_bool _cancelled{false};

private:
    [[nodiscard]] std::size_t pixel_count() const noexcept;
    void prepare_sobol_table(std::uint32_t total_samples);
    void prepare_film_pass_accessor();
    void initialize(const RenderSettings &settings);
    [[nodiscard]] bool write_passes(contract::OutputSink &output);
    void deliver_path_trace();
    void deliver_surface_closure_count_histogram();

  public:
    LuisaRenderSession(
        std::shared_ptr<LuisaSceneData> scene,
        LuisaPathTracerOptions options,
        const RenderSettings &settings);

    void reset(const RenderSettings &settings) override;
    bool render_samples(
        const SampleRange &samples,
        contract::OutputSink &output) override;
    void cancel() noexcept override;
    [[nodiscard]] std::optional<contract::RenderSampleTimings>
    last_render_timings() const noexcept override;
};

[[nodiscard]] MaterialBindingGpu to_luisa(
    MaterialBinding binding) noexcept;
[[nodiscard]] Var<ShaderEvaluationStateCall>
pack_shader_evaluation_state(
    const cycles_path_state::ShaderEvaluationState &state) noexcept;
[[nodiscard]] luisa::float3 to_luisa(Vec3f value) noexcept;
[[nodiscard]] luisa::float2 to_luisa(Vec2f value) noexcept;
[[nodiscard]] luisa::float4x4 to_luisa(Mat4f value) noexcept;
[[nodiscard]] Vec3f matrix_axis(
    const Mat4f &matrix,
    std::size_t column) noexcept;
[[nodiscard]] Vec3f matrix_translation(
    const Mat4f &matrix) noexcept;
[[nodiscard]] float scalar_parameter_value(
    const contract::SocketValue &value) noexcept;
[[nodiscard]] luisa::float3 vector_parameter_value(
    const contract::SocketValue &value) noexcept;
[[nodiscard]] luisa::float3 unsigned_parameter_value(
    const contract::SocketValue &value) noexcept;
[[nodiscard]] PixelWindow effective_window(
    const RenderSettings &settings) noexcept;
[[nodiscard]] std::uint32_t pass_channels(
    const PassRequest &pass) noexcept;
[[nodiscard]] luisa::float3 safe_divide_even_color(
    const luisa::float4 &numerator,
    const luisa::float4 &denominator) noexcept;
[[nodiscard]] bool supported_pass(PassKind pass) noexcept;
void diagnose(
    std::vector<RenderDiagnostic> &diagnostics,
    std::string message);

}// namespace psycles::luisa_backend::detail
