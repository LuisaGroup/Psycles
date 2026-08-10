#pragma once

#include <psycles/luisa/path_tracer.h>

#include <psycles/compiler/material_library.h>
#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/cycles_sampler.h>
#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/volume_majorant_hierarchy.h>

#include "path_tracer_types.h"
#include "path_kernel_executor.h"
#include "path_tracer_volume_metadata.h"
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

namespace cycles_sampler =
    ::psycles::luisa_backend::cycles_sampler;
namespace tabulated_sobol =
    ::psycles::sampling::tabulated_sobol;

using compiler::CompiledMaterial;
using compiler::MaterialLibrary;
using compiler::ShaderCompiler;
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
using luisa::compute::Expr;
using luisa::compute::Float;
using luisa::compute::Float2;
using luisa::compute::Float3;
using luisa::compute::Float4;
using luisa::compute::ImageFloat;
using luisa::compute::Int;
using luisa::compute::Kernel1D;
using luisa::compute::Kernel2D;
using luisa::compute::Image;
using luisa::compute::Mesh;
using luisa::compute::ProceduralPrimitive;
using luisa::compute::Shader1D;
using luisa::compute::Stream;
using luisa::compute::Triangle;
using luisa::compute::UInt;
using luisa::compute::UInt2;
using luisa::compute::Var;
using luisa::compute::cast;
using luisa::compute::clamp;
using luisa::compute::cross;
using luisa::compute::def;
using luisa::compute::dispatch_id;
using luisa::compute::dispatch_x;
using luisa::compute::dot;
using luisa::compute::inverse;
using luisa::compute::length_squared;
using luisa::compute::make_float2;
using luisa::compute::make_float3;
using luisa::compute::make_float4;
using luisa::compute::make_float4x4;
using luisa::compute::make_uint2;
using luisa::compute::make_ray;
using luisa::compute::max;
using luisa::compute::min;
using luisa::compute::normalize;
using luisa::compute::offset_ray_origin;
using luisa::compute::select;
using luisa::compute::set_block_size;
using luisa::compute::synchronize;
using luisa::compute::transpose;
using luisa::compute::triangle_interpolate;

constexpr auto pi = 3.14159265358979323846f;
constexpr auto ray_maximum = 1.0e30f;
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
constexpr std::uint32_t attribute_domain_point = 0u;
constexpr std::uint32_t attribute_domain_corner = 1u;
constexpr std::uint32_t attribute_domain_face = 2u;
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
constexpr std::uint32_t light_flag_full_spread = 1u << 4u;
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

struct NishitaTextureBinding {
    std::uint32_t parameter_block{};
    std::uint32_t sky_index{};
    std::uint32_t texture_slot{};
    contract::NishitaSkyDesc parameters;
    luisa::float3 pixel_bottom_xyz{};
    luisa::float3 pixel_top_xyz{};
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
    std::size_t tangent_attribute_index{};
    std::size_t undisplaced_tangent_attribute_index{};
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

struct LuisaSceneData {
    luisa::compute::Device device;
    std::uint64_t revision{};
    MaterialLibrary materials;
    SurfaceDispatch surfaces;
    // Scene-level Cycles has_bssrdf_bump materials mapped onto the
    // structure-deduplicated SurfaceDispatch tags. Shared tags form the
    // conservative image of this exact material predicate; closures and
    // parameters remain device expressions.
    luisa::vector<luisa::uint> surface_bssrdf_bump_tags;
    // Host-stage scene capability used to decide whether the production path
    // kernel must record spatial BSSRDF transport at all. The predicate is
    // derived from the raw SurfaceProgram plus its parameter block; it never
    // rewrites, bakes, or approximates a material closure.
    bool has_subsurface{};
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
    // Shader literals retain their compiler IR type. Both buffers use the
    // same material-block/ParameterId address so dispatch remains compact,
    // while no float4 type erasure crosses the device ABI.
    Buffer<float> scalar_parameter_buffer;
    Buffer<luisa::float3> vector_parameter_buffer;
    Buffer<float> cycles_bsdf_table_buffer;
    std::vector<GeometryResource> geometries;
    std::vector<CurveGeometryResource> curve_geometries;
    std::vector<Image<float>> images;
    BindlessArray texture_heap;
    std::optional<std::uint32_t> environment_texture_slot;
    std::uint32_t environment_width{};
    std::uint32_t environment_height{};
    std::vector<contract::EnvironmentSunDesc> environment_suns;
    std::optional<NishitaEnvironmentRuntime>
        nishita_environment;
    bool environment_emission_is_constant{true};
    Buffer<luisa::float2> background_conditional_cdf;
    Buffer<luisa::float2> background_marginal_cdf;
    std::uint32_t background_map_width{1u};
    std::uint32_t background_map_height{1u};
    float background_map_weight{};
    float background_guided_sun_weight{};
    luisa::float3 background_guided_sun_axis{
        0.0f, 0.0f, 1.0f};
    float background_guided_sun_radius{};
    std::vector<NishitaTextureBinding>
        nishita_texture_bindings;
    BindlessArray heap;
    Buffer<GeometryGpu> geometry_buffer;
    Buffer<InstanceGpu> instance_buffer;
    Buffer<PrimitiveCompletionGpu> primitive_completion_buffer;
    Buffer<luisa::uint> primitive_completion_instance_buffer;
    // Only instances whose exact Cycles support needs endpoint completion
    // participate. Dense object identities use one direct device read; very
    // sparse identities use a sorted table whose size is the number of
    // special instances rather than the whole scene.
    Buffer<luisa::uint> cycles_completion_source_dense_buffer;
    std::uint32_t cycles_completion_source_dense_count{};
    Buffer<luisa::uint2> cycles_completion_source_sparse_buffer;
    std::uint32_t cycles_completion_source_sparse_count{};
    Buffer<MaterialBindingGpu> geometry_material_buffer;
    Buffer<MaterialBindingGpu> override_material_buffer;
    Buffer<AttributeBindingGpu> attribute_binding_buffer;
    Buffer<AttributeRangeGpu> attribute_range_buffer;
    std::uint32_t attribute_binding_slot{};
    std::uint32_t attribute_range_slot{};
    Buffer<LightGpu> light_buffer;
    std::uint32_t light_count{};
    Buffer<EmissiveTriangleGpu> emissive_triangle_buffer;
    std::uint32_t emissive_triangle_count{};
    Buffer<LightDistributionGpu> light_distribution_buffer;
    std::uint32_t light_distribution_count{};
    Buffer<LightTreeNodeGpu> light_tree_node_buffer;
    Buffer<LightTreeEmitterGpu> light_tree_emitter_buffer;
    // Indexed by stable emitter id: reordered-emitter index and leaf node.
    Buffer<luisa::uint2> light_tree_emitter_mapping_buffer;
    Buffer<luisa::uint4> light_tree_triangle_lookup_buffer;
    std::uint32_t light_tree_node_count{};
    std::uint32_t light_tree_emitter_count{};
    std::uint32_t light_tree_triangle_lookup_count{};
    std::uint32_t light_tree_root{~std::uint32_t{0u}};
    float triangle_area_pdf{};
    float light_selection_pdf{};
    bool environment_in_light_distribution{};
    luisa::float3 background{};
    contract::ShaderColorSpace shader_color_space;
    Accel accel;
    CameraDesc camera;
    VolumeSceneMetadata volume_metadata;
    Buffer<luisa::uint> volume_surface_flag_buffer;
    std::uint32_t volume_surface_flag_count{};
    Buffer<VolumeMajorantNodeGpu>
        volume_majorant_node_buffer;
    Buffer<VolumeMajorantRootGpu>
        volume_majorant_root_buffer;
    Buffer<VolumeMajorantRootRangeGpu>
        volume_majorant_range_buffer;
    std::uint32_t volume_majorant_root_count{};
    std::uint32_t volume_majorant_node_count{};
    std::uint32_t volume_majorant_range_count{};
    std::uint32_t volume_majorant_world_range{
        ~std::uint32_t{0u}};
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
    std::uint32_t _sobol_sequence_size{};
    std::uint32_t _total_aa_samples{};
    std::uint32_t _rendered_samples{};
    RenderKernelParameters _kernel_parameters{};
    PathKernelExecutor _render_executor;
    std::unique_ptr<VolumeGuidingFilter>
        _volume_guiding_filter;
    bool _path_trace_delivered{false};
    std::atomic_bool _cancelled{false};

private:
    [[nodiscard]] std::size_t pixel_count() const noexcept;
    void prepare_sobol_table(std::uint32_t total_samples);
    void initialize(const RenderSettings &settings);
    [[nodiscard]] bool write_passes(contract::OutputSink &output);
    void deliver_path_trace();

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
};

[[nodiscard]] MaterialBindingGpu to_luisa(
    MaterialBinding binding) noexcept;
[[nodiscard]] Var<ShaderEvaluationStateCall>
pack_shader_evaluation_state(
    const cycles_path_state::ShaderEvaluationState &state) noexcept;
[[nodiscard]] cycles_path_state::ShaderEvaluationState
unpack_shader_evaluation_state(
    const Var<ShaderEvaluationStateCall> &state) noexcept;
[[nodiscard]] Var<SurfacePointCall> pack_surface_point(
    const SurfacePoint &point) noexcept;
[[nodiscard]] SurfacePoint unpack_surface_point(
    const Var<SurfacePointCall> &point) noexcept;
[[nodiscard]] Var<SurfaceEvaluationCall> pack_surface_evaluation(
    const SurfaceEvaluation &evaluation) noexcept;
[[nodiscard]] SurfaceEvaluation unpack_surface_evaluation(
    const Var<SurfaceEvaluationCall> &evaluation) noexcept;
[[nodiscard]] Var<SurfaceSampleCall> pack_surface_sample(
    const SurfaceSample &sample) noexcept;
[[nodiscard]] SurfaceSample unpack_surface_sample(
    const Var<SurfaceSampleCall> &sample) noexcept;
[[nodiscard]] Var<SurfaceClosureTraceCall>
pack_surface_closure_trace(
    const SurfaceClosureTrace &trace) noexcept;
[[nodiscard]] SurfaceClosureTrace
unpack_surface_closure_trace(
    const Var<SurfaceClosureTraceCall> &trace) noexcept;
[[nodiscard]] Var<SurfaceSampleTraceCall>
pack_surface_sample_trace(
    const SurfaceSampleTrace &trace) noexcept;
[[nodiscard]] SurfaceSampleTrace
unpack_surface_sample_trace(
    const Var<SurfaceSampleTraceCall> &trace) noexcept;
[[nodiscard]] Var<SurfacePreparationQueryCall>
pack_surface_preparation_query(
    const SurfacePreparationQuery &query) noexcept;
[[nodiscard]] SurfacePreparationQuery
unpack_surface_preparation_query(
    const Var<SurfacePreparationQueryCall> &query) noexcept;
[[nodiscard]] Var<SurfacePreparationCall> pack_surface_preparation(
    const SurfacePreparation &preparation) noexcept;
[[nodiscard]] SurfacePreparation unpack_surface_preparation(
    const Var<SurfacePreparationCall> &preparation) noexcept;
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
