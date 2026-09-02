#pragma once

#include <psycles/luisa/path_tracer.h>

#include <psycles/compiler/material_library.h>
#include <psycles/compiler/cycles_svm_scene.h>
#include <psycles/compiler/cycles_svm_object_scene.h>
#include <psycles/compiler/surface_execution_plan.h>
#include <psycles/compiler/surface_svm_program.h>
#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/cycles_sampler.h>
#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/surface_closure_reachability.h>
#include <psycles/luisa/surface_value_runtime_limits.h>
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
    std::size_t surface_program_execution_histogram_base{};
    std::size_t surface_program_execution_histogram_slot_count{};
    std::size_t allocation_slot_count{};
};

// Each float lane is an exact integer only below 2^24. Sixteen float4 shards
// raise the per-topology checked domain to roughly one billion populations
// while retaining the existing cross-backend float-atomic contract.
inline constexpr std::size_t
    surface_program_execution_histogram_shards_per_topology = 16u;
inline constexpr std::size_t
    surface_program_execution_histogram_lanes_per_topology =
        4u * surface_program_execution_histogram_shards_per_topology;

[[nodiscard]] PathDiagnosticBufferLayout path_diagnostic_buffer_layout(
    const LuisaPathTracerOptions &options,
    std::size_t surface_value_topology_count) noexcept;

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

// Scene-pruned typed value program used by the compact surface-population
// path. The host image is retained because Luisa records one evaluator body
// per exact semantic variant while the path kernel is constructed. Device
// streams contain only program control, typed addresses, and late-bound table
// parameter ids; authored material parameters remain in their existing SoA
// buffers.
struct SurfaceValueRuntimeTopology {
    std::shared_ptr<const compiler::SurfaceProgram> program;
};

// Exact host/JIT identity of one closure-leaf evaluator in the replacement
// SVM. `static_variant` selects the closure algorithm and its topology flags;
// `principled_features` is the per-leaf Cycles feature mask, not a scene-wide
// union. Keeping the pair explicit is what lets the device interpreter emit
// only the Principled branches which can actually inhabit a runtime leaf.
struct SurfaceSvmClosureVariant {
    std::uint32_t static_variant{};
    compiler::PrincipledClosureFeatureMask principled_features{};

    auto operator<=>(const SurfaceSvmClosureVariant &) const noexcept =
        default;
};

// Fixed semantic slots in the compact surface program's private bindless
// device view. The view has scene-independent shape, so adding material
// programs changes data only; it does not grow the path-kernel resource ABI.
enum class SurfaceValueRuntimeBufferSlot : std::uint32_t {
  svm_program,
  svm_instruction,
  svm_value_operand,
  svm_metadata_static_u0,
  svm_metadata_parameter,
  svm_metadata_static_range,
  svm_static_data,
  svm_closure_operand,
  count,
};

[[nodiscard]] constexpr std::uint32_t
surface_value_runtime_buffer_slot(SurfaceValueRuntimeBufferSlot slot) noexcept {
    return static_cast<std::uint32_t>(slot);
}

inline constexpr auto surface_value_runtime_buffer_slot_count =
    surface_value_runtime_buffer_slot(SurfaceValueRuntimeBufferSlot::count);
static_assert(surface_value_runtime_buffer_slot_count == 8u);

struct SurfaceValueRuntime {
    static constexpr std::uint32_t programs_per_topology = 2u;
    static constexpr std::uint32_t preparation_program_offset = 0u;
    static constexpr std::uint32_t emission_program_offset = 1u;
    // This is the Cycles-compatible 32-bit lane-stack execution capacity, not
    // a weakly typed value ABI. The builder rejects a program whose formally
    // colored live ranges exceed the finite SVM stack.
    static constexpr auto storage_capacity =
        compact_surface_value_storage_capacity;
    static constexpr std::uint32_t stack_capacity =
        storage_capacity.stack_lanes;

    // Exact evaluator domain proven directly from the unified CFG and its
    // source-provenance relation. No split-stream executable participates in
    // production handler selection.
    std::vector<compiler::SurfaceValueStaticVariant> value_variants;
    // Executed single-stream image. `svm_instruction_variants` is retained
    // strictly as a host-side construction proof and domain projection. The
    // device instruction selects its typed handler from its own control word;
    // no scene-parallel evaluator side stream is uploaded or read.
    compiler::SurfaceSvmSceneImage svm_scene;
    std::vector<std::uint32_t> svm_instruction_variants;
    std::vector<SurfaceSvmClosureVariant>
        preparation_svm_closure_variants;
    std::vector<SurfaceSvmClosureVariant>
        emission_svm_closure_variants;
    std::vector<SurfaceSvmClosureVariant>
        bssrdf_svm_closure_variants;
    std::vector<SurfaceValueRuntimeTopology> topologies;
    // Sorted unique host/JIT semantic keys. The device switches on the same
    // masked control word, so AST size is bounded by closure algorithms used
    // by the scene rather than by material topology count.
    std::vector<std::uint32_t> closure_static_variants;
    compiler::PrincipledClosureFeatureMask used_principled_closure_features{};
    // Exact host/JIT image of the canonical physical identities reachable
    // from the scene's closure opcodes and Principled feature stream. This is
    // specialization metadata only; authored parameters remain device data.
    SurfaceClosureReachability physical_closure_reachability{};
    // Exact semantic domains induced by each projected one-stream program.
    // The automatic-normal prefix, graph-expanded Bump samples, and endpoint
    // root form one transaction; no hidden callable domain exists.
    std::vector<std::uint32_t> preparation_value_static_variants;
    // Emission is a separately projected consumer and therefore has its own
    // transaction domain. These sets are host/JIT metadata only; material data
    // and the device program id remain runtime values.
    std::vector<std::uint32_t> emission_value_static_variants;
    std::vector<std::uint32_t> emission_closure_static_variants;
    compiler::PrincipledClosureFeatureMask
        emission_principled_closure_features{};
    // BSSRDF exit reconstruction can only be invoked for the conservative
    // topology-tag set derived from Cycles' has_bssrdf_bump capability. Its
    // interpreter domain is the exact program image of that set. Closure
    // variants still include every physical leaf in each selected topology:
    // non-BSSRDF leaves consume the same finite Cycles closure budget and
    // therefore cannot be erased independently.
    std::vector<std::uint32_t> bssrdf_value_static_variants;
    std::vector<std::uint32_t> bssrdf_closure_static_variants;
    compiler::PrincipledClosureFeatureMask bssrdf_principled_closure_features{};

    // Device projection of the replacement scene. Program descriptors occupy
    // two uint4 words in the exact 32-byte host layout; all instruction-owned
    // offsets are already scene-absolute, so no per-invocation side-range data
    // is uploaded.
    luisa::vector<luisa::uint4> svm_program_descriptors;
    luisa::vector<luisa::uint4> svm_instructions;
    luisa::vector<luisa::uint> svm_value_operands;
    // Exact device projection of the only unbounded static_u0 consumer:
    // Nishita sky indices. Other metadata records carry zero here because no
    // value handler observes their static_u0 through this buffer.
    luisa::vector<luisa::uint> svm_metadata_static_u0;
    luisa::vector<luisa::uint> svm_metadata_parameters;
    luisa::vector<luisa::uint2> svm_metadata_static_ranges;
    luisa::vector<float> svm_static_data;
    luisa::vector<luisa::uint> svm_closure_operands;

    Buffer<luisa::uint4> svm_program_buffer;
    Buffer<luisa::uint4> svm_instruction_buffer;
    Buffer<luisa::uint> svm_value_operand_buffer;
    Buffer<luisa::uint> svm_metadata_static_u0_buffer;
    Buffer<luisa::uint> svm_metadata_parameter_buffer;
    Buffer<luisa::uint2> svm_metadata_static_range_buffer;
    Buffer<float> svm_static_data_buffer;
    Buffer<luisa::uint> svm_closure_operand_buffer;
    // Declared after the bound buffers so reverse member destruction releases
    // the descriptor view before its resources.
    BindlessArray device_view;
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
    Buffer<compiler::cycles_svm::KernelCurve> curve_buffer;
};

struct CyclesSvmObjectRuntime {
    CyclesSvmObjectSceneImage image;
    Buffer<compiler::cycles_svm::KernelObject> object_buffer;
    Buffer<luisa::uint> object_flag_buffer;
};

struct CyclesSvmRuntime {
    compiler::cycles_svm::CompiledShaderTable compilation;
    compiler::cycles_svm::ObjectIdentityPlan object_identities;
    compiler::cycles_svm::ParticleTableImage particles;
    std::map<contract::MaterialId, std::uint32_t>
        material_shader_indices;
    luisa::vector<CyclesSvmImageBindingGpu> image_bindings;
    luisa::vector<CyclesSvmParticleGpu> particle_records;
    std::optional<Buffer<luisa::uint>> word_buffer;
    std::optional<Buffer<float>> ies_buffer;
    std::optional<Buffer<CyclesSvmImageBindingGpu>>
        image_binding_buffer;
    std::optional<Buffer<CyclesSvmParticleGpu>> particle_buffer;
    // Null is the only pre-finalization state. The complete typed image is
    // installed atomically after displacement has fixed the final geometry.
    std::unique_ptr<CyclesSvmGeometryRuntime> geometry;
    std::unique_ptr<CyclesSvmObjectRuntime> objects;
};

struct LuisaSceneData {
    luisa::compute::Device device;
    std::uint64_t revision{};
    MaterialLibrary materials;
    SurfaceDispatch surfaces;
    // Production Cycles-style single shader population. The independent
    // environment opt-out is retained only as an exact expanded-graph A/B
    // oracle for diagnosing interpreter and population effects separately.
    bool populate_surface_once{};
    // Replacement native Cycles bytecode image. During the migration this is
    // built and uploaded transactionally before the legacy surface evaluator
    // is constructed; shade_surface wiring consumes only this image once all
    // closure families are present.
    std::unique_ptr<CyclesSvmRuntime> cycles_svm;
    // Production scene-pruned ShaderGraph -> SVM image. A transactional build
    // failure produces no partial device image; an explicit environment
    // opt-out selects the established expanded diagnostic implementation.
    std::unique_ptr<SurfaceValueRuntime> surface_values;
    // Scene-level Cycles has_bssrdf_bump materials mapped onto the
    // structure-deduplicated SurfaceDispatch tags. Shared tags form only the
    // conservative host/JIT dispatch image; MaterialBindingGpu keeps the
    // exact per-parameter-block predicate used at runtime. Closures and
    // parameters remain device expressions.
    luisa::vector<luisa::uint> surface_bssrdf_bump_tags;
    // Host-stage scene capability used to decide whether the production path
    // kernel must record spatial BSSRDF transport at all. The predicate is
    // derived from the raw SurfaceProgram plus its parameter block; it never
    // rewrites, bakes, or approximates a material closure.
    bool has_subsurface{};
    std::uint32_t subsurface_instance_count{};
    // Exact scene-level shader-raytrace capabilities. They are derived from
    // reachable SurfaceProgram instructions before kernel construction, so a
    // scene without AO neither builds its secondary local TLAS nor records AO
    // traversal callables.
    bool has_ambient_occlusion{};
    bool has_local_ambient_occlusion{};
    // One-element material-like scene data, allocated only when an AO node is
    // reachable. Keeping it out of RenderKernelParameters preserves the exact
    // no-AO kernel ABI and prevents the authored distance from specializing
    // shader cache identity.
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
    float background_portal_weight{};
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
    Buffer<AttributeBindingGpu> attribute_binding_buffer;
    Buffer<AttributeRangeGpu> attribute_range_buffer;
    std::uint32_t attribute_binding_slot{};
    std::uint32_t attribute_range_slot{};
    Buffer<LightGpu> light_buffer;
    std::uint32_t light_count{};
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
    [[nodiscard]] std::size_t
    surface_program_execution_histogram_topology_count() const noexcept;
    void prepare_sobol_table(std::uint32_t total_samples);
    void initialize(const RenderSettings &settings);
    [[nodiscard]] bool write_passes(contract::OutputSink &output);
    void deliver_path_trace();
    void deliver_surface_closure_count_histogram();
    void deliver_surface_program_execution_histogram();

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
