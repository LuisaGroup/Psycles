#pragma once

#include <psycles/luisa/path_tracer.h>

#include <psycles/compiler/material_library.h>
#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/cycles_sampler.h>
#include <psycles/luisa/graph_surface.h>

#include "path_tracer_types.h"
#include "path_tracer_volume_metadata.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
constexpr std::uint32_t geometry_bindless_stride = 9u;
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
constexpr std::uint32_t light_flag_normalize = 1u << 0u;
constexpr std::uint32_t light_flag_ellipse = 1u << 1u;
constexpr std::uint32_t light_flag_sphere = 1u << 2u;
constexpr std::uint32_t light_flag_use_mis = 1u << 3u;
constexpr std::uint32_t light_flag_full_spread = 1u << 4u;
constexpr std::uint32_t light_flag_forward_intersectable =
    1u << 5u;


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
    Mesh mesh;
};

struct AttributeUpload {
    std::uint64_t id{};
    std::uint32_t domain{};
    luisa::vector<luisa::float4> values;
};

struct GeometryUpload {
    std::uint32_t attribute_domains{};
    luisa::float4x4 generated_transform{};
    luisa::vector<luisa::float3> positions;
    luisa::vector<luisa::float3> normals;
    luisa::vector<luisa::float2> uv;
    luisa::vector<luisa::float4> uv_tangents;
    luisa::vector<luisa::float3> generated;
    luisa::vector<Triangle> triangles;
    luisa::vector<luisa::uint> triangle_material_slots;
    luisa::vector<float> triangle_random_per_island;
    luisa::vector<luisa::uint> triangle_smooth;
    std::vector<AttributeUpload> attributes;
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
    std::map<contract::MaterialId, MaterialBinding>
        material_bindings;
    std::optional<MaterialBinding> world_surface;
    std::uint32_t cycles_background_object_index{
        ~std::uint32_t{0u}};
    Buffer<luisa::float4> parameter_buffer;
    Buffer<float> cycles_bsdf_table_buffer;
    std::vector<GeometryResource> geometries;
    std::vector<Image<float>> images;
    BindlessArray texture_heap;
    std::optional<std::uint32_t> environment_texture_slot;
    std::uint32_t environment_width{};
    std::uint32_t environment_height{};
    std::vector<contract::EnvironmentSunDesc> environment_suns;
    std::optional<NishitaEnvironmentRuntime>
        nishita_environment;
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
    float triangle_area_pdf{};
    float light_selection_pdf{};
    bool environment_in_light_distribution{};
    luisa::float3 background{};
    contract::ShaderColorSpace shader_color_space;
    Accel accel;
    CameraDesc camera;
    VolumeSceneMetadata volume_metadata;
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


using RenderShader = Shader1D<
    Buffer<luisa::float4>,
    Buffer<luisa::float4>,
    Buffer<luisa::float4>,
    Buffer<luisa::float4>,
    Buffer<luisa::uint>,
    Buffer<luisa::float4>,
    std::uint32_t,
    std::uint32_t,
    Buffer<luisa::float4>,
    Buffer<float>,
    RenderKernelParameters>;

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
    Buffer<luisa::float4> _path_trace;
    Buffer<luisa::float4> _sobol_table;
    Buffer<float> _pixel_filter_table;
    std::uint32_t _sobol_sequence_size{};
    std::uint32_t _total_aa_samples{};
    RenderKernelParameters _kernel_parameters{};
    RenderShader _render_shader;
    bool _path_trace_delivered{false};
    std::atomic_bool _cancelled{false};

private:
    [[nodiscard]] std::size_t pixel_count() const noexcept;
    void prepare_sobol_table(std::uint32_t total_samples);
    void initialize(const RenderSettings &settings);
    void write_passes(contract::OutputSink &output);
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
[[nodiscard]] Var<SurfaceAovCall> pack_surface_aov(
    const SurfaceAov &aov) noexcept;
[[nodiscard]] SurfaceAov unpack_surface_aov(
    const Var<SurfaceAovCall> &aov) noexcept;
[[nodiscard]] luisa::float3 to_luisa(Vec3f value) noexcept;
[[nodiscard]] luisa::float2 to_luisa(Vec2f value) noexcept;
[[nodiscard]] luisa::float4x4 to_luisa(Mat4f value) noexcept;
[[nodiscard]] Vec3f matrix_axis(
    const Mat4f &matrix,
    std::size_t column) noexcept;
[[nodiscard]] Vec3f matrix_translation(
    const Mat4f &matrix) noexcept;
[[nodiscard]] luisa::float4 parameter_value(
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
