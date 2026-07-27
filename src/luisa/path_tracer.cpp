#include <psycles/luisa/path_tracer.h>

#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/material_library.h>
#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/graph_surface.h>

#include "cycles_shader_tables_4_5_10.inl"

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

#include <stb/stb_image.h>

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

struct GeometryGpu {
    luisa::uint bindless_base{};
    luisa::uint material_offset{};
    luisa::uint material_count{};
    luisa::uint padding{};
};

struct InstanceGpu {
    luisa::uint geometry_index{};
    luisa::uint override_offset{};
    luisa::uint override_count{};
    float object_random{};
    luisa::uint particle_index{};
};

struct LightGpu {
    luisa::uint type{};
    luisa::float3 position{};
    luisa::float3 axis_x{};
    luisa::float3 axis_y{};
    luisa::float3 axis_z{};
    luisa::float3 color{};
    float power{};
    float size{};
    float spread{};
    float padding{};
};

struct EmissiveTriangleGpu {
    luisa::uint instance_index{};
    luisa::uint geometry_index{};
    luisa::uint primitive_index{};
    luisa::uint surface_tag{};
};

}// namespace psycles::luisa_backend::detail

LUISA_STRUCT(
    psycles::luisa_backend::detail::GeometryGpu,
    bindless_base,
    material_offset,
    material_count,
    padding) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::InstanceGpu,
    geometry_index,
    override_offset,
    override_count,
    object_random,
    particle_index) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::LightGpu,
    type,
    position,
    axis_x,
    axis_y,
    axis_z,
    color,
    power,
    size,
    spread,
    padding) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::EmissiveTriangleGpu,
    instance_index,
    geometry_index,
    primitive_index,
    surface_tag) {};

namespace psycles::luisa_backend {

namespace {

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
using detail::GeometryGpu;
using detail::EmissiveTriangleGpu;
using detail::InstanceGpu;
using detail::LightGpu;
using luisa::compute::Accel;
using luisa::compute::AccelVar;
using luisa::compute::BindlessArray;
using luisa::compute::Bool;
using luisa::compute::Buffer;
using luisa::compute::BufferFloat4;
using luisa::compute::BufferUInt;
using luisa::compute::Callable;
using luisa::compute::Expr;
using luisa::compute::Float;
using luisa::compute::Float2;
using luisa::compute::Float3;
using luisa::compute::Float4;
using luisa::compute::Kernel1D;
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
using luisa::compute::dispatch_x;
using luisa::compute::dot;
using luisa::compute::inverse;
using luisa::compute::length_squared;
using luisa::compute::make_float2;
using luisa::compute::make_float3;
using luisa::compute::make_float4;
using luisa::compute::make_float4x4;
using luisa::compute::make_ray;
using luisa::compute::max;
using luisa::compute::min;
using luisa::compute::normalize;
using luisa::compute::offset_ray_origin;
using luisa::compute::select;
using luisa::compute::synchronize;
using luisa::compute::transpose;
using luisa::compute::triangle_interpolate;

constexpr auto pi = 3.14159265358979323846f;
constexpr auto uniform_sphere_pdf = 1.0f / (4.0f * pi);
constexpr auto ray_maximum = 1.0e30f;
constexpr std::uint32_t geometry_bindless_stride = 8u;
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

[[nodiscard]] luisa::float3 to_luisa(Vec3f value) noexcept {
    return luisa::make_float3(value.x, value.y, value.z);
}

[[nodiscard]] luisa::float2 to_luisa(Vec2f value) noexcept {
    return luisa::make_float2(value.x, value.y);
}

[[nodiscard]] luisa::float4x4 to_luisa(Mat4f value) noexcept {
    const auto &e = value.elements;
    return make_float4x4(
        luisa::make_float4(e[0u], e[1u], e[2u], e[3u]),
        luisa::make_float4(e[4u], e[5u], e[6u], e[7u]),
        luisa::make_float4(e[8u], e[9u], e[10u], e[11u]),
        luisa::make_float4(e[12u], e[13u], e[14u], e[15u]));
}

[[nodiscard]] Vec3f matrix_axis(
    const Mat4f &matrix,
    std::size_t column) noexcept {
    const auto offset = column * 4u;
    return {
        matrix.elements[offset],
        matrix.elements[offset + 1u],
        matrix.elements[offset + 2u]};
}

[[nodiscard]] Vec3f matrix_translation(
    const Mat4f &matrix) noexcept {
    return {
        matrix.elements[12u],
        matrix.elements[13u],
        matrix.elements[14u]};
}

[[nodiscard]] luisa::float4 parameter_value(
    const contract::SocketValue &value) noexcept {
    using contract::SocketType;
    switch (value.type) {
        case SocketType::boolean:
            return luisa::make_float4(
                std::get<bool>(value.value) ? 1.0f : 0.0f);
        case SocketType::integer:
            return luisa::make_float4(
                static_cast<float>(
                    std::get<std::int64_t>(value.value)));
        case SocketType::unsigned_integer:
            return luisa::make_float4(
                static_cast<float>(
                    std::get<std::uint64_t>(value.value)));
        case SocketType::floating:
            return luisa::make_float4(
                std::get<float>(value.value));
        case SocketType::float2: {
            const auto v = std::get<Vec2f>(value.value);
            return luisa::make_float4(v.x, v.y, 0.0f, 0.0f);
        }
        case SocketType::float3:
        case SocketType::color:
        case SocketType::spectrum:
        case SocketType::point:
        case SocketType::vector:
        case SocketType::normal: {
            const auto v = std::get<Vec3f>(value.value);
            return luisa::make_float4(v.x, v.y, v.z, 0.0f);
        }
        default:
            return luisa::make_float4(0.0f);
    }
}

struct AttributeBinding {
    std::uint64_t id{};
    std::uint32_t geometry_index{};
    std::uint32_t triangle_slot{};
    std::uint32_t value_slot{};
};

class BufferShaderServices final : public ShaderServices {

private:
    const Buffer<luisa::float4> &_parameters;
    const Buffer<float> &_cycles_bsdf_tables;
    const BindlessArray &_textures;
    const BindlessArray &_geometry_heap;
    const std::vector<AttributeBinding> &_attributes;

public:
    explicit BufferShaderServices(
        const Buffer<luisa::float4> &parameters,
        const Buffer<float> &cycles_bsdf_tables,
        const BindlessArray &textures,
        const BindlessArray &geometry_heap,
        const std::vector<AttributeBinding> &attributes) noexcept
        : _parameters{parameters},
          _cycles_bsdf_tables{cycles_bsdf_tables},
          _textures{textures},
          _geometry_heap{geometry_heap},
          _attributes{attributes} {}

    [[nodiscard]] Float4 texture_2d(
        Expr<std::uint32_t> handle,
        Expr<luisa::float2> uv,
        Expr<luisa::float2>,
        Expr<luisa::float2>) const noexcept override {
        return _textures->tex2d(handle).sample(uv);
    }

    [[nodiscard]] Float4 attribute(
        Expr<std::uint64_t> attribute_id,
        const SurfacePoint &point) const noexcept override {
        Float4 result = make_float4(0.0f);
        for (const auto &binding : _attributes) {
            $if ((attribute_id == binding.id) &
                 (point.geometry_index ==
                  binding.geometry_index)) {
                Var<Triangle> triangle =
                    _geometry_heap
                        ->buffer<Triangle>(
                            binding.triangle_slot)
                        .read(point.primitive_id);
                auto v0 =
                    _geometry_heap
                        ->buffer<luisa::float4>(
                            binding.value_slot)
                        .read(triangle.i0);
                auto v1 =
                    _geometry_heap
                        ->buffer<luisa::float4>(
                            binding.value_slot)
                        .read(triangle.i1);
                auto v2 =
                    _geometry_heap
                        ->buffer<luisa::float4>(
                            binding.value_slot)
                        .read(triangle.i2);
                result = triangle_interpolate(
                    point.barycentric, v0, v1, v2);
            };
        }
        return result;
    }

    [[nodiscard]] Float parameter_float(
        std::uint32_t block,
        std::uint32_t slot) const noexcept override {
        return _parameters->read(block + slot).x;
    }

    [[nodiscard]] Float3 parameter_float3(
        std::uint32_t block,
        std::uint32_t slot) const noexcept override {
        return _parameters->read(block + slot).xyz();
    }

    [[nodiscard]] Float cycles_bsdf_data(
        Expr<std::uint32_t> index) const noexcept override {
        return _cycles_bsdf_tables->read(index);
    }
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
    std::vector<Buffer<luisa::float4>> color_attributes;
    Mesh mesh;
};

struct ColorAttributeUpload {
    std::uint64_t id{};
    luisa::vector<luisa::float4> values;
};

struct GeometryUpload {
    luisa::vector<luisa::float3> positions;
    luisa::vector<luisa::float3> normals;
    luisa::vector<luisa::float2> uv;
    luisa::vector<luisa::float4> uv_tangents;
    luisa::vector<luisa::float3> generated;
    luisa::vector<Triangle> triangles;
    luisa::vector<luisa::uint> triangle_material_slots;
    luisa::vector<float> triangle_random_per_island;
    std::vector<ColorAttributeUpload> color_attributes;
};

struct LuisaSceneData {
    luisa::compute::Device device;
    std::uint64_t revision{};
    MaterialLibrary materials;
    SurfaceDispatch surfaces;
    std::map<contract::MaterialId, std::uint32_t> material_tags;
    std::optional<std::uint32_t> world_surface_tag;
    Buffer<luisa::float4> parameter_buffer;
    Buffer<float> cycles_bsdf_table_buffer;
    std::vector<GeometryResource> geometries;
    std::vector<Image<float>> images;
    BindlessArray texture_heap;
    std::optional<std::uint32_t> environment_texture_slot;
    std::uint32_t environment_width{};
    std::uint32_t environment_height{};
    std::vector<contract::EnvironmentSunDesc> environment_suns;
    BindlessArray heap;
    Buffer<GeometryGpu> geometry_buffer;
    Buffer<InstanceGpu> instance_buffer;
    Buffer<luisa::uint> geometry_material_buffer;
    Buffer<luisa::uint> override_material_buffer;
    std::vector<AttributeBinding> attribute_bindings;
    Buffer<LightGpu> light_buffer;
    std::uint32_t light_count{};
    Buffer<EmissiveTriangleGpu> emissive_triangle_buffer;
    std::uint32_t emissive_triangle_count{};
    luisa::float3 background{};
    Accel accel;
    CameraDesc camera;
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

[[nodiscard]] PixelWindow effective_window(
    const RenderSettings &settings) noexcept {
    if (settings.window.width != 0u &&
        settings.window.height != 0u) {
        return settings.window;
    }
    return {
        .x = 0u,
        .y = 0u,
        .width = settings.full_extent.width,
        .height = settings.full_extent.height};
}

[[nodiscard]] std::uint32_t pass_channels(
    const PassRequest &pass) noexcept {
    return std::max(pass.channels, 1u);
}

[[nodiscard]] bool supported_pass(PassKind pass) noexcept {
    switch (pass) {
        case PassKind::combined:
        case PassKind::normal:
        case PassKind::albedo:
        case PassKind::denoising_normal:
        case PassKind::denoising_albedo:
        case PassKind::sample_count:
            return true;
        default:
            return false;
    }
}

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
    Buffer<luisa::uint> _sample_count;
    Shader1D<
        Buffer<luisa::float4>,
        Buffer<luisa::float4>,
        Buffer<luisa::float4>,
        Buffer<luisa::uint>,
        std::uint32_t,
        std::uint32_t,
        std::uint32_t>
        _render_shader;
    std::atomic_bool _cancelled{false};

private:
    [[nodiscard]] std::size_t pixel_count() const noexcept {
        return static_cast<std::size_t>(_window.width) *
               static_cast<std::size_t>(_window.height);
    }

    void initialize(const RenderSettings &settings) {
        _settings = settings;
        _window = effective_window(settings);
        const auto count = std::max<std::size_t>(pixel_count(), 1u);
        _combined =
            _scene->device.create_buffer<luisa::float4>(count);
        _normal =
            _scene->device.create_buffer<luisa::float4>(count);
        _albedo =
            _scene->device.create_buffer<luisa::float4>(count);
        _sample_count =
            _scene->device.create_buffer<luisa::uint>(count);

        luisa::vector<luisa::float4> zeros_float(count);
        luisa::vector<luisa::uint> zeros_uint(count);
        _stream << _combined.copy_from(luisa::span{zeros_float})
                << _normal.copy_from(luisa::span{zeros_float})
                << _albedo.copy_from(luisa::span{zeros_float})
                << _sample_count.copy_from(luisa::span{zeros_uint})
                << synchronize();

        auto scene = _scene;
        const auto render_settings = _settings;
        const auto render_window = _window;
        const auto integrator = render_settings.integrator;
        const auto max_bounces = integrator.max_bounces;
        const auto min_bounces = integrator.min_bounces;
        const auto max_diffuse_bounces =
            integrator.diffuse_bounces;
        const auto max_glossy_bounces =
            integrator.glossy_bounces;
        const auto max_transmission_bounces =
            integrator.transmission_bounces;
        const auto transparent_min_bounces =
            integrator.transparent_min_bounces;
        const auto transparent_max_bounces =
            integrator.transparent_max_bounces;
        // Blender exposes clamp per RGB channel. Cycles' device kernel
        // compares the sum of absolute RGB components, so scene sync
        // multiplies non-zero UI values by three.
        const auto sample_clamp_direct =
            integrator.sample_clamp_direct > 0.0f
                ? integrator.sample_clamp_direct * 3.0f
                : 0.0f;
        const auto sample_clamp_indirect =
            integrator.sample_clamp_indirect > 0.0f
                ? integrator.sample_clamp_indirect * 3.0f
                : 0.0f;
        const auto light_inv_rr_threshold =
            !integrator.use_light_tree &&
                    integrator.light_sampling_threshold > 0.0f
                ? integrator.film_exposure /
                      integrator.light_sampling_threshold
                : 0.0f;
        const auto reflective_caustics =
            integrator.reflective_caustics;
        const auto refractive_caustics =
            integrator.refractive_caustics;
        const auto max_path_steps = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(
                static_cast<std::uint64_t>(
                    std::max(max_bounces, 1u)) +
                    static_cast<std::uint64_t>(std::max(
                        transparent_max_bounces, 1u)) +
                    1u,
                1024u));
        const auto next_event_estimation =
            _options.next_event_estimation &&
            integrator.direct_light_sampling !=
                contract::DirectLightSampling::
                    forward_path_tracing;
        const auto direct_light_sampling =
            next_event_estimation
                ? integrator.direct_light_sampling
                : contract::DirectLightSampling::
                      forward_path_tracing;
        const auto camera_transform = to_luisa(scene->camera.transform);
        const auto camera_projection = scene->camera.projection;
        const auto camera_aspect =
            static_cast<float>(std::max(
                render_settings.full_extent.width, 1u)) /
            static_cast<float>(std::max(
                render_settings.full_extent.height, 1u));
        const auto camera_horizontal_fit =
            scene->camera.sensor_fit ==
                CameraSensorFit::horizontal ||
            (scene->camera.sensor_fit ==
                 CameraSensorFit::automatic &&
             camera_aspect >= 1.0f);
        const auto camera_vertical_tangent =
            camera_horizontal_fit
                ? std::tan(
                      scene->camera.horizontal_field_of_view *
                      0.5f) /
                      camera_aspect
                : std::tan(
                      scene->camera.field_of_view * 0.5f);
        const auto camera_horizontal_tangent =
            camera_vertical_tangent * camera_aspect;
        const auto camera_ortho_scale =
            scene->camera.orthographic_scale;
        const auto camera_shift_x =
            scene->camera.lens_shift_x *
            (camera_horizontal_fit
                 ? 1.0f
                 : 1.0f / camera_aspect);
        const auto camera_shift_y =
            scene->camera.lens_shift_y *
            (camera_horizontal_fit
                 ? camera_aspect
                 : 1.0f);
        const auto camera_near = scene->camera.near_clip;
        const auto camera_far = scene->camera.far_clip;
        const auto camera_aperture_radius =
            scene->camera.aperture_radius;
        const auto camera_focal_distance =
            scene->camera.focal_distance;
        const auto camera_aperture_ratio =
            scene->camera.aperture_ratio;
        const auto background = scene->background;
        const auto pixel_filter =
            render_settings.pixel_filter;
        const auto filter_width =
            std::max(render_settings.filter_width, 1.0e-5f);
        const auto pass_alpha_threshold = std::clamp(
            render_settings.pass_alpha_threshold,
            0.0f,
            1.0f);

        Callable random_float = [](UInt &state) noexcept {
            state = state * 747796405u + 2891336453u;
            UInt word =
                ((state >> ((state >> 28u) + 4u)) ^ state) *
                277803737u;
            word = (word >> 22u) ^ word;
            return cast<float>(word >> 8u) *
                   (1.0f / 16777216.0f);
        };
        Callable hash_seed = [](
                                 UInt pixel,
                                 UInt sample,
                                 UInt seed) noexcept {
            UInt value =
                pixel * 0x9e3779b9u ^
                sample * 0x85ebca6bu ^
                seed * 0xc2b2ae35u;
            value ^= value >> 16u;
            value *= 0x7feb352du;
            value ^= value >> 15u;
            value *= 0x846ca68bu;
            value ^= value >> 16u;
            return value;
        };
        Callable sample_box_filter =
            [filter_width](Float u) noexcept {
                return 0.5f +
                       (u - 0.5f) * filter_width;
            };
        Callable sample_gaussian_filter =
            [filter_width](Float u) noexcept {
                // Cycles' Gaussian filter is exp(-2*x*x) over
                // [-width/2, width/2]. Invert its normalized CDF
                // numerically so raster sampling remains entirely in
                // the Luisa program.
                const auto radius = filter_width * 0.5f;
                const auto target =
                    abs(u * 2.0f - 1.0f);
                Float x = radius * target;
                Float total = 0.0f;
                Float integral = 0.0f;
                constexpr auto integration_steps = 32u;
                $for (i, integration_steps) {
                    const auto x0 =
                        radius *
                        cast<float>(i) /
                        static_cast<float>(
                            integration_steps);
                    const auto x1 =
                        radius *
                        cast<float>(i + 1u) /
                        static_cast<float>(
                            integration_steps);
                    total +=
                        0.5f * (x1 - x0) *
                        (exp(-2.0f * x0 * x0) +
                         exp(-2.0f * x1 * x1));
                };
                $for (iteration, 8u) {
                    static_cast<void>(iteration);
                    integral = 0.0f;
                    $for (i, integration_steps) {
                        const auto x0 =
                            x *
                            cast<float>(i) /
                            static_cast<float>(
                                integration_steps);
                        const auto x1 =
                            x *
                            cast<float>(i + 1u) /
                            static_cast<float>(
                                integration_steps);
                        integral +=
                            0.5f * (x1 - x0) *
                            (exp(-2.0f * x0 * x0) +
                             exp(-2.0f * x1 * x1));
                    };
                    x = clamp(
                        x -
                            (integral - target * total) /
                                max(
                                    exp(-2.0f * x * x),
                                    1.0e-7f),
                        0.0f,
                        radius);
                };
                return 0.5f +
                       select(-x, x, u >= 0.5f);
            };
        Callable sample_blackman_harris_filter =
            [filter_width](Float u) noexcept {
                // This is the continuous form of Cycles'
                // BLACKMAN_HARRIS inverse-CDF table. Cycles doubles
                // filter_width before building this symmetric filter,
                // so the positive support radius is filter_width.
                constexpr auto a0 = 0.35875f;
                constexpr auto a1 = 0.48829f;
                constexpr auto a2 = 0.14128f;
                constexpr auto a3 = 0.01168f;
                constexpr auto two_pi =
                    6.28318530717958647692f;
                const auto radius = filter_width;
                const auto full_width = radius * 2.0f;
                const auto target_fraction =
                    abs(u * 2.0f - 1.0f);
                const auto target =
                    target_fraction * a0 * radius;
                Float x = radius * target_fraction;
                $for (iteration, 8u) {
                    static_cast<void>(iteration);
                    const auto phase =
                        two_pi * x / full_width;
                    const auto integral =
                        a0 * x +
                        a1 * full_width / two_pi *
                            sin(phase) +
                        a2 * full_width /
                            (2.0f * two_pi) *
                            sin(2.0f * phase) +
                        a3 * full_width /
                            (3.0f * two_pi) *
                            sin(3.0f * phase);
                    const auto density =
                        a0 + a1 * cos(phase) +
                        a2 * cos(2.0f * phase) +
                        a3 * cos(3.0f * phase);
                    x = clamp(
                        x -
                            (integral - target) /
                                max(density, 1.0e-7f),
                        0.0f,
                        radius);
                };
                return 0.5f +
                       select(-x, x, u >= 0.5f);
            };
        Callable sample_pixel_filter =
            [=](Float u) noexcept {
                if (pixel_filter ==
                    contract::PixelFilter::box) {
                    return sample_box_filter(u);
                }
                if (pixel_filter ==
                    contract::PixelFilter::gaussian) {
                    return sample_gaussian_filter(u);
                }
                return sample_blackman_harris_filter(u);
            };
        Callable safe_normalize = [](
                                      Float3 value,
                                      Float3 fallback) noexcept {
            auto valid = dot(value, value) > 1.0e-20f;
            auto selected = select(fallback, value, valid);
            return normalize(select(
                make_float3(0.0f, 0.0f, 1.0f),
                selected,
                dot(selected, selected) > 1.0e-20f));
        };
        Callable power_heuristic = [](
                                       Float sampled_pdf,
                                       Float other_pdf) noexcept {
            auto sampled_squared = sampled_pdf * sampled_pdf;
            auto other_squared = other_pdf * other_pdf;
            return sampled_squared /
                   max(
                       sampled_squared + other_squared,
                       1.0e-20f);
        };
        Callable forward_light_weight =
            [=](Float forward_pdf,
                Float nee_pdf,
                Bool competing,
                Bool nee_available) noexcept {
                if (direct_light_sampling ==
                    contract::DirectLightSampling::
                        forward_path_tracing) {
                    return Float{1.0f};
                }
                if (direct_light_sampling ==
                    contract::DirectLightSampling::
                        next_event_estimation) {
                    return select(
                        1.0f,
                        0.0f,
                        competing & nee_available);
                }
                return select(
                    1.0f,
                    power_heuristic(
                        forward_pdf, nee_pdf),
                    competing & nee_available);
            };
        Callable nee_light_weight =
            [=](Float nee_pdf,
                Float forward_pdf) noexcept {
                if (direct_light_sampling ==
                    contract::DirectLightSampling::
                        next_event_estimation) {
                    return Float{1.0f};
                }
                if (direct_light_sampling ==
                    contract::DirectLightSampling::
                        forward_path_tracing) {
                    return select(
                        0.0f,
                        1.0f,
                        forward_pdf <= 0.0f);
                }
                return power_heuristic(
                    nee_pdf, forward_pdf);
            };
        Callable clamp_light_contribution =
            [=](Float3 contribution, UInt depth) noexcept {
                Float limit = select(
                    sample_clamp_direct,
                    sample_clamp_indirect,
                    depth > 0u);
                Float magnitude =
                    abs(contribution.x) +
                    abs(contribution.y) +
                    abs(contribution.z);
                Bool should_clamp =
                    (limit > 0.0f) & (magnitude > limit);
                return select(
                    contribution,
                    contribution *
                        (limit / max(magnitude, 1.0e-20f)),
                    should_clamp);
            };
        Callable light_sample_roulette_weight =
            [=](Float3 unshadowed_contribution,
                Float random) noexcept {
                Float maximum = max(
                    abs(unshadowed_contribution.x),
                    max(
                        abs(unshadowed_contribution.y),
                        abs(unshadowed_contribution.z)));
                Float probability =
                    maximum * light_inv_rr_threshold;
                Bool roulette =
                    (light_inv_rr_threshold > 0.0f) &
                    (probability < 1.0f);
                Bool survives =
                    (!roulette) | (random < probability);
                Float inverse_probability = select(
                    1.0f,
                    1.0f / max(probability, 1.0e-20f),
                    roulette);
                return select(
                    0.0f,
                    inverse_probability,
                    survives);
            };

        Kernel1D kernel = [=, &surfaces = scene->surfaces](
                              BufferFloat4 combined,
                              BufferFloat4 normal,
                              BufferFloat4 albedo,
                              BufferUInt sample_count,
                              UInt sample_first,
                              UInt samples,
                              UInt seed) noexcept {
            UInt pixel = dispatch_x();
            UInt local_x =
                pixel % static_cast<std::uint32_t>(
                            std::max(render_window.width, 1u));
            UInt local_y =
                pixel / static_cast<std::uint32_t>(
                            std::max(render_window.width, 1u));
            UInt full_x = local_x + render_window.x;
            UInt full_y = local_y + render_window.y;
            Float4 combined_sum = combined.read(pixel);
            Float4 normal_sum = normal.read(pixel);
            Float4 albedo_sum = albedo.read(pixel);
            UInt completed = sample_count.read(pixel);

            BufferShaderServices services{
                scene->parameter_buffer,
                scene->cycles_bsdf_table_buffer,
                scene->texture_heap,
                scene->heap,
                scene->attribute_bindings};
            SurfaceQuery surface_query{
                .lobe_mask =
                    static_cast<std::uint32_t>(
                        contract::event_diffuse |
                        contract::event_glossy |
                        contract::event_singular |
                        contract::event_reflection |
                        contract::event_transmission |
                        contract::event_transparent),
                .transport_mode =
                    static_cast<std::uint32_t>(
                        contract::TransportMode::radiance)};
            auto evaluate_world_graph =
                [&](Float3 direction) noexcept {
                    Float3 world = background;
                    if (scene->world_surface_tag) {
                        SurfacePoint world_point{
                            .position = make_float3(0.0f),
                            .object_position =
                                make_float3(0.0f),
                            .object_location =
                                make_float3(0.0f),
                            .generated = make_float3(0.0f),
                            .geometric_normal = -direction,
                            .shading_normal = -direction,
                            .dpdu = make_float3(
                                1.0f, 0.0f, 0.0f),
                            .dpdv = make_float3(
                                0.0f, 1.0f, 0.0f),
                            .dPdx = make_float3(0.0f),
                            .dPdy = make_float3(0.0f),
                            .object_dPdx = make_float3(0.0f),
                            .object_dPdy = make_float3(0.0f),
                            .generated_dx = make_float3(0.0f),
                            .generated_dy = make_float3(0.0f),
                            .incoming = -direction,
                            .uv = make_float2(0.0f),
                            .uv_dx = make_float2(0.0f),
                            .uv_dy = make_float2(0.0f),
                            .geometry_index = ~0u,
                            .barycentric =
                                make_float2(0.0f),
                            .barycentric_dx =
                                make_float2(0.0f),
                            .barycentric_dy =
                                make_float2(0.0f),
                            .instance_id = 0u,
                            .primitive_id = 0u,
                            .object_random = 0.0f,
                            .particle_index = 0u,
                            .random_per_island = 0.0f,
                            .ray_visibility =
                                camera_visibility,
                            .ray_events = 0u,
                            .ray_depth = 0u,
                            .diffuse_depth = 0u,
                            .glossy_depth = 0u,
                            .transparent_depth = 0u,
                            .transmission_depth = 0u,
                            .ray_length = 0.0f,
                            .time = 0.0f,
                            .back_facing = false};
                        world += surfaces.emission(
                            UInt{*scene->world_surface_tag},
                            services,
                            world_point,
                            -direction);
                    }
                    return world;
                };
            auto evaluate_environment_base =
                [&](Float3 direction) noexcept {
                    if (scene->environment_texture_slot) {
                        auto u = fract(
                            (pi - atan2(direction.y, direction.x)) /
                            (2.0f * pi));
                        auto half_texel_y =
                            0.5f /
                            static_cast<float>(std::max(
                                scene->environment_height, 1u));
                        auto v = clamp(
                            acos(clamp(direction.z, -1.0f, 1.0f)) /
                                pi,
                            half_texel_y,
                            1.0f - half_texel_y);
                        return scene->texture_heap
                            ->tex2d(*scene->environment_texture_slot)
                            .sample(make_float2(u, v))
                            .xyz();
                    }
                    return evaluate_world_graph(direction);
                };
            auto evaluate_environment_sun =
                [&](Float3 direction,
                    const contract::EnvironmentSunDesc &sun) noexcept {
                    Float3 axis = safe_normalize(
                        to_luisa(sun.direction),
                        make_float3(0.0f, 0.0f, 1.0f));
                    auto cosine = clamp(
                        dot(direction, axis), -1.0f, 1.0f);
                    auto radius = std::max(
                        sun.angular_radius, 1.0e-7f);
                    auto radial_distance =
                        acos(cosine) / radius;
                    auto limb =
                        0.4f +
                        0.6f *
                            sqrt(max(
                                1.0f -
                                    radial_distance *
                                        radial_distance,
                                0.0f));
                    auto inside =
                        cosine >= std::cos(sun.angular_radius);
                    return select(
                        make_float3(0.0f),
                        to_luisa(sun.radiance) *
                            (limb / 0.8f),
                        inside);
                };
            auto trace_shadow =
                [&](Var<luisa::compute::Ray> shadow_ray) noexcept {
                    Float3 transmittance = make_float3(1.0f);
                    auto committed =
                        scene->accel
                            ->traverse(
                                shadow_ray,
                                {.visibility_mask =
                                     shadow_visibility})
                            .on_surface_candidate(
                                [&](luisa::compute::SurfaceCandidate
                                        &candidate) noexcept {
                                    auto hit = candidate.hit();
                                    Var<InstanceGpu> instance =
                                        scene->instance_buffer->read(
                                            hit->inst);
                                    Var<GeometryGpu> geometry =
                                        scene->geometry_buffer->read(
                                            instance.geometry_index);
                                    Var<Triangle> triangle =
                                        scene->heap
                                            ->buffer<Triangle>(
                                                geometry.bindless_base)
                                            .read(hit->prim);
                                    Float3 p0 =
                                        scene->heap
                                            ->buffer<luisa::float3>(
                                                geometry.bindless_base +
                                                1u)
                                            .read(triangle.i0);
                                    Float3 p1 =
                                        scene->heap
                                            ->buffer<luisa::float3>(
                                                geometry.bindless_base +
                                                1u)
                                            .read(triangle.i1);
                                    Float3 p2 =
                                        scene->heap
                                            ->buffer<luisa::float3>(
                                                geometry.bindless_base +
                                                1u)
                                            .read(triangle.i2);
                                    Float3 n0 =
                                        scene->heap
                                            ->buffer<luisa::float3>(
                                                geometry.bindless_base +
                                                2u)
                                            .read(triangle.i0);
                                    Float3 n1 =
                                        scene->heap
                                            ->buffer<luisa::float3>(
                                                geometry.bindless_base +
                                                2u)
                                            .read(triangle.i1);
                                    Float3 n2 =
                                        scene->heap
                                            ->buffer<luisa::float3>(
                                                geometry.bindless_base +
                                                2u)
                                            .read(triangle.i2);
                                    Float2 uv0 =
                                        scene->heap
                                            ->buffer<luisa::float2>(
                                                geometry.bindless_base +
                                                3u)
                                            .read(triangle.i0);
                                    Float2 uv1 =
                                        scene->heap
                                            ->buffer<luisa::float2>(
                                                geometry.bindless_base +
                                                3u)
                                            .read(triangle.i1);
                                    Float2 uv2 =
                                        scene->heap
                                            ->buffer<luisa::float2>(
                                                geometry.bindless_base +
                                                3u)
                                            .read(triangle.i2);
                                    Float4 tangent0 =
                                        scene->heap
                                            ->buffer<luisa::float4>(
                                                geometry.bindless_base +
                                                7u)
                                            .read(triangle.i0);
                                    Float4 tangent1 =
                                        scene->heap
                                            ->buffer<luisa::float4>(
                                                geometry.bindless_base +
                                                7u)
                                            .read(triangle.i1);
                                    Float4 tangent2 =
                                        scene->heap
                                            ->buffer<luisa::float4>(
                                                geometry.bindless_base +
                                                7u)
                                            .read(triangle.i2);
                                    Float3 generated0 =
                                        scene->heap
                                            ->buffer<luisa::float3>(
                                                geometry.bindless_base +
                                                5u)
                                            .read(triangle.i0);
                                    Float3 generated1 =
                                        scene->heap
                                            ->buffer<luisa::float3>(
                                                geometry.bindless_base +
                                                5u)
                                            .read(triangle.i1);
                                    Float3 generated2 =
                                        scene->heap
                                            ->buffer<luisa::float3>(
                                                geometry.bindless_base +
                                                5u)
                                            .read(triangle.i2);
                                    Float random_per_island =
                                        scene->heap
                                            ->buffer<float>(
                                                geometry.bindless_base +
                                                6u)
                                            .read(hit->prim);
                                    UInt material_slot =
                                        scene->heap
                                            ->buffer<luisa::uint>(
                                                geometry.bindless_base +
                                                4u)
                                            .read(hit->prim);

                                    auto object_to_world =
                                        scene->accel
                                            ->instance_transform(
                                                hit->inst);
                                    auto normal_to_world =
                                        transpose(inverse(
                                            object_to_world));
                                    Float3 wp0 =
                                        (object_to_world *
                                         make_float4(p0, 1.0f))
                                            .xyz();
                                    Float3 wp1 =
                                        (object_to_world *
                                         make_float4(p1, 1.0f))
                                            .xyz();
                                    Float3 wp2 =
                                        (object_to_world *
                                         make_float4(p2, 1.0f))
                                            .xyz();
                                    auto candidate_ray =
                                        candidate.ray();
                                    Float3 geometric_normal =
                                        safe_normalize(
                                            cross(
                                                wp1 - wp0,
                                                wp2 - wp0),
                                            -candidate_ray
                                                 ->direction());
                                    Float3 object_shading_normal =
                                        triangle_interpolate(
                                            hit->bary,
                                            n0,
                                            n1,
                                            n2);
                                    Float4 object_tangent =
                                        triangle_interpolate(
                                            hit->bary,
                                            tangent0,
                                            tangent1,
                                            tangent2);
                                    Float3 shading_normal =
                                        safe_normalize(
                                            (normal_to_world *
                                             make_float4(
                                                 object_shading_normal,
                                                 0.0f))
                                                .xyz(),
                                            geometric_normal);
                                    Bool back_facing =
                                        dot(
                                            geometric_normal,
                                            -candidate_ray
                                                 ->direction()) <
                                        0.0f;
                                    geometric_normal = select(
                                        geometric_normal,
                                        -geometric_normal,
                                        back_facing);
                                    shading_normal = select(
                                        shading_normal,
                                        -shading_normal,
                                        back_facing);
                                    shading_normal = select(
                                        shading_normal,
                                        -shading_normal,
                                        dot(
                                            shading_normal,
                                            geometric_normal) <
                                            0.0f);
                                    Float3 position =
                                        candidate_ray->origin() +
                                        candidate_ray->direction() *
                                            hit->committed_ray_t;
                                    Float3 tangent =
                                        safe_normalize(
                                            (object_to_world *
                                             make_float4(
                                                 object_tangent.xyz(),
                                                 0.0f))
                                                .xyz(),
                                            safe_normalize(
                                                (wp1 - wp0) -
                                                    geometric_normal *
                                                        dot(
                                                            wp1 - wp0,
                                                            geometric_normal),
                                                make_float3(
                                                    1.0f,
                                                    0.0f,
                                                    0.0f)));
                                    UInt surface_tag =
                                        scene
                                            ->geometry_material_buffer
                                            ->read(
                                                geometry.material_offset +
                                                min(
                                                    material_slot,
                                                    max(
                                                        geometry
                                                            .material_count,
                                                        1u) -
                                                        1u));
                                    $if (material_slot <
                                         instance.override_count) {
                                        surface_tag =
                                            scene
                                                ->override_material_buffer
                                                ->read(
                                                    instance
                                                        .override_offset +
                                                    material_slot);
                                    };
                                    SurfacePoint point{
                                        .position = position,
                                        .object_position =
                                            triangle_interpolate(
                                                hit->bary,
                                                p0,
                                                p1,
                                                p2),
                                        .object_location =
                                            (object_to_world *
                                             make_float4(
                                                 0.0f,
                                                 0.0f,
                                                 0.0f,
                                                 1.0f))
                                                .xyz(),
                                        .generated =
                                            triangle_interpolate(
                                                hit->bary,
                                                generated0,
                                                generated1,
                                                generated2),
                                        .geometric_normal =
                                            geometric_normal,
                                        .shading_normal =
                                            shading_normal,
                                        .object_shading_normal =
                                            object_shading_normal,
                                        .object_tangent =
                                            object_tangent.xyz(),
                                        .tangent_sign =
                                            object_tangent.w,
                                        .normal_to_world_x =
                                            (normal_to_world *
                                             make_float4(
                                                 1.0f,
                                                 0.0f,
                                                 0.0f,
                                                 0.0f))
                                                .xyz(),
                                        .normal_to_world_y =
                                            (normal_to_world *
                                             make_float4(
                                                 0.0f,
                                                 1.0f,
                                                 0.0f,
                                                 0.0f))
                                                .xyz(),
                                        .normal_to_world_z =
                                            (normal_to_world *
                                             make_float4(
                                                 0.0f,
                                                 0.0f,
                                                 1.0f,
                                                 0.0f))
                                                .xyz(),
                                        .dpdu = tangent,
                                        .dpdv = cross(
                                            shading_normal, tangent),
                                        .dPdx = make_float3(0.0f),
                                        .dPdy = make_float3(0.0f),
                                        .object_dPdx =
                                            make_float3(0.0f),
                                        .object_dPdy =
                                            make_float3(0.0f),
                                        .generated_dx =
                                            make_float3(0.0f),
                                        .generated_dy =
                                            make_float3(0.0f),
                                        .incoming =
                                            -candidate_ray
                                                 ->direction(),
                                        .uv = triangle_interpolate(
                                            hit->bary,
                                            uv0,
                                            uv1,
                                            uv2),
                                        .uv_dx = make_float2(0.0f),
                                        .uv_dy = make_float2(0.0f),
                                        .geometry_index =
                                            instance.geometry_index,
                                        .barycentric = hit->bary,
                                        .barycentric_dx =
                                            make_float2(0.0f),
                                        .barycentric_dy =
                                            make_float2(0.0f),
                                        .instance_id = hit->inst,
                                        .primitive_id = hit->prim,
                                        .object_random =
                                            instance.object_random,
                                        .particle_index =
                                            instance.particle_index,
                                        .random_per_island =
                                            random_per_island,
                                        .ray_visibility =
                                            shadow_visibility,
                                        .ray_events = 0u,
                                        .ray_depth = 0u,
                                        .diffuse_depth = 0u,
                                        .glossy_depth = 0u,
                                        .transparent_depth = 0u,
                                        .transmission_depth = 0u,
                                        .ray_length =
                                            hit->committed_ray_t,
                                        .time = 0.0f,
                                        .back_facing = back_facing};
                                    auto transparent =
                                        clamp(
                                            surfaces
                                                .transparent_extinction(
                                                    surface_tag,
                                                    services,
                                                    point),
                                            make_float3(0.0f),
                                            make_float3(1.0f));
                                    auto carries_light =
                                        max(
                                            transparent.x,
                                            max(
                                                transparent.y,
                                                transparent.z)) >
                                        0.0f;
                                    transmittance *= select(
                                        make_float3(1.0f),
                                        transparent,
                                        carries_light);
                                    $if (!carries_light) {
                                        candidate.commit();
                                    };
                                })
                            .on_procedural_candidate(
                                [](luisa::compute::
                                       ProceduralCandidate &) noexcept {})
                            .trace();
                    return select(
                        make_float3(0.0f),
                        transmittance,
                        committed->miss());
                };
            auto emissive_triangle_pdf =
                [&](UInt instance_index,
                    UInt primitive_index,
                    Float3 reference_position,
                    Float3 light_position,
                    Float3 p0,
                    Float3 p1,
                    Float3 p2) noexcept {
                    Bool found = false;
                    $for (emissive_index,
                          scene->emissive_triangle_count) {
                        Var<EmissiveTriangleGpu> emitter =
                            scene->emissive_triangle_buffer->read(
                                emissive_index);
                        found =
                            found |
                            ((emitter.instance_index ==
                              instance_index) &
                             (emitter.primitive_index ==
                              primitive_index));
                    };
                    Float3 edge01 = p1 - p0;
                    Float3 edge02 = p2 - p0;
                    Float3 unnormalized_normal =
                        cross(edge01, edge02);
                    Float doubled_area = sqrt(max(
                        length_squared(unnormalized_normal),
                        0.0f));
                    Float area = 0.5f * doubled_area;
                    Float3 offset =
                        light_position - reference_position;
                    Float distance2 = length_squared(offset);
                    Float3 wi = offset / sqrt(max(
                        distance2, 1.0e-20f));
                    Float3 light_normal =
                        unnormalized_normal /
                        max(doubled_area, 1.0e-20f);
                    Float cosine =
                        abs(dot(light_normal, -wi));
                    Float pdf =
                        distance2 *
                        static_cast<float>(
                            scene->emissive_triangle_count) /
                        max(cosine * area, 1.0e-20f);
                    return select(
                        0.0f,
                        pdf,
                        found &
                            (distance2 > 1.0e-12f) &
                            (cosine > 0.0f) &
                            (area > 0.0f));
                };

            $for (sample_offset, samples) {
                UInt sample_index =
                    sample_first + sample_offset;
                UInt state = hash_seed(
                    pixel, sample_index, seed);
                Float jitter_x =
                    sample_pixel_filter(random_float(state));
                Float jitter_y =
                    sample_pixel_filter(random_float(state));
                Float width = static_cast<float>(
                    std::max(
                        render_settings.full_extent.width, 1u));
                Float height = static_cast<float>(
                    std::max(
                        render_settings.full_extent.height, 1u));
                Float screen_x =
                    2.0f *
                        (cast<float>(full_x) + jitter_x) /
                        width -
                    1.0f +
                    2.0f * camera_shift_x;
                Float screen_y =
                    1.0f -
                    2.0f *
                        (cast<float>(full_y) + jitter_y) /
                        height +
                    2.0f * camera_shift_y;
                Float aspect = width / height;

                Float3 local_origin = make_float3(0.0f);
                Float3 local_origin_dx = make_float3(0.0f);
                Float3 local_origin_dy = make_float3(0.0f);
                Float3 local_direction =
                    make_float3(0.0f, 0.0f, -1.0f);
                Float3 local_direction_dx = local_direction;
                Float3 local_direction_dy = local_direction;
                Float ray_dP = 0.0f;
                Float ray_dD = 0.0f;
                if (camera_projection ==
                    CameraProjection::perspective) {
                    local_direction = normalize(make_float3(
                        screen_x * camera_horizontal_tangent,
                        screen_y * camera_vertical_tangent,
                        -1.0f));
                    local_direction_dx =
                        normalize(make_float3(
                            (screen_x + 2.0f / width) *
                                camera_horizontal_tangent,
                            screen_y *
                                camera_vertical_tangent,
                            -1.0f));
                    local_direction_dy =
                        normalize(make_float3(
                            screen_x *
                                camera_horizontal_tangent,
                            (screen_y - 2.0f / height) *
                                camera_vertical_tangent,
                            -1.0f));
                    ray_dD =
                        0.5f *
                        (length(
                             local_direction_dx -
                             local_direction) +
                         length(
                             local_direction_dy -
                             local_direction));
                } else if (
                    camera_projection ==
                    CameraProjection::orthographic) {
                    local_origin = make_float3(
                        screen_x * camera_ortho_scale *
                            aspect * 0.5f,
                        screen_y * camera_ortho_scale * 0.5f,
                        0.0f);
                    local_origin_dx = make_float3(
                        (screen_x + 2.0f / width) *
                            camera_ortho_scale * aspect * 0.5f,
                        screen_y * camera_ortho_scale * 0.5f,
                        0.0f);
                    local_origin_dy = make_float3(
                        screen_x * camera_ortho_scale *
                            aspect * 0.5f,
                        (screen_y - 2.0f / height) *
                            camera_ortho_scale * 0.5f,
                        0.0f);
                    ray_dP =
                        0.5f *
                        (camera_ortho_scale * aspect / width +
                         camera_ortho_scale / height);
                } else {
                    Float longitude = screen_x * pi;
                    Float latitude = screen_y * pi * 0.5f;
                    Float cosine_latitude = cos(latitude);
                    local_direction = make_float3(
                        cosine_latitude * sin(longitude),
                        sin(latitude),
                        -cosine_latitude * cos(longitude));
                    Float longitude_dx =
                        (screen_x + 2.0f / width) * pi;
                    Float latitude_dy =
                        (screen_y - 2.0f / height) *
                        pi * 0.5f;
                    local_direction_dx =
                        make_float3(
                            cosine_latitude *
                                sin(longitude_dx),
                            sin(latitude),
                            -cosine_latitude *
                                cos(longitude_dx));
                    Float cosine_latitude_dy =
                        cos(latitude_dy);
                    local_direction_dy =
                        make_float3(
                            cosine_latitude_dy *
                                sin(longitude),
                            sin(latitude_dy),
                            -cosine_latitude_dy *
                                cos(longitude));
                    ray_dD =
                        0.5f *
                        (length(
                             local_direction_dx -
                             local_direction) +
                         length(
                             local_direction_dy -
                             local_direction));
                }
                if (camera_projection ==
                        CameraProjection::perspective &&
                    camera_aperture_radius > 0.0f &&
                    camera_focal_distance > 0.0f) {
                    Float lens_radius =
                        sqrt(random_float(state));
                    Float lens_angle =
                        2.0f * pi * random_float(state);
                    Float2 lens_position =
                        make_float2(
                            cos(lens_angle),
                            sin(lens_angle)) *
                        (lens_radius *
                         camera_aperture_radius);
                    lens_position.x /=
                        std::max(
                            camera_aperture_ratio, 1.0e-5f);
                    Float focus_scale =
                        camera_focal_distance /
                        max(-local_direction.z, 1.0e-6f);
                    Float3 focus_position =
                        local_direction * focus_scale;
                    Float focus_scale_dx =
                        camera_focal_distance /
                        max(-local_direction_dx.z, 1.0e-6f);
                    Float focus_scale_dy =
                        camera_focal_distance /
                        max(-local_direction_dy.z, 1.0e-6f);
                    Float3 focus_position_dx =
                        local_direction_dx * focus_scale_dx;
                    Float3 focus_position_dy =
                        local_direction_dy * focus_scale_dy;
                    local_origin = make_float3(
                        lens_position.x,
                        lens_position.y,
                        0.0f);
                    local_origin_dx = local_origin;
                    local_origin_dy = local_origin;
                    local_direction = normalize(
                        focus_position - local_origin);
                    local_direction_dx = normalize(
                        focus_position_dx - local_origin_dx);
                    local_direction_dy = normalize(
                        focus_position_dy - local_origin_dy);
                }
                Float3 ray_origin =
                    (camera_transform *
                     make_float4(local_origin, 1.0f))
                        .xyz();
                Float3 ray_direction = safe_normalize(
                    (camera_transform *
                     make_float4(local_direction, 0.0f))
                        .xyz(),
                    make_float3(0.0f, 0.0f, -1.0f));
                Float3 differential_origin_x =
                    (camera_transform *
                     make_float4(local_origin_dx, 1.0f))
                        .xyz();
                Float3 differential_origin_y =
                    (camera_transform *
                     make_float4(local_origin_dy, 1.0f))
                        .xyz();
                Float3 differential_direction_x =
                    safe_normalize(
                        (camera_transform *
                         make_float4(
                             local_direction_dx, 0.0f))
                            .xyz(),
                        ray_direction);
                Float3 differential_direction_y =
                    safe_normalize(
                        (camera_transform *
                         make_float4(
                             local_direction_dy, 0.0f))
                            .xyz(),
                        ray_direction);
                Bool use_full_ray_differentials = true;
                Var<luisa::compute::Ray> ray = make_ray(
                    ray_origin,
                    ray_direction,
                    camera_near,
                    camera_far);
                UInt ray_visibility = camera_visibility;

                Float3 radiance = make_float3(0.0f);
                Float3 throughput = make_float3(1.0f);
                Float3 sample_normal = make_float3(0.0f);
                Float3 sample_albedo = make_float3(0.0f);
                Float sample_alpha =
                    render_settings.transparent_background
                        ? 0.0f
                        : 1.0f;
                Bool primary_recorded = false;
                Float previous_bsdf_pdf = 0.0f;
                Bool previous_delta = true;
                UInt ray_events = 0u;
                UInt diffuse_depth = 0u;
                UInt glossy_depth = 0u;
                UInt transparent_depth = 0u;
                UInt transmission_depth = 0u;
                UInt path_depth = 0u;
                Bool terminate_after_transparent = false;
                Bool terminate_on_next_surface = false;

                $for (path_step, max_path_steps) {
                    static_cast<void>(path_step);
                    Var<luisa::compute::SurfaceHit> hit =
                        scene->accel->intersect(
                            ray,
                            {.visibility_mask =
                                 ray_visibility});
                    $if (hit->miss()) {
                        Bool competing =
                            (path_depth > 0u) & (!previous_delta);
                        Float environment_weight =
                            forward_light_weight(
                                previous_bsdf_pdf,
                                uniform_sphere_pdf,
                                competing,
                                true);
                        radiance += clamp_light_contribution(
                            throughput *
                            evaluate_environment_base(
                                ray->direction()) *
                                environment_weight,
                            path_depth);
                        for (const auto &sun :
                             scene->environment_suns) {
                            const auto solid_angle =
                                2.0f * pi *
                                (1.0f -
                                 std::cos(
                                     sun.angular_radius));
                            const auto sun_pdf =
                                1.0f /
                                std::max(
                                    solid_angle, 1.0e-20f);
                            Float sun_weight =
                                forward_light_weight(
                                    previous_bsdf_pdf,
                                    sun_pdf,
                                    competing,
                                    true);
                            radiance += clamp_light_contribution(
                                throughput *
                                evaluate_environment_sun(
                                    ray->direction(), sun) *
                                    sun_weight,
                                path_depth);
                        }
                        $break;
                    };

                    Var<InstanceGpu> instance =
                        scene->instance_buffer->read(hit->inst);
                    Var<GeometryGpu> geometry =
                        scene->geometry_buffer->read(
                            instance.geometry_index);
                    Var<Triangle> triangle =
                        scene->heap
                            ->buffer<Triangle>(
                                geometry.bindless_base)
                            .read(hit->prim);
                    Float3 p0 =
                        scene->heap
                            ->buffer<luisa::float3>(
                                geometry.bindless_base + 1u)
                            .read(triangle.i0);
                    Float3 p1 =
                        scene->heap
                            ->buffer<luisa::float3>(
                                geometry.bindless_base + 1u)
                            .read(triangle.i1);
                    Float3 p2 =
                        scene->heap
                            ->buffer<luisa::float3>(
                                geometry.bindless_base + 1u)
                            .read(triangle.i2);
                    Float3 n0 =
                        scene->heap
                            ->buffer<luisa::float3>(
                                geometry.bindless_base + 2u)
                            .read(triangle.i0);
                    Float3 n1 =
                        scene->heap
                            ->buffer<luisa::float3>(
                                geometry.bindless_base + 2u)
                            .read(triangle.i1);
                    Float3 n2 =
                        scene->heap
                            ->buffer<luisa::float3>(
                                geometry.bindless_base + 2u)
                            .read(triangle.i2);
                    Float2 uv0 =
                        scene->heap
                            ->buffer<luisa::float2>(
                                geometry.bindless_base + 3u)
                            .read(triangle.i0);
                    Float2 uv1 =
                        scene->heap
                            ->buffer<luisa::float2>(
                                geometry.bindless_base + 3u)
                            .read(triangle.i1);
                    Float2 uv2 =
                        scene->heap
                            ->buffer<luisa::float2>(
                                geometry.bindless_base + 3u)
                            .read(triangle.i2);
                    Float4 tangent0 =
                        scene->heap
                            ->buffer<luisa::float4>(
                                geometry.bindless_base + 7u)
                            .read(triangle.i0);
                    Float4 tangent1 =
                        scene->heap
                            ->buffer<luisa::float4>(
                                geometry.bindless_base + 7u)
                            .read(triangle.i1);
                    Float4 tangent2 =
                        scene->heap
                            ->buffer<luisa::float4>(
                                geometry.bindless_base + 7u)
                            .read(triangle.i2);
                    Float3 generated0 =
                        scene->heap
                            ->buffer<luisa::float3>(
                                geometry.bindless_base + 5u)
                            .read(triangle.i0);
                    Float3 generated1 =
                        scene->heap
                            ->buffer<luisa::float3>(
                                geometry.bindless_base + 5u)
                            .read(triangle.i1);
                    Float3 generated2 =
                        scene->heap
                            ->buffer<luisa::float3>(
                                geometry.bindless_base + 5u)
                            .read(triangle.i2);
                    Float random_per_island =
                        scene->heap
                            ->buffer<float>(
                                geometry.bindless_base + 6u)
                            .read(hit->prim);
                    UInt material_slot =
                        scene->heap
                            ->buffer<luisa::uint>(
                                geometry.bindless_base + 4u)
                            .read(hit->prim);

                    auto object_to_world =
                        scene->accel
                            ->instance_transform(hit->inst);
                    auto world_to_object =
                        inverse(object_to_world);
                    auto normal_to_world =
                        transpose(world_to_object);
                    Float3 wp0 =
                        (object_to_world *
                         make_float4(p0, 1.0f))
                            .xyz();
                    Float3 wp1 =
                        (object_to_world *
                         make_float4(p1, 1.0f))
                            .xyz();
                    Float3 wp2 =
                        (object_to_world *
                         make_float4(p2, 1.0f))
                            .xyz();
                    Float3 geometric_normal = safe_normalize(
                        cross(wp1 - wp0, wp2 - wp0),
                        -ray->direction());
                    Float3 object_shading_normal =
                        triangle_interpolate(
                            hit->bary, n0, n1, n2);
                    Float4 object_tangent =
                        triangle_interpolate(
                            hit->bary,
                            tangent0,
                            tangent1,
                            tangent2);
                    Float3 shading_normal = safe_normalize(
                        (normal_to_world *
                         make_float4(
                             object_shading_normal,
                             0.0f))
                            .xyz(),
                        geometric_normal);
                    Bool back_facing =
                        dot(geometric_normal, -ray->direction()) <
                        0.0f;
                    geometric_normal = select(
                        geometric_normal,
                        -geometric_normal,
                        back_facing);
                    shading_normal = select(
                        shading_normal,
                        -shading_normal,
                        back_facing);
                    shading_normal = select(
                        shading_normal,
                        -shading_normal,
                        dot(shading_normal, geometric_normal) <
                            0.0f);
                    Float3 hit_position =
                        ray->origin() +
                        ray->direction() * hit->committed_ray_t;
                    Float3 tangent = safe_normalize(
                        (object_to_world *
                         make_float4(
                             object_tangent.xyz(), 0.0f))
                            .xyz(),
                        safe_normalize(
                            (wp1 - wp0) -
                                geometric_normal *
                                    dot(
                                        wp1 - wp0,
                                        geometric_normal),
                            make_float3(
                                1.0f, 0.0f, 0.0f)));
                    Float3 differential_bitangent =
                        safe_normalize(
                            cross(geometric_normal, tangent),
                            make_float3(0.0f, 1.0f, 0.0f));
                    Float surface_radius =
                        ray_dP +
                        hit->committed_ray_t * ray_dD;
                    Float3 approximate_dPdx =
                        tangent * surface_radius;
                    Float3 approximate_dPdy =
                        differential_bitangent *
                        surface_radius;
                    auto surface_ray_differential =
                        [&](Float3 differential_origin,
                            Float3 differential_direction,
                            Float3 approximation) noexcept {
                            auto ray_direction =
                                ray->direction();
                            auto origin_delta =
                                differential_origin -
                                ray->origin();
                            auto direction_delta =
                                differential_direction -
                                ray_direction;
                            auto denominator = dot(
                                geometric_normal,
                                ray_direction);
                            auto valid =
                                use_full_ray_differentials &
                                (abs(denominator) > 1.0e-7f);
                            auto safe_denominator = select(
                                1.0f, denominator, valid);
                            auto distance_delta =
                                -dot(
                                    geometric_normal,
                                    origin_delta +
                                        hit->committed_ray_t *
                                            direction_delta) /
                                safe_denominator;
                            auto exact =
                                origin_delta +
                                hit->committed_ray_t *
                                    direction_delta +
                                ray_direction * distance_delta;
                            return select(
                                approximation, exact, valid);
                        };
                    Float3 dPdx = surface_ray_differential(
                        differential_origin_x,
                        differential_direction_x,
                        approximate_dPdx);
                    Float3 dPdy = surface_ray_differential(
                        differential_origin_y,
                        differential_direction_y,
                        approximate_dPdy);
                    Float differential_radius =
                        0.5f *
                        (length(dPdx) + length(dPdy));
                    Float3 edge1 = wp1 - wp0;
                    Float3 edge2 = wp2 - wp0;
                    Float gram00 = dot(edge1, edge1);
                    Float gram01 = dot(edge1, edge2);
                    Float gram11 = dot(edge2, edge2);
                    Float gram_determinant =
                        gram00 * gram11 -
                        gram01 * gram01;
                    Bool valid_gram =
                        abs(gram_determinant) > 1.0e-20f;
                    Float safe_gram_determinant = select(
                        1.0f,
                        gram_determinant,
                        valid_gram);
                    auto barycentric_differential =
                        [&](Float3 differential) noexcept {
                            Float projected1 =
                                dot(differential, edge1);
                            Float projected2 =
                                dot(differential, edge2);
                            Float2 delta = make_float2(
                                (projected1 * gram11 -
                                 projected2 * gram01) /
                                    safe_gram_determinant,
                                (projected2 * gram00 -
                                 projected1 * gram01) /
                                    safe_gram_determinant);
                            return select(
                                make_float2(0.0f),
                                delta,
                                valid_gram);
                        };
                    Float2 barycentric_dx =
                        barycentric_differential(dPdx);
                    Float2 barycentric_dy =
                        barycentric_differential(dPdy);
                    Float3 object_dPdx =
                        (world_to_object *
                         make_float4(dPdx, 0.0f))
                            .xyz();
                    Float3 object_dPdy =
                        (world_to_object *
                         make_float4(dPdy, 0.0f))
                            .xyz();
                    Float3 generated_dx =
                        (generated1 - generated0) *
                            barycentric_dx.x +
                        (generated2 - generated0) *
                            barycentric_dx.y;
                    Float3 generated_dy =
                        (generated1 - generated0) *
                            barycentric_dy.x +
                        (generated2 - generated0) *
                            barycentric_dy.y;
                    Float2 uv = triangle_interpolate(
                        hit->bary, uv0, uv1, uv2);
                    Float2 uv_dx =
                        (uv1 - uv0) * barycentric_dx.x +
                        (uv2 - uv0) * barycentric_dx.y;
                    Float2 uv_dy =
                        (uv1 - uv0) * barycentric_dy.x +
                        (uv2 - uv0) * barycentric_dy.y;

                    UInt surface_tag =
                        scene->geometry_material_buffer->read(
                            geometry.material_offset +
                            min(
                                material_slot,
                                max(
                                    geometry.material_count,
                                    1u) -
                                    1u));
                    $if (material_slot <
                         instance.override_count) {
                        surface_tag =
                            scene->override_material_buffer->read(
                                instance.override_offset +
                                material_slot);
                    };

                    SurfacePoint point{
                        .position = hit_position,
                        .object_position =
                            triangle_interpolate(
                                hit->bary, p0, p1, p2),
                        .object_location =
                            (object_to_world *
                             make_float4(
                                 0.0f, 0.0f, 0.0f, 1.0f))
                                .xyz(),
                        .generated =
                            triangle_interpolate(
                                hit->bary,
                                generated0,
                                generated1,
                                generated2),
                        .geometric_normal = geometric_normal,
                        .shading_normal = shading_normal,
                        .object_shading_normal =
                            object_shading_normal,
                        .object_tangent =
                            object_tangent.xyz(),
                        .tangent_sign = object_tangent.w,
                        .normal_to_world_x =
                            (normal_to_world *
                             make_float4(
                                 1.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f))
                                .xyz(),
                        .normal_to_world_y =
                            (normal_to_world *
                             make_float4(
                                 0.0f,
                                 1.0f,
                                 0.0f,
                                 0.0f))
                                .xyz(),
                        .normal_to_world_z =
                            (normal_to_world *
                             make_float4(
                                 0.0f,
                                 0.0f,
                                 1.0f,
                                 0.0f))
                                .xyz(),
                        .dpdu = tangent,
                        .dpdv = cross(
                            shading_normal, tangent),
                        .dPdx = dPdx,
                        .dPdy = dPdy,
                        .object_dPdx = object_dPdx,
                        .object_dPdy = object_dPdy,
                        .generated_dx = generated_dx,
                        .generated_dy = generated_dy,
                        .incoming = -ray->direction(),
                        .uv = uv,
                        .uv_dx = uv_dx,
                        .uv_dy = uv_dy,
                        .geometry_index =
                            instance.geometry_index,
                        .barycentric = hit->bary,
                        .barycentric_dx =
                            barycentric_dx,
                        .barycentric_dy =
                            barycentric_dy,
                        .instance_id = hit->inst,
                        .primitive_id = hit->prim,
                        .object_random =
                            instance.object_random,
                        .particle_index =
                            instance.particle_index,
                        .random_per_island =
                            random_per_island,
                        .ray_visibility = ray_visibility,
                        .ray_events = ray_events,
                        .ray_depth = path_depth,
                        .diffuse_depth = diffuse_depth,
                        .glossy_depth = glossy_depth,
                        .transparent_depth =
                            transparent_depth,
                        .transmission_depth =
                            transmission_depth,
                        .ray_length =
                            hit->committed_ray_t,
                        .time = 0.0f,
                        .back_facing = back_facing};
                    UInt path_lobe_mask =
                        surface_query.lobe_mask;
                    Bool previous_ray_was_diffuse =
                        (ray_events &
                         static_cast<std::uint32_t>(
                             contract::event_diffuse)) != 0u;
                    if (!reflective_caustics) {
                        path_lobe_mask = select(
                            path_lobe_mask,
                            path_lobe_mask &
                                ~static_cast<std::uint32_t>(
                                    contract::event_glossy),
                            previous_ray_was_diffuse);
                    }
                    if (!refractive_caustics) {
                        path_lobe_mask = select(
                            path_lobe_mask,
                            path_lobe_mask &
                                ~static_cast<std::uint32_t>(
                                    contract::event_transmission),
                            previous_ray_was_diffuse);
                    }
                    // Cycles' PATH_RAY_TERMINATE_AFTER_TRANSPARENT
                    // evaluates emission but allocates only transparent
                    // closures. Filtering the query here also renormalizes
                    // mixed transparent/opaque closure selection instead of
                    // probabilistically losing the transparent branch.
                    path_lobe_mask = select(
                        path_lobe_mask,
                        static_cast<std::uint32_t>(
                            contract::event_transparent),
                        terminate_after_transparent);
                    SurfaceQuery path_surface_query{
                        .lobe_mask = path_lobe_mask,
                        .transport_mode =
                            surface_query.transport_mode};

                    Float3 emitted = surfaces.emission(
                        surface_tag,
                        services,
                        point,
                        point.incoming);
                    Float emission_weight = 1.0f;
                    if (next_event_estimation &&
                        scene->emissive_triangle_count > 0u) {
                        Bool competing =
                            (path_depth > 0u) & (!previous_delta);
                        Float light_pdf =
                            emissive_triangle_pdf(
                                hit->inst,
                                hit->prim,
                                ray->origin(),
                                hit_position,
                                wp0,
                                wp1,
                                wp2);
                        emission_weight =
                            forward_light_weight(
                                previous_bsdf_pdf,
                                light_pdf,
                                competing,
                                light_pdf > 0.0f);
                    }
                    radiance += clamp_light_contribution(
                        throughput * emitted * emission_weight,
                        path_depth);

                    // PATH_RAY_TERMINATE_ON_NEXT_SURFACE still records
                    // surface emission, then stops before data passes, direct
                    // lighting, or another closure sample.
                    $if (terminate_on_next_surface) {
                        $break;
                    };

                    // Cycles performs continuation roulette only after the
                    // next ray is known to hit a surface. Background and
                    // surface-emission contributions above are therefore
                    // retained even when the path does not continue
                    // scattering.
                    Bool arrived_through_transparency =
                        (ray_events &
                         static_cast<std::uint32_t>(
                             contract::event_transparent)) != 0u;
                    // Cycles scene sync stores both UI minimum-bounce
                    // settings as value + 1. A depth of zero is the camera
                    // segment, so roulette starts only after that extra
                    // guaranteed bounce.
                    Bool use_roulette = select(
                        path_depth > min_bounces + 1u,
                        transparent_depth >
                            transparent_min_bounces + 1u,
                        arrived_through_transparency);
                    $if (use_roulette) {
                        Float survival = min(
                            sqrt(
                                max(
                                    abs(throughput.x),
                                    max(
                                        abs(throughput.y),
                                        abs(throughput.z)))),
                            1.0f);
                        $if (survival <= 0.0f) {
                            $break;
                        };
                        $if (random_float(state) >= survival) {
                            $break;
                        };
                        throughput /= survival;
                    };

                    // Cycles writes camera data passes only along the
                    // transparent-background chain. Diffuse Color is
                    // throughput-weighted at every surface in that chain;
                    // Normal is captured once, after skipping transparent
                    // surfaces below the View Layer alpha threshold.
                    $if (path_depth == 0u) {
                        auto aov = surfaces.aov(
                            surface_tag, services, point);
                        sample_albedo +=
                            throughput * aov.albedo;
                        auto surface_alpha =
                            clamp(
                                make_float3(1.0f) -
                                    aov.transparency,
                                make_float3(0.0f),
                                make_float3(1.0f));
                        auto average_alpha =
                            (surface_alpha.x +
                             surface_alpha.y +
                             surface_alpha.z) *
                            (1.0f / 3.0f);
                        auto writes_normal =
                            (!primary_recorded) &
                            ((pass_alpha_threshold == 0.0f) |
                             (average_alpha >=
                              pass_alpha_threshold));
                        sample_normal = select(
                            sample_normal,
                            aov.normal,
                            writes_normal);
                        primary_recorded =
                            primary_recorded |
                            writes_normal;
                        sample_alpha = select(
                            sample_alpha,
                            1.0f,
                            average_alpha > 0.0f);
                    };

                    if (next_event_estimation) {
                        {
                            Float environment_z =
                                1.0f -
                                2.0f * random_float(state);
                            Float environment_phi =
                                2.0f * pi *
                                random_float(state);
                            Float environment_radius = sqrt(max(
                                1.0f -
                                    environment_z *
                                        environment_z,
                                0.0f));
                            Float3 wi = make_float3(
                                environment_radius *
                                    cos(environment_phi),
                                environment_radius *
                                    sin(environment_phi),
                                environment_z);
                            Float3 shadow_origin =
                                offset_ray_origin(
                                    hit_position,
                                    geometric_normal,
                                    wi);
                            Var<luisa::compute::Ray>
                                environment_shadow_ray =
                                    make_ray(
                                        shadow_origin,
                                        wi,
                                        0.0f,
                                        ray_maximum);
                            Float3 shadow_transmittance =
                                trace_shadow(
                                    environment_shadow_ray);
                            $if (any(
                                shadow_transmittance > 0.0f)) {
                                auto evaluation =
                                    surfaces.evaluate(
                                        surface_tag,
                                        services,
                                        point,
                                        wi,
                                        path_surface_query);
                                Float mis_weight =
                                    nee_light_weight(
                                        uniform_sphere_pdf,
                                        evaluation.pdf);
                                Float3 unshadowed_contribution =
                                    evaluation.f *
                                    evaluate_environment_base(wi) *
                                    (mis_weight /
                                     uniform_sphere_pdf);
                                Float roulette_weight =
                                    light_sample_roulette_weight(
                                        unshadowed_contribution,
                                        random_float(state));
                                radiance += clamp_light_contribution(
                                    throughput *
                                    unshadowed_contribution *
                                    shadow_transmittance *
                                    roulette_weight,
                                    path_depth);
                            };
                        }
                        for (const auto &sun :
                             scene->environment_suns) {
                            const auto cosine_max =
                                std::cos(sun.angular_radius);
                            const auto solid_angle =
                                2.0f * pi *
                                (1.0f - cosine_max);
                            const auto sun_pdf =
                                1.0f /
                                std::max(
                                    solid_angle, 1.0e-20f);
                            Float3 sun_axis = safe_normalize(
                                to_luisa(sun.direction),
                                make_float3(
                                    0.0f, 0.0f, 1.0f));
                            Float3 basis_reference = select(
                                make_float3(
                                    0.0f, 0.0f, 1.0f),
                                make_float3(
                                    0.0f, 1.0f, 0.0f),
                                abs(sun_axis.z) > 0.999f);
                            Float3 sun_tangent = safe_normalize(
                                cross(
                                    basis_reference, sun_axis),
                                make_float3(
                                    1.0f, 0.0f, 0.0f));
                            Float3 sun_bitangent =
                                cross(sun_axis, sun_tangent);
                            Float cosine_theta =
                                1.0f -
                                random_float(state) *
                                    (1.0f - cosine_max);
                            Float sine_theta = sqrt(max(
                                1.0f -
                                    cosine_theta *
                                        cosine_theta,
                                0.0f));
                            Float phi =
                                2.0f * pi *
                                random_float(state);
                            Float3 wi =
                                sun_tangent *
                                    (cos(phi) * sine_theta) +
                                sun_bitangent *
                                    (sin(phi) * sine_theta) +
                                sun_axis * cosine_theta;
                            Float3 shadow_origin =
                                offset_ray_origin(
                                    hit_position,
                                    geometric_normal,
                                    wi);
                            Var<luisa::compute::Ray>
                                sun_shadow_ray = make_ray(
                                    shadow_origin,
                                    wi,
                                    0.0f,
                                    ray_maximum);
                            Float3 shadow_transmittance =
                                trace_shadow(sun_shadow_ray);
                            $if (any(
                                shadow_transmittance > 0.0f)) {
                                auto evaluation =
                                    surfaces.evaluate(
                                        surface_tag,
                                        services,
                                        point,
                                        wi,
                                        path_surface_query);
                                auto mis_weight =
                                    nee_light_weight(
                                        sun_pdf,
                                        evaluation.pdf);
                                Float3 unshadowed_contribution =
                                    evaluation.f *
                                    evaluate_environment_sun(
                                        wi, sun) *
                                    (mis_weight / sun_pdf);
                                Float roulette_weight =
                                    light_sample_roulette_weight(
                                        unshadowed_contribution,
                                        random_float(state));
                                radiance += clamp_light_contribution(
                                    throughput *
                                    unshadowed_contribution *
                                    shadow_transmittance *
                                    roulette_weight,
                                    path_depth);
                            };
                        }
                        if (scene->emissive_triangle_count > 0u) {
                            UInt selected_emitter = min(
                                cast<luisa::uint>(
                                    random_float(state) *
                                    static_cast<float>(
                                        scene
                                            ->emissive_triangle_count)),
                                scene->emissive_triangle_count - 1u);
                            Var<EmissiveTriangleGpu> emitter =
                                scene->emissive_triangle_buffer->read(
                                    selected_emitter);
                            Var<GeometryGpu> light_geometry =
                                scene->geometry_buffer->read(
                                    emitter.geometry_index);
                            Var<Triangle> light_triangle =
                                scene->heap
                                    ->buffer<Triangle>(
                                        light_geometry.bindless_base)
                                    .read(emitter.primitive_index);
                            Float3 lp0 =
                                scene->heap
                                    ->buffer<luisa::float3>(
                                        light_geometry.bindless_base +
                                        1u)
                                    .read(light_triangle.i0);
                            Float3 lp1 =
                                scene->heap
                                    ->buffer<luisa::float3>(
                                        light_geometry.bindless_base +
                                        1u)
                                    .read(light_triangle.i1);
                            Float3 lp2 =
                                scene->heap
                                    ->buffer<luisa::float3>(
                                        light_geometry.bindless_base +
                                        1u)
                                    .read(light_triangle.i2);
                            Float3 ln0 =
                                scene->heap
                                    ->buffer<luisa::float3>(
                                        light_geometry.bindless_base +
                                        2u)
                                    .read(light_triangle.i0);
                            Float3 ln1 =
                                scene->heap
                                    ->buffer<luisa::float3>(
                                        light_geometry.bindless_base +
                                        2u)
                                    .read(light_triangle.i1);
                            Float3 ln2 =
                                scene->heap
                                    ->buffer<luisa::float3>(
                                        light_geometry.bindless_base +
                                        2u)
                                    .read(light_triangle.i2);
                            Float2 luv0 =
                                scene->heap
                                    ->buffer<luisa::float2>(
                                        light_geometry.bindless_base +
                                        3u)
                                    .read(light_triangle.i0);
                            Float2 luv1 =
                                scene->heap
                                    ->buffer<luisa::float2>(
                                        light_geometry.bindless_base +
                                        3u)
                                    .read(light_triangle.i1);
                            Float2 luv2 =
                                scene->heap
                                    ->buffer<luisa::float2>(
                                        light_geometry.bindless_base +
                                        3u)
                                    .read(light_triangle.i2);
                            Float4 light_tangent0 =
                                scene->heap
                                    ->buffer<luisa::float4>(
                                        light_geometry.bindless_base +
                                        7u)
                                    .read(light_triangle.i0);
                            Float4 light_tangent1 =
                                scene->heap
                                    ->buffer<luisa::float4>(
                                        light_geometry.bindless_base +
                                        7u)
                                    .read(light_triangle.i1);
                            Float4 light_tangent2 =
                                scene->heap
                                    ->buffer<luisa::float4>(
                                        light_geometry.bindless_base +
                                        7u)
                                    .read(light_triangle.i2);
                            Float3 light_generated0 =
                                scene->heap
                                    ->buffer<luisa::float3>(
                                        light_geometry.bindless_base +
                                        5u)
                                    .read(light_triangle.i0);
                            Float3 light_generated1 =
                                scene->heap
                                    ->buffer<luisa::float3>(
                                        light_geometry.bindless_base +
                                        5u)
                                    .read(light_triangle.i1);
                            Float3 light_generated2 =
                                scene->heap
                                    ->buffer<luisa::float3>(
                                        light_geometry.bindless_base +
                                        5u)
                                    .read(light_triangle.i2);
                            Float light_random_per_island =
                                scene->heap
                                    ->buffer<float>(
                                        light_geometry.bindless_base +
                                        6u)
                                    .read(
                                        emitter.primitive_index);
                            Var<InstanceGpu> light_instance =
                                scene->instance_buffer->read(
                                    emitter.instance_index);
                            Float3 local_lp0 = lp0;
                            Float3 local_lp1 = lp1;
                            Float3 local_lp2 = lp2;
                            auto light_object_to_world =
                                scene->accel->instance_transform(
                                    emitter.instance_index);
                            auto light_normal_to_world =
                                transpose(inverse(
                                    light_object_to_world));
                            lp0 =
                                (light_object_to_world *
                                 make_float4(lp0, 1.0f))
                                    .xyz();
                            lp1 =
                                (light_object_to_world *
                                 make_float4(lp1, 1.0f))
                                    .xyz();
                            lp2 =
                                (light_object_to_world *
                                 make_float4(lp2, 1.0f))
                                    .xyz();
                            Float root_u =
                                sqrt(random_float(state));
                            Float second_u =
                                random_float(state);
                            Float2 light_barycentric =
                                make_float2(
                                    root_u * (1.0f - second_u),
                                    root_u * second_u);
                            Float3 light_position =
                                triangle_interpolate(
                                    light_barycentric,
                                    lp0,
                                    lp1,
                                    lp2);
                            Float3 light_unnormalized_normal =
                                cross(lp1 - lp0, lp2 - lp0);
                            Float3 light_geometric_normal =
                                safe_normalize(
                                    light_unnormalized_normal,
                                    make_float3(
                                        0.0f, 0.0f, 1.0f));
                            Float3 light_object_shading_normal =
                                triangle_interpolate(
                                    light_barycentric,
                                    ln0,
                                    ln1,
                                    ln2);
                            Float4 light_object_tangent =
                                triangle_interpolate(
                                    light_barycentric,
                                    light_tangent0,
                                    light_tangent1,
                                    light_tangent2);
                            Float3 light_shading_normal =
                                safe_normalize(
                                    (light_normal_to_world *
                                     make_float4(
                                         light_object_shading_normal,
                                         0.0f))
                                        .xyz(),
                                    light_geometric_normal);
                            Float3 light_offset =
                                light_position - hit_position;
                            Float light_distance2 =
                                length_squared(light_offset);
                            Float light_distance =
                                sqrt(max(
                                    light_distance2, 1.0e-20f));
                            Float3 wi =
                                light_offset / light_distance;
                            Bool light_back_facing =
                                dot(
                                    light_geometric_normal,
                                    -wi) < 0.0f;
                            light_geometric_normal = select(
                                light_geometric_normal,
                                -light_geometric_normal,
                                light_back_facing);
                            light_shading_normal = select(
                                light_shading_normal,
                                -light_shading_normal,
                                light_back_facing);
                            light_shading_normal = select(
                                light_shading_normal,
                                -light_shading_normal,
                                dot(
                                    light_shading_normal,
                                    light_geometric_normal) <
                                    0.0f);
                            Float3 light_tangent =
                                safe_normalize(
                                    (light_object_to_world *
                                     make_float4(
                                         light_object_tangent.xyz(),
                                         0.0f))
                                        .xyz(),
                                    safe_normalize(
                                        (lp1 - lp0) -
                                            light_geometric_normal *
                                                dot(
                                                    lp1 - lp0,
                                                    light_geometric_normal),
                                        make_float3(
                                            1.0f,
                                            0.0f,
                                            0.0f)));
                            SurfacePoint light_point{
                                .position = light_position,
                                .object_position =
                                    triangle_interpolate(
                                        light_barycentric,
                                        local_lp0,
                                        local_lp1,
                                        local_lp2),
                                .object_location =
                                    (light_object_to_world *
                                     make_float4(
                                         0.0f,
                                         0.0f,
                                         0.0f,
                                         1.0f))
                                        .xyz(),
                                .generated =
                                    triangle_interpolate(
                                        light_barycentric,
                                        light_generated0,
                                        light_generated1,
                                        light_generated2),
                                .geometric_normal =
                                    light_geometric_normal,
                                .shading_normal =
                                    light_shading_normal,
                                .object_shading_normal =
                                    light_object_shading_normal,
                                .object_tangent =
                                    light_object_tangent.xyz(),
                                .tangent_sign =
                                    light_object_tangent.w,
                                .normal_to_world_x =
                                    (light_normal_to_world *
                                     make_float4(
                                         1.0f,
                                         0.0f,
                                         0.0f,
                                         0.0f))
                                        .xyz(),
                                .normal_to_world_y =
                                    (light_normal_to_world *
                                     make_float4(
                                         0.0f,
                                         1.0f,
                                         0.0f,
                                         0.0f))
                                        .xyz(),
                                .normal_to_world_z =
                                    (light_normal_to_world *
                                     make_float4(
                                         0.0f,
                                         0.0f,
                                         1.0f,
                                         0.0f))
                                        .xyz(),
                                .dpdu = light_tangent,
                                .dpdv = cross(
                                    light_shading_normal,
                                    light_tangent),
                                .dPdx = make_float3(0.0f),
                                .dPdy = make_float3(0.0f),
                                .object_dPdx =
                                    make_float3(0.0f),
                                .object_dPdy =
                                    make_float3(0.0f),
                                .generated_dx =
                                    make_float3(0.0f),
                                .generated_dy =
                                    make_float3(0.0f),
                                .incoming = -wi,
                                .uv = triangle_interpolate(
                                    light_barycentric,
                                    luv0,
                                    luv1,
                                    luv2),
                                .uv_dx = make_float2(0.0f),
                                .uv_dy = make_float2(0.0f),
                                .geometry_index =
                                    emitter.geometry_index,
                                .barycentric =
                                    light_barycentric,
                                .barycentric_dx =
                                    make_float2(0.0f),
                                .barycentric_dy =
                                    make_float2(0.0f),
                                .instance_id =
                                    emitter.instance_index,
                                .primitive_id =
                                    emitter.primitive_index,
                                .object_random =
                                    light_instance.object_random,
                                .particle_index =
                                    light_instance.particle_index,
                                .random_per_island =
                                    light_random_per_island,
                                .ray_visibility =
                                    shadow_visibility,
                                .ray_events = 0u,
                                .ray_depth = path_depth,
                                .diffuse_depth =
                                    diffuse_depth,
                                .glossy_depth =
                                    glossy_depth,
                                .transparent_depth =
                                    transparent_depth,
                                .transmission_depth =
                                    transmission_depth,
                                .ray_length =
                                    light_distance,
                                .time = 0.0f,
                                .back_facing =
                                    light_back_facing};
                            Float3 light_radiance =
                                surfaces.emission(
                                    emitter.surface_tag,
                                    services,
                                    light_point,
                                    -wi);
                            Float light_pdf =
                                emissive_triangle_pdf(
                                    emitter.instance_index,
                                    emitter.primitive_index,
                                    hit_position,
                                    light_position,
                                    lp0,
                                    lp1,
                                    lp2);
                            $if ((light_pdf > 0.0f) &
                                 any(light_radiance > 0.0f)) {
                                Float3 shadow_origin =
                                    offset_ray_origin(
                                        hit_position,
                                        geometric_normal,
                                        wi);
                                Var<luisa::compute::Ray>
                                    mesh_light_shadow_ray =
                                        make_ray(
                                            shadow_origin,
                                            wi,
                                            0.0f,
                                            max(
                                                light_distance -
                                                    1.0e-4f,
                                                0.0f));
                                Float3 shadow_transmittance =
                                    trace_shadow(
                                        mesh_light_shadow_ray);
                                $if (any(
                                    shadow_transmittance > 0.0f)) {
                                    auto evaluation =
                                        surfaces.evaluate(
                                            surface_tag,
                                            services,
                                            point,
                                            wi,
                                            path_surface_query);
                                    Float mis_weight =
                                        nee_light_weight(
                                            light_pdf,
                                            evaluation.pdf);
                                    Float3 unshadowed_contribution =
                                        evaluation.f *
                                        light_radiance *
                                        (mis_weight / light_pdf);
                                    Float roulette_weight =
                                        light_sample_roulette_weight(
                                            unshadowed_contribution,
                                            random_float(state));
                                    radiance += clamp_light_contribution(
                                        throughput *
                                        unshadowed_contribution *
                                        shadow_transmittance *
                                        roulette_weight,
                                        path_depth);
                                };
                            };
                        }
                        for (std::uint32_t light_index = 0u;
                             light_index < scene->light_count;
                             ++light_index) {
                            Var<LightGpu> light =
                                scene->light_buffer->read(
                                    light_index);
                            Float3 wi = make_float3(0.0f);
                            Float3 light_radiance =
                                make_float3(0.0f);
                            Float light_distance = ray_maximum;
                            Float light_pdf = 0.0f;
                            Bool light_valid = false;

                            $if (light.type ==
                                 static_cast<std::uint32_t>(
                                     LightType::area)) {
                                Float u = random_float(state) -
                                          0.5f;
                                Float v = random_float(state) -
                                          0.5f;
                                Float3 light_position =
                                    light.position +
                                    light.axis_x * (u * light.size) +
                                    light.axis_y * (v * light.size);
                                Float3 offset =
                                    light_position - hit_position;
                                Float distance2 =
                                    length_squared(offset);
                                light_distance = sqrt(max(
                                    distance2, 1.0e-20f));
                                wi = offset / light_distance;
                                Float cosine = max(
                                    dot(
                                        -light.axis_z,
                                        -wi),
                                    0.0f);
                                Float area = max(
                                    light.size * light.size,
                                    1.0e-12f);
                                light_pdf =
                                    distance2 /
                                    max(cosine * area, 1.0e-20f);
                                light_radiance =
                                    light.color * light.power;
                                light_valid =
                                    (distance2 > 1.0e-12f) &
                                    (cosine > 0.0f);
                            }
                            $elif (
                                light.type ==
                                    static_cast<std::uint32_t>(
                                        LightType::distant)) {
                                wi = -light.axis_z;
                                light_radiance =
                                    light.color * light.power;
                                light_pdf = 1.0f;
                                light_valid = true;
                            }
                            $else {
                                Float3 offset =
                                    light.position - hit_position;
                                Float distance2 =
                                    length_squared(offset);
                                light_distance = sqrt(max(
                                    distance2, 1.0e-20f));
                                wi = offset / light_distance;
                                light_radiance =
                                    light.color *
                                    (light.power /
                                     (4.0f * pi *
                                      max(distance2, 1.0e-20f)));
                                light_pdf = 1.0f;
                                light_valid =
                                    distance2 > 1.0e-12f;
                                $if (light.type ==
                                     static_cast<std::uint32_t>(
                                         LightType::spot)) {
                                    Float cone =
                                        dot(-light.axis_z, -wi);
                                    light_valid =
                                        light_valid &
                                        (cone >=
                                         cos(
                                             max(
                                                 light.spread,
                                                 0.0f) *
                                             0.5f));
                                };
                            };

                            $if (light_valid &
                                 (light_pdf > 0.0f)) {
                                Float3 shadow_origin =
                                    offset_ray_origin(
                                        hit_position,
                                        geometric_normal,
                                        wi);
                                Var<luisa::compute::Ray>
                                    shadow_ray = make_ray(
                                        shadow_origin,
                                        wi,
                                        0.0f,
                                        max(
                                            light_distance -
                                                1.0e-4f,
                                            0.0f));
                                Float3 shadow_transmittance =
                                    trace_shadow(shadow_ray);
                                $if (any(
                                    shadow_transmittance > 0.0f)) {
                                    auto evaluation =
                                        surfaces.evaluate(
                                            surface_tag,
                                            services,
                                            point,
                                            wi,
                                            path_surface_query);
                                    // Analytic lights are not part of the
                                    // Psycles acceleration structure yet, so
                                    // forward BSDF rays cannot hit them. Their
                                    // NEE estimator has no competing forward
                                    // technique and must carry full weight.
                                    Float mis_weight = 1.0f;
                                    Float3 unshadowed_contribution =
                                        evaluation.f *
                                        light_radiance *
                                        (mis_weight /
                                         max(
                                             light_pdf,
                                             1.0e-20f));
                                    Float roulette_weight =
                                        light_sample_roulette_weight(
                                            unshadowed_contribution,
                                            random_float(state));
                                    radiance += clamp_light_contribution(
                                        throughput *
                                        unshadowed_contribution *
                                        shadow_transmittance *
                                        roulette_weight,
                                        path_depth);
                                };
                            };
                        }
                    }

                    auto surface_sample = surfaces.sample(
                        surface_tag,
                        services,
                        point,
                        random_float(state),
                        make_float2(
                            random_float(state),
                            random_float(state)),
                        path_surface_query);
                    $if (!surface_sample.valid |
                         (surface_sample.evaluation.pdf <=
                          0.0f)) {
                        $break;
                    };

                    Bool transparent =
                        (surface_sample.evaluation.events &
                         static_cast<std::uint32_t>(
                             contract::event_transparent)) != 0u;
                    Bool transmission =
                        (surface_sample.evaluation.events &
                         static_cast<std::uint32_t>(
                             contract::event_transmission)) != 0u;
                    Bool glossy =
                        (surface_sample.evaluation.events &
                         static_cast<std::uint32_t>(
                             contract::event_glossy)) != 0u;
                    Bool diffuse =
                        (surface_sample.evaluation.events &
                         static_cast<std::uint32_t>(
                             contract::event_diffuse)) != 0u;
                    Bool singular =
                        (surface_sample.evaluation.events &
                         static_cast<std::uint32_t>(
                             contract::event_singular)) != 0u;
                    Bool reflection =
                        (surface_sample.evaluation.events &
                         static_cast<std::uint32_t>(
                             contract::event_reflection)) != 0u;
                    Bool diffuse_reflection =
                        diffuse & reflection & (!transparent);
                    Bool glossy_reflection =
                        glossy & reflection & (!transparent);
                    Bool material_transmission =
                        transmission & (!transparent);

                    throughput *=
                        surface_sample.evaluation.f /
                        surface_sample.evaluation.pdf;
                    $if (any(luisa::compute::dsl::isnan(
                             throughput)) |
                         any(throughput < 0.0f)) {
                        $break;
                    };

                    UInt scattered_visibility = select(
                        diffuse_visibility,
                        glossy_visibility,
                        glossy);
                    scattered_visibility = select(
                        scattered_visibility,
                        transmission_visibility,
                        transmission);
                    ray_visibility = select(
                        scattered_visibility,
                        ray_visibility,
                        transparent);
                    ray_events = select(
                        surface_sample.evaluation.events,
                        ray_events |
                            surface_sample.evaluation.events,
                        transparent);
                    diffuse_depth += select(
                        0u, 1u, diffuse_reflection);
                    glossy_depth += select(
                        0u, 1u, glossy_reflection);
                    transmission_depth += select(
                        0u, 1u, material_transmission);
                    transparent_depth += select(
                        0u, 1u, transparent);
                    path_depth += select(
                        0u, 1u, !transparent);
                    terminate_on_next_surface |=
                        transparent &
                        (transparent_depth >=
                         transparent_max_bounces);
                    terminate_after_transparent |=
                        (!transparent) &
                        ((path_depth >= max_bounces) |
                         (diffuse_reflection &
                          (diffuse_depth >=
                           max_diffuse_bounces)) |
                         (glossy_reflection &
                          (glossy_depth >=
                           max_glossy_bounces)) |
                         (material_transmission &
                          (transmission_depth >=
                           max_transmission_bounces)));
                    // Cycles does not update forward-MIS state for a
                    // transparent bounce. The next emitter remains paired
                    // with the most recent non-transparent BSDF technique.
                    previous_bsdf_pdf = select(
                        surface_sample.evaluation.pdf,
                        previous_bsdf_pdf,
                        transparent);
                    previous_delta = select(
                        singular,
                        previous_delta,
                        transparent);
                    Float3 next_origin = select(
                        offset_ray_origin(
                            hit_position,
                            geometric_normal,
                            surface_sample.wi),
                        hit_position +
                            ray->direction() * 1.0e-4f,
                        transparent);
                    // Preserve the two full camera differentials through
                    // transparent hits. After the first material scatter,
                    // rebase the compact secondary-ray footprint to the
                    // measured surface differential radius.
                    ray_dP = differential_radius;
                    differential_origin_x = select(
                        differential_origin_x,
                        next_origin + dPdx,
                        transparent);
                    differential_origin_y = select(
                        differential_origin_y,
                        next_origin + dPdy,
                        transparent);
                    use_full_ray_differentials =
                        use_full_ray_differentials &
                        transparent;
                    ray = make_ray(
                        next_origin,
                        surface_sample.wi,
                        0.0f,
                        ray_maximum);
                };

                radiance = select(
                    radiance,
                    make_float3(0.0f),
                    any(luisa::compute::dsl::isnan(radiance)));
                combined_sum +=
                    make_float4(radiance, sample_alpha);
                normal_sum +=
                    make_float4(sample_normal, 1.0f);
                albedo_sum +=
                    make_float4(sample_albedo, 1.0f);
                completed += 1u;
            };
            combined.write(pixel, combined_sum);
            normal.write(pixel, normal_sum);
            albedo.write(pixel, albedo_sum);
            sample_count.write(pixel, completed);
        };

        _render_shader = _scene->device.compile(
            kernel,
            luisa::compute::ShaderOption{
                .enable_cache = true,
                .enable_fast_math = false,
                .name = "psycles_luisa_path_tracer"});
    }

    void write_passes(contract::OutputSink &output) {
        const auto count = pixel_count();
        luisa::vector<luisa::float4> combined(count);
        luisa::vector<luisa::float4> normal(count);
        luisa::vector<luisa::float4> albedo(count);
        luisa::vector<luisa::uint> samples(count);
        _stream << _combined.copy_to(luisa::span{combined})
                << _normal.copy_to(luisa::span{normal})
                << _albedo.copy_to(luisa::span{albedo})
                << _sample_count.copy_to(luisa::span{samples})
                << synchronize();

        output.begin(_settings);
        for (const auto &pass : _settings.passes) {
            if (!supported_pass(pass.kind)) {
                continue;
            }
            const auto channels = pass_channels(pass);
            std::vector<float> pixels(
                count * static_cast<std::size_t>(channels));
            for (std::size_t i = 0u; i < count; ++i) {
                const auto denominator =
                    static_cast<float>(std::max(samples[i], 1u));
                luisa::float4 value{};
                switch (pass.kind) {
                    case PassKind::combined:
                        value = combined[i] / denominator;
                        break;
                    case PassKind::normal:
                    case PassKind::denoising_normal:
                        value = normal[i] / denominator;
                        break;
                    case PassKind::albedo:
                    case PassKind::denoising_albedo:
                        value = albedo[i] / denominator;
                        break;
                    case PassKind::sample_count:
                        value = luisa::make_float4(
                            static_cast<float>(samples[i]));
                        break;
                    default:
                        break;
                }
                const std::array source{
                    value.x, value.y, value.z, value.w};
                for (std::uint32_t channel = 0u;
                     channel < channels;
                     ++channel) {
                    pixels[i * channels + channel] =
                        source[std::min<std::uint32_t>(
                            channel, 3u)];
                }
            }
            output.write(PassTile{
                .pass = pass,
                .window = _window,
                .full_extent = _settings.full_extent,
                .pixels = std::span<const float>{pixels}});
        }
        output.end(_cancelled.load());
    }

public:
    LuisaRenderSession(
        std::shared_ptr<LuisaSceneData> scene,
        LuisaPathTracerOptions options,
        const RenderSettings &settings)
        : _scene{std::move(scene)},
          _options{options},
          _stream{_scene->device.create_stream()} {
        initialize(settings);
    }

    void reset(const RenderSettings &settings) override {
        _cancelled.store(false);
        initialize(settings);
    }

    bool render_samples(
        const SampleRange &samples,
        contract::OutputSink &output) override {
        if (_cancelled.load() || samples.count == 0u) {
            return false;
        }
        _stream
            << _render_shader(
                   _combined,
                   _normal,
                   _albedo,
                   _sample_count,
                   samples.first + samples.offset,
                   samples.count,
                   _settings.seed)
                   .dispatch(
                       static_cast<std::uint32_t>(
                           std::max<std::size_t>(
                               pixel_count(), 1u)))
            << synchronize();
        if (_cancelled.load()) {
            return false;
        }
        write_passes(output);
        return true;
    }

    void cancel() noexcept override {
        _cancelled.store(true);
    }
};

void diagnose(
    std::vector<RenderDiagnostic> &diagnostics,
    std::string message) {
    diagnostics.emplace_back(RenderDiagnostic{
        .message = std::move(message)});
}

}// namespace

LuisaPathTracerBackend::LuisaPathTracerBackend(
    luisa::compute::Device device,
    LuisaPathTracerOptions options) noexcept
    : _device{std::move(device)},
      _options{options} {}

contract::SceneCompilation LuisaPathTracerBackend::compile_scene(
    const SceneSnapshot &snapshot) {
    contract::SceneCompilation result;
    if (!_device) {
        diagnose(result.diagnostics, "Luisa device is invalid.");
        return result;
    }
    if (!snapshot.active_camera) {
        diagnose(result.diagnostics, "Scene has no active camera.");
        return result;
    }
    const auto camera_iter =
        snapshot.cameras.find(*snapshot.active_camera);
    if (camera_iter == snapshot.cameras.end()) {
        diagnose(
            result.diagnostics,
            "Active camera does not exist in the scene.");
        return result;
    }
    for (const auto &[id, instance] : snapshot.instances) {
        static_cast<void>(id);
        if (!instance.motion.empty()) {
            diagnose(
                result.diagnostics,
                "Luisa vertical slice does not yet accept instance "
                "motion; the scene must be exported at a fixed frame.");
        }
        if (instance.visibility_mask !=
                ~std::uint32_t{0u} &&
            instance.visibility_mask > 0xffu) {
            diagnose(
                result.diagnostics,
                "Luisa ray visibility masks are eight-bit; the scene "
                "contains a non-default mask outside that range.");
        }
    }
    if (!result.diagnostics.empty()) {
        return result;
    }

    auto data = std::make_shared<LuisaSceneData>();
    data->device =
        luisa::compute::Device{_device.impl_shared()};
    data->revision = snapshot.revision;
    data->camera = camera_iter->second;

    ShaderCompiler shader_compiler{
        compiler::make_core_node_registry()};
    auto material_update =
        data->materials.update(snapshot, shader_compiler);
    if (!material_update.committed) {
        for (const auto &diagnostic :
             material_update.diagnostics) {
            diagnose(
                result.diagnostics,
                "Material " +
                    std::to_string(diagnostic.material.value) +
                    ": " + diagnostic.message);
        }
        return result;
    }

    luisa::vector<luisa::float4> parameters;
    for (const auto &[id, material] :
         data->materials.materials()) {
        const auto base =
            static_cast<std::uint32_t>(parameters.size());
        const auto tag =
            data->surfaces.create<GraphSurface>(
                material.surface_program(), base);
        data->material_tags.emplace(id, tag);
        for (const auto &parameter :
             material.surface_program()->parameters()) {
            const auto *value =
                material.parameters().find(parameter.id);
            parameters.emplace_back(
                value != nullptr
                    ? parameter_value(*value)
                    : luisa::make_float4(0.0f));
        }
    }
    if (snapshot.world_shader) {
        auto iter =
            data->material_tags.find(*snapshot.world_shader);
        if (iter != data->material_tags.end()) {
            data->world_surface_tag = iter->second;
        }
    }
    if (parameters.empty()) {
        parameters.emplace_back(luisa::make_float4(0.0f));
    }
    data->parameter_buffer =
        data->device.create_buffer<luisa::float4>(
            parameters.size());

    using namespace cycles45_tables;
    static_assert(
        std::size(table_ggx_E) == ggx_e_size);
    static_assert(
        std::size(table_ggx_Eavg) == ggx_eavg_size);
    static_assert(
        std::size(table_ggx_glass_E) ==
        ggx_glass_e_size);
    static_assert(
        std::size(table_ggx_glass_Eavg) ==
        ggx_glass_eavg_size);
    static_assert(
        std::size(table_ggx_glass_inv_E) ==
        ggx_glass_inv_e_size);
    static_assert(
        std::size(table_ggx_glass_inv_Eavg) ==
        ggx_glass_inv_eavg_size);
    static_assert(
        std::size(table_sheen_ltc) == sheen_ltc_size);
    static_assert(
        std::size(table_ggx_gen_schlick_ior_s) ==
        ggx_gen_schlick_ior_s_size);
    static_assert(
        std::size(table_ggx_gen_schlick_s) ==
        ggx_gen_schlick_s_size);

    luisa::vector<float> cycles_bsdf_values;
    cycles_bsdf_values.reserve(total_size);
    const auto append_cycles_table =
        [&cycles_bsdf_values](const auto &table) noexcept {
            for (const auto value : table) {
                cycles_bsdf_values.emplace_back(value);
            }
        };
    append_cycles_table(table_ggx_E);
    append_cycles_table(table_ggx_Eavg);
    append_cycles_table(table_ggx_glass_E);
    append_cycles_table(table_ggx_glass_Eavg);
    append_cycles_table(table_ggx_glass_inv_E);
    append_cycles_table(table_ggx_glass_inv_Eavg);
    append_cycles_table(table_sheen_ltc);
    append_cycles_table(table_ggx_gen_schlick_ior_s);
    append_cycles_table(table_ggx_gen_schlick_s);
    if (cycles_bsdf_values.size() != total_size) {
        diagnose(
            result.diagnostics,
            "Internal Cycles BSDF table layout mismatch.");
        return result;
    }
    data->cycles_bsdf_table_buffer =
        data->device.create_buffer<float>(
            cycles_bsdf_values.size());

    std::size_t color_attribute_count = 0u;
    for (const auto &[id, geometry] :
         snapshot.geometries) {
        static_cast<void>(id);
        color_attribute_count +=
            geometry.color_attributes.size();
    }
    const auto fixed_geometry_slots =
        snapshot.geometries.size() *
        geometry_bindless_stride;
    const auto bindless_slots =
        std::max<std::size_t>(
            fixed_geometry_slots +
                color_attribute_count,
            1u);
    data->heap =
        data->device.create_bindless_array(bindless_slots);
    data->geometries.reserve(snapshot.geometries.size());
    std::vector<GeometryUpload> uploads;
    uploads.reserve(snapshot.geometries.size());
    std::map<contract::GeometryId, std::uint32_t>
        geometry_indices;
    luisa::vector<GeometryGpu> geometry_gpu;
    luisa::vector<luisa::uint> geometry_materials;
    auto next_attribute_slot =
        static_cast<std::uint32_t>(
            fixed_geometry_slots);
    Stream stream = data->device.create_stream();
    stream << data->parameter_buffer.copy_from(
                  luisa::span{parameters})
           << data->cycles_bsdf_table_buffer.copy_from(
                  luisa::span{cycles_bsdf_values});

    std::size_t texture_slot_count = 1u;
    for (const auto &[image_id, image] : snapshot.images) {
        static_cast<void>(image);
        texture_slot_count = std::max(
            texture_slot_count,
            static_cast<std::size_t>(image_id.value) + 1u);
    }
    if (snapshot.environment) {
        data->environment_texture_slot =
            static_cast<std::uint32_t>(texture_slot_count);
        data->environment_width = snapshot.environment->width;
        data->environment_height = snapshot.environment->height;
        data->environment_suns = snapshot.environment->suns;
        ++texture_slot_count;
    }
    data->texture_heap =
        data->device.create_bindless_array(texture_slot_count);
    data->images.reserve(
        snapshot.images.size() +
        (snapshot.environment ? 2u : 1u));
    std::vector<luisa::vector<std::byte>> texture_uploads;
    texture_uploads.reserve(snapshot.images.size() + 1u);
    std::vector<luisa::vector<luisa::float4>>
        float_texture_uploads;
    float_texture_uploads.reserve(
        snapshot.environment ? 1u : 0u);

    auto &dummy_pixels = texture_uploads.emplace_back(4u);
    dummy_pixels[0u] = std::byte{255u};
    dummy_pixels[1u] = std::byte{0u};
    dummy_pixels[2u] = std::byte{255u};
    dummy_pixels[3u] = std::byte{255u};
    auto &dummy_image = data->images.emplace_back(
        data->device.create_image<float>(
            luisa::compute::PixelStorage::BYTE4, 1u, 1u));
    data->texture_heap.emplace_on_update(
        0u,
        dummy_image,
        luisa::compute::Sampler::linear_point_repeat());
    stream << dummy_image.copy_from(
        luisa::span{dummy_pixels});

    for (const auto &[image_id, image] : snapshot.images) {
        if (image.encoded_data.empty() ||
            image.encoded_data.size() >
                static_cast<std::size_t>(
                    std::numeric_limits<int>::max())) {
            diagnose(
                result.diagnostics,
                "Image '" + image.name +
                    "' has no decodable payload.");
            continue;
        }
        int width = 0;
        int height = 0;
        int channels = 0;
        auto *decoded = stbi_load_from_memory(
            image.encoded_data.data(),
            static_cast<int>(image.encoded_data.size()),
            &width,
            &height,
            &channels,
            STBI_rgb_alpha);
        if (decoded == nullptr || width <= 0 || height <= 0) {
            diagnose(
                result.diagnostics,
                "Failed to decode image '" + image.name + "'.");
            stbi_image_free(decoded);
            continue;
        }
        const auto pixel_bytes =
            static_cast<std::size_t>(width) *
            static_cast<std::size_t>(height) * 4u;
        auto &pixels =
            texture_uploads.emplace_back(pixel_bytes);
        std::memcpy(pixels.data(), decoded, pixel_bytes);
        stbi_image_free(decoded);
        if (image.alpha_type ==
            contract::ImageAlphaType::straight) {
            for (std::size_t offset = 0u;
                 offset < pixel_bytes;
                 offset += 4u) {
                const auto alpha =
                    static_cast<std::uint32_t>(
                        std::to_integer<std::uint8_t>(
                            pixels[offset + 3u]));
                for (std::size_t channel = 0u;
                     channel < 3u;
                     ++channel) {
                    const auto value =
                        static_cast<std::uint32_t>(
                            std::to_integer<std::uint8_t>(
                                pixels[offset + channel]));
                    pixels[offset + channel] =
                        static_cast<std::byte>(
                            (value * alpha) / 255u);
                }
            }
        } else if (
            image.alpha_type ==
            contract::ImageAlphaType::ignore) {
            for (std::size_t offset = 3u;
                 offset < pixel_bytes;
                 offset += 4u) {
                pixels[offset] =
                    static_cast<std::byte>(255u);
            }
        }

        auto &resource = data->images.emplace_back(
            data->device.create_image<float>(
                luisa::compute::PixelStorage::BYTE4,
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height)));
        data->texture_heap.emplace_on_update(
            static_cast<std::uint32_t>(image_id.value),
            resource,
            luisa::compute::Sampler::linear_point_repeat());
        stream << resource.copy_from(luisa::span{pixels});
    }
    if (snapshot.environment &&
        data->environment_texture_slot) {
        const auto &environment = *snapshot.environment;
        auto &pixels = float_texture_uploads.emplace_back();
        pixels.reserve(environment.pixels.size());
        for (const auto value : environment.pixels) {
            pixels.emplace_back(
                luisa::make_float4(
                    value.x, value.y, value.z, 1.0f));
        }
        auto &resource = data->images.emplace_back(
            data->device.create_image<float>(
                luisa::compute::PixelStorage::FLOAT4,
                environment.width,
                environment.height));
        data->texture_heap.emplace_on_update(
            *data->environment_texture_slot,
            resource,
            luisa::compute::Sampler::linear_point_repeat());
        stream << resource.copy_from(luisa::span{pixels});
    }
    if (!result.diagnostics.empty()) {
        return result;
    }

    for (const auto &[geometry_id, geometry] :
         snapshot.geometries) {
        if (geometry.positions.empty() ||
            geometry.triangles.empty()) {
            diagnose(
                result.diagnostics,
                "Geometry '" + geometry.name +
                    "' has no triangles.");
            continue;
        }
        auto &upload = uploads.emplace_back();
        upload.positions.reserve(geometry.positions.size());
        upload.normals.reserve(geometry.positions.size());
        upload.uv.reserve(geometry.positions.size());
        upload.uv_tangents.reserve(geometry.positions.size());
        upload.generated.reserve(geometry.positions.size());
        auto bounds_min = geometry.positions.front();
        auto bounds_max = geometry.positions.front();
        for (const auto position : geometry.positions) {
            bounds_min.x = std::min(bounds_min.x, position.x);
            bounds_min.y = std::min(bounds_min.y, position.y);
            bounds_min.z = std::min(bounds_min.z, position.z);
            bounds_max.x = std::max(bounds_max.x, position.x);
            bounds_max.y = std::max(bounds_max.y, position.y);
            bounds_max.z = std::max(bounds_max.z, position.z);
        }
        const auto generated_fallback = [&](
                                            Vec3f position) noexcept {
            const auto map_axis = [](
                                      float value,
                                      float minimum,
                                      float maximum) noexcept {
                const auto extent = maximum - minimum;
                return std::abs(extent) > 1.0e-20f
                           ? (value - minimum) / extent
                           : 0.5f;
            };
            return Vec3f{
                map_axis(position.x, bounds_min.x, bounds_max.x),
                map_axis(position.y, bounds_min.y, bounds_max.y),
                map_axis(position.z, bounds_min.z, bounds_max.z)};
        };
        for (std::size_t i = 0u;
             i < geometry.positions.size();
             ++i) {
            upload.positions.emplace_back(
                to_luisa(geometry.positions[i]));
            upload.normals.emplace_back(
                i < geometry.normals.size()
                    ? to_luisa(geometry.normals[i])
                    : luisa::make_float3(0.0f));
            upload.uv.emplace_back(
                i < geometry.uv.size()
                    ? to_luisa(geometry.uv[i])
                    : luisa::make_float2(0.0f));
            const auto uv_tangent =
                i < geometry.uv_tangents.size()
                    ? geometry.uv_tangents[i]
                    : Vec4f{};
            upload.uv_tangents.emplace_back(
                luisa::make_float4(
                    uv_tangent.x,
                    uv_tangent.y,
                    uv_tangent.z,
                    uv_tangent.w));
            upload.generated.emplace_back(
                to_luisa(
                    i < geometry.generated.size()
                        ? geometry.generated[i]
                        : generated_fallback(
                              geometry.positions[i])));
        }
        upload.color_attributes.reserve(
            geometry.color_attributes.size());
        for (const auto &[name, values] :
             geometry.color_attributes) {
            auto &attribute =
                upload.color_attributes.emplace_back();
            attribute.id = contract::attribute_id(name);
            attribute.values.reserve(values.size());
            for (const auto value : values) {
                attribute.values.emplace_back(
                    luisa::make_float4(
                        value.x,
                        value.y,
                        value.z,
                        value.w));
            }
        }
        upload.triangles.reserve(geometry.triangles.size());
        upload.triangle_material_slots.reserve(
            geometry.triangles.size());
        upload.triangle_random_per_island.reserve(
            geometry.triangles.size());
        for (std::size_t i = 0u;
             i < geometry.triangles.size();
             ++i) {
            const auto triangle = geometry.triangles[i];
            if (triangle[0u] >= geometry.positions.size() ||
                triangle[1u] >= geometry.positions.size() ||
                triangle[2u] >= geometry.positions.size()) {
                diagnose(
                    result.diagnostics,
                    "Geometry '" + geometry.name +
                        "' contains an out-of-range index.");
                break;
            }
            upload.triangles.emplace_back(Triangle{
                triangle[0u],
                triangle[1u],
                triangle[2u]});
            upload.triangle_material_slots.emplace_back(
                i < geometry.triangle_material_slots.size()
                    ? geometry.triangle_material_slots[i]
                    : 0u);
            upload.triangle_random_per_island.emplace_back(
                i <
                        geometry.triangle_random_per_island
                            .size()
                    ? geometry
                          .triangle_random_per_island[i]
                    : 0.0f);
        }
        if (!result.diagnostics.empty()) {
            continue;
        }
        const auto index =
            static_cast<std::uint32_t>(
                data->geometries.size());
        const auto bindless_base =
            index * geometry_bindless_stride;
        const auto material_offset =
            static_cast<std::uint32_t>(
                geometry_materials.size());
        for (const auto material_id :
             geometry.material_slots) {
            const auto material_iter =
                data->material_tags.find(material_id);
            if (material_iter == data->material_tags.end()) {
                diagnose(
                    result.diagnostics,
                    "Geometry '" + geometry.name +
                        "' references an unavailable material.");
                break;
            }
            geometry_materials.emplace_back(
                material_iter->second);
        }
        if (geometry.material_slots.empty()) {
            geometry_materials.emplace_back(0u);
        }
        if (!result.diagnostics.empty()) {
            continue;
        }

        auto &resource =
            data->geometries.emplace_back();
        resource.positions =
            data->device.create_buffer<luisa::float3>(
                upload.positions.size());
        resource.normals =
            data->device.create_buffer<luisa::float3>(
                upload.normals.size());
        resource.uv =
            data->device.create_buffer<luisa::float2>(
                upload.uv.size());
        resource.uv_tangents =
            data->device.create_buffer<luisa::float4>(
                upload.uv_tangents.size());
        resource.generated =
            data->device.create_buffer<luisa::float3>(
                upload.generated.size());
        resource.triangles =
            data->device.create_buffer<Triangle>(
                upload.triangles.size());
        resource.triangle_material_slots =
            data->device.create_buffer<luisa::uint>(
                upload.triangle_material_slots.size());
        resource.triangle_random_per_island =
            data->device.create_buffer<float>(
                upload.triangle_random_per_island.size());
        resource.color_attributes.reserve(
            upload.color_attributes.size());
        resource.mesh = data->device.create_mesh(
            resource.positions, resource.triangles);
        data->heap.emplace_on_update(
            bindless_base, resource.triangles);
        data->heap.emplace_on_update(
            bindless_base + 1u, resource.positions);
        data->heap.emplace_on_update(
            bindless_base + 2u, resource.normals);
        data->heap.emplace_on_update(
            bindless_base + 3u, resource.uv);
        data->heap.emplace_on_update(
            bindless_base + 4u,
            resource.triangle_material_slots);
        data->heap.emplace_on_update(
            bindless_base + 5u,
            resource.generated);
        data->heap.emplace_on_update(
            bindless_base + 6u,
            resource.triangle_random_per_island);
        data->heap.emplace_on_update(
            bindless_base + 7u,
            resource.uv_tangents);
        for (const auto &attribute :
             upload.color_attributes) {
            auto &attribute_resource =
                resource.color_attributes.emplace_back(
                    data->device.create_buffer<luisa::float4>(
                        attribute.values.size()));
            const auto attribute_slot =
                next_attribute_slot++;
            data->heap.emplace_on_update(
                attribute_slot, attribute_resource);
            data->attribute_bindings.emplace_back(
                AttributeBinding{
                    .id = attribute.id,
                    .geometry_index = index,
                    .triangle_slot = bindless_base,
                    .value_slot = attribute_slot});
            stream << attribute_resource.copy_from(
                luisa::span{attribute.values});
        }
        stream << resource.positions.copy_from(
                      luisa::span{upload.positions})
               << resource.normals.copy_from(
                      luisa::span{upload.normals})
               << resource.uv.copy_from(
                      luisa::span{upload.uv})
               << resource.uv_tangents.copy_from(
                      luisa::span{upload.uv_tangents})
               << resource.generated.copy_from(
                      luisa::span{upload.generated})
               << resource.triangles.copy_from(
                      luisa::span{upload.triangles})
               << resource.triangle_material_slots.copy_from(
                      luisa::span{
                          upload.triangle_material_slots})
               << resource.triangle_random_per_island
                      .copy_from(
                          luisa::span{
                              upload
                                  .triangle_random_per_island})
               << resource.mesh.build();
        geometry_indices.emplace(geometry_id, index);
        geometry_gpu.emplace_back(GeometryGpu{
            .bindless_base = bindless_base,
            .material_offset = material_offset,
            .material_count =
                static_cast<std::uint32_t>(
                    std::max<std::size_t>(
                        geometry.material_slots.size(), 1u)),
            .padding = 0u});
    }
    if (!result.diagnostics.empty()) {
        return result;
    }

    std::map<contract::MaterialId, bool> material_may_emit;
    for (const auto &[material_id, material] :
         data->materials.materials()) {
        material_may_emit.emplace(
            material_id,
            std::any_of(
                material.surface_program()
                    ->closure_instructions()
                    .begin(),
                material.surface_program()
                    ->closure_instructions()
                    .end(),
                [](const compiler::ClosureInstruction &closure) {
                    return closure.operation ==
                           compiler::ClosureOperation::emission;
                }));
    }

    luisa::vector<InstanceGpu> instances;
    luisa::vector<luisa::uint> override_materials;
    luisa::vector<EmissiveTriangleGpu> emissive_triangles;
    data->accel = data->device.create_accel();
    for (const auto &[instance_id, instance] :
         snapshot.instances) {
        static_cast<void>(instance_id);
        const auto geometry_iter =
            geometry_indices.find(instance.geometry);
        if (geometry_iter == geometry_indices.end()) {
            diagnose(
                result.diagnostics,
                "Instance '" + instance.name +
                    "' references unavailable geometry.");
            continue;
        }
        const auto override_offset =
            static_cast<std::uint32_t>(
                override_materials.size());
        for (const auto material_id :
             instance.material_overrides) {
            const auto material_iter =
                data->material_tags.find(material_id);
            if (material_iter == data->material_tags.end()) {
                diagnose(
                    result.diagnostics,
                    "Instance '" + instance.name +
                        "' references unavailable override "
                        "material.");
                break;
            }
            override_materials.emplace_back(
                material_iter->second);
        }
        if (!result.diagnostics.empty()) {
            continue;
        }
        const auto instance_index =
            static_cast<std::uint32_t>(instances.size());
        instances.emplace_back(InstanceGpu{
            .geometry_index = geometry_iter->second,
            .override_offset = override_offset,
            .override_count =
                static_cast<std::uint32_t>(
                    instance.material_overrides.size()),
            .object_random = std::clamp(
                instance.random, 0.0f, 1.0f),
            .particle_index = instance.particle_index});
        const auto &geometry =
            snapshot.geometries.at(instance.geometry);
        const auto light_visible =
            instance.visibility_mask ==
                ~std::uint32_t{0u} ||
            (instance.visibility_mask &
             (diffuse_visibility |
              glossy_visibility |
              transmission_visibility)) != 0u;
        if (light_visible) {
            for (std::size_t primitive_index = 0u;
                 primitive_index < geometry.triangles.size();
                 ++primitive_index) {
                const auto material_slot =
                    primitive_index <
                            geometry
                                .triangle_material_slots
                                .size()
                        ? geometry.triangle_material_slots
                              [primitive_index]
                        : 0u;
                std::optional<contract::MaterialId> material_id;
                if (material_slot <
                    instance.material_overrides.size()) {
                    material_id =
                        instance.material_overrides[
                            material_slot];
                } else if (!geometry.material_slots.empty()) {
                    material_id =
                        geometry.material_slots[
                            std::min<std::size_t>(
                                material_slot,
                                geometry.material_slots.size() -
                                    1u)];
                }
                if (!material_id ||
                    !material_may_emit[*material_id]) {
                    continue;
                }
                const auto tag_iter =
                    data->material_tags.find(*material_id);
                if (tag_iter == data->material_tags.end()) {
                    diagnose(
                        result.diagnostics,
                        "Instance '" + instance.name +
                            "' has an unavailable emissive "
                            "material.");
                    break;
                }
                emissive_triangles.emplace_back(
                    EmissiveTriangleGpu{
                        .instance_index = instance_index,
                        .geometry_index =
                            geometry_iter->second,
                        .primitive_index =
                            static_cast<std::uint32_t>(
                                primitive_index),
                        .surface_tag = tag_iter->second});
            }
        }
        const auto visibility =
            instance.visibility_mask ==
                    ~std::uint32_t{0u}
                ? std::uint8_t{0xffu}
                : static_cast<std::uint8_t>(
                      instance.visibility_mask);
        data->accel.emplace_back(
            data->geometries[geometry_iter->second].mesh,
            to_luisa(instance.transform),
            visibility,
            false,
            instance_index);
    }
    if (!result.diagnostics.empty()) {
        return result;
    }

    if (instances.empty()) {
        // Luisa requires at least one TLAS instance. Represent an empty
        // Cycles scene with a backend-only triangle whose visibility mask is
        // zero. It cannot be hit by any Psycles ray and never appears in the
        // logical scene, passes, material tables, or emitter distribution.
        luisa::vector<luisa::float3> dummy_positions;
        dummy_positions.emplace_back(
            luisa::make_float3(0.0f, 0.0f, 0.0f));
        dummy_positions.emplace_back(
            luisa::make_float3(1.0f, 0.0f, 0.0f));
        dummy_positions.emplace_back(
            luisa::make_float3(0.0f, 1.0f, 0.0f));
        luisa::vector<Triangle> dummy_triangles;
        dummy_triangles.emplace_back(
            Triangle{0u, 1u, 2u});
        auto &dummy = data->geometries.emplace_back();
        dummy.positions =
            data->device.create_buffer<luisa::float3>(
                dummy_positions.size());
        dummy.triangles =
            data->device.create_buffer<Triangle>(
                dummy_triangles.size());
        dummy.mesh = data->device.create_mesh(
            dummy.positions, dummy.triangles);
        stream << dummy.positions.copy_from(
                      luisa::span{dummy_positions})
               << dummy.triangles.copy_from(
                      luisa::span{dummy_triangles})
               << dummy.mesh.build();
        data->accel.emplace_back(
            dummy.mesh,
            to_luisa(Mat4f{}),
            std::uint8_t{0u},
            false,
            0u);
    }

    luisa::vector<LightGpu> lights;
    Vec3f background{};
    for (const auto &[light_id, light] :
         snapshot.lights) {
        static_cast<void>(light_id);
        if (light.type == LightType::background) {
            background.x += light.color.x * light.power;
            background.y += light.color.y * light.power;
            background.z += light.color.z * light.power;
            continue;
        }
        lights.emplace_back(LightGpu{
            .type =
                static_cast<std::uint32_t>(light.type),
            .position =
                to_luisa(matrix_translation(light.transform)),
            .axis_x =
                to_luisa(matrix_axis(light.transform, 0u)),
            .axis_y =
                to_luisa(matrix_axis(light.transform, 1u)),
            .axis_z =
                to_luisa(matrix_axis(light.transform, 2u)),
            .color = to_luisa(light.color),
            .power = light.power,
            .size = light.size,
            .spread = light.spread,
            .padding = 0.0f});
    }
    data->background = to_luisa(background);
    data->light_count =
        static_cast<std::uint32_t>(lights.size());
    data->emissive_triangle_count =
        static_cast<std::uint32_t>(
            emissive_triangles.size());
    if (lights.empty()) {
        lights.emplace_back(LightGpu{});
    }
    if (emissive_triangles.empty()) {
        emissive_triangles.emplace_back(
            EmissiveTriangleGpu{});
    }
    if (geometry_materials.empty()) {
        geometry_materials.emplace_back(0u);
    }
    if (override_materials.empty()) {
        override_materials.emplace_back(0u);
    }
    // Empty-world renders are valid Cycles scenes. Luisa buffers cannot be
    // zero-sized, so keep inert storage records while leaving the acceleration
    // structure itself empty; the render kernel only reads these buffers after
    // a committed hit.
    if (geometry_gpu.empty()) {
        geometry_gpu.emplace_back(GeometryGpu{});
    }
    if (instances.empty()) {
        instances.emplace_back(InstanceGpu{});
    }

    data->geometry_buffer =
        data->device.create_buffer<GeometryGpu>(
            geometry_gpu.size());
    data->instance_buffer =
        data->device.create_buffer<InstanceGpu>(
            instances.size());
    data->geometry_material_buffer =
        data->device.create_buffer<luisa::uint>(
            geometry_materials.size());
    data->override_material_buffer =
        data->device.create_buffer<luisa::uint>(
            override_materials.size());
    data->light_buffer =
        data->device.create_buffer<LightGpu>(
            lights.size());
    data->emissive_triangle_buffer =
        data->device.create_buffer<EmissiveTriangleGpu>(
            emissive_triangles.size());
    stream << data->geometry_buffer.copy_from(
                  luisa::span{geometry_gpu})
           << data->instance_buffer.copy_from(
                  luisa::span{instances})
           << data->geometry_material_buffer.copy_from(
                  luisa::span{geometry_materials})
           << data->override_material_buffer.copy_from(
                  luisa::span{override_materials})
           << data->light_buffer.copy_from(
                  luisa::span{lights})
           << data->emissive_triangle_buffer.copy_from(
                  luisa::span{emissive_triangles})
           << data->texture_heap.update()
           << data->heap.update()
           << data->accel.build()
           << synchronize();

    result.scene =
        std::make_unique<LuisaCompiledScene>(
            std::move(data));
    return result;
}

std::unique_ptr<contract::RenderSession>
LuisaPathTracerBackend::create_session(
    const contract::CompiledScene &scene,
    const RenderSettings &settings) {
    const auto &compiled =
        static_cast<const LuisaCompiledScene &>(scene);
    if (settings.full_extent.width == 0u ||
        settings.full_extent.height == 0u) {
        return nullptr;
    }
    if (settings.integrator.use_light_tree) {
        return nullptr;
    }
    if (
        !_options.next_event_estimation &&
        settings.integrator.direct_light_sampling !=
            contract::DirectLightSampling::
                forward_path_tracing) {
        return nullptr;
    }
    if (
        settings.integrator.direct_light_sampling ==
            contract::DirectLightSampling::
                forward_path_tracing &&
        compiled.data()->light_count > 0u) {
        // Analytic lights are not acceleration-structure primitives yet, so
        // a forward-only path cannot reach them.
        return nullptr;
    }
    for (const auto &pass : settings.passes) {
        if (!supported_pass(pass.kind)) {
            return nullptr;
        }
    }
    return std::make_unique<LuisaRenderSession>(
        compiled.data(), _options, settings);
}

}// namespace psycles::luisa_backend
