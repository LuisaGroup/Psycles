#include "../src/luisa/cycles_texture_sampling.h"
#include "../src/luisa/path_tracer_texture_sampling.h"
#include "../src/luisa/surface_image_box.h"

#include "luisa_shader_shape_test_support.h"
#include "luisa_surface_test_support.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;
using psycles::luisa_backend::SurfaceImageBoxInput;
using psycles::test_support::approximately_equal;
using psycles::test_support::xir_instruction_count;

constexpr auto coordinate_count = std::uint32_t{7u};
constexpr auto source_interpolation_mode_count = std::uint32_t{4u};
constexpr auto source_extension_mode_count = std::uint32_t{4u};
constexpr auto comparison_variant_count =
    source_interpolation_mode_count * source_extension_mode_count;
constexpr auto result_stride = std::uint32_t{2u};

class InlineImageBoxTextureSampler final
    : public SurfaceImageBoxTextureSampler {

private:
    Expr<BindlessArray> _textures;
    std::uint32_t _interpolation;
    std::uint32_t _extension;

public:
    InlineImageBoxTextureSampler(
        Expr<BindlessArray> textures,
        std::uint32_t interpolation,
        std::uint32_t extension) noexcept
        : _textures{std::move(textures)},
          _interpolation{interpolation},
          _extension{extension} {}

    [[nodiscard]] Float4 sample(
        Expr<std::uint32_t> texture_handle,
        Float2 uv) const noexcept override {
        return sample_cycles_texture_2d(
            _textures,
            texture_handle,
            uv,
            _interpolation,
            _extension);
    }
};

class PoisonUnusedFaceSampler final
    : public SurfaceImageBoxTextureSampler {

private:
    Float2 _selected_uv;

public:
    explicit PoisonUnusedFaceSampler(Float2 selected_uv) noexcept
        : _selected_uv{std::move(selected_uv)} {}

    [[nodiscard]] Float4 sample(
        Expr<std::uint32_t>,
        Float2 uv) const noexcept override {
        constexpr auto tolerance = 1.0e-6f;
        const auto selected =
            (abs(uv.x - _selected_uv.x) <= tolerance) &
            (abs(uv.y - _selected_uv.y) <= tolerance);
        const auto poison =
            std::numeric_limits<float>::quiet_NaN();
        return select(
            make_float4(poison),
            make_float4(0.25f, 0.5f, 0.75f, 1.0f),
            selected);
    }
};

[[nodiscard]] Kernel1D<BindlessArray, Buffer<luisa::float4>>
make_repeated_sampling_kernel(bool shared) {
    constexpr auto repetition_count = std::uint32_t{8u};
    const auto callables = make_texture_2d_sampling_callables();
    return [callables, shared](
               BindlessVar textures,
               BufferFloat4 output) noexcept {
        CallableTexture2DSamplingProvider provider{
            textures,
            callables};
        for (auto index = std::uint32_t{0u};
             index < repetition_count;
             ++index) {
            const auto uv = make_float2(
                0.17f + 0.031f * static_cast<float>(index),
                0.29f + 0.023f * static_cast<float>(index));
            output.write(
                index,
                shared
                    ? provider.sample(0u, uv, 1u, 0u)
                    : sample_cycles_texture_2d(
                          textures, 0u, uv, 1u, 0u));
        }
    };
}

[[nodiscard]] Kernel1D<BindlessArray, Buffer<luisa::float4>>
make_repeated_image_box_kernel(bool shared) {
    constexpr auto repetition_count = std::uint32_t{8u};
    const auto callables = make_texture_2d_sampling_callables();
    return [callables, shared](
               BindlessVar textures,
               BufferFloat4 output) noexcept {
        CallableTexture2DSamplingProvider provider{
            textures,
            callables};
        InlineImageBoxTextureSampler inline_sampler{
            textures,
            1u,
            0u};
        for (auto index = std::uint32_t{0u};
             index < repetition_count;
             ++index) {
            const SurfaceImageBoxInput input{
                .coordinate = make_float3(
                    0.17f + 0.031f * static_cast<float>(index),
                    0.29f + 0.023f * static_cast<float>(index),
                    0.41f + 0.019f * static_cast<float>(index)),
                .signed_normal = make_float3(
                    0.31f,
                    -0.47f,
                    0.22f),
                .blend = 0.37f,
                .texture_handle = 0u,
                .unassociate_alpha = true,
                .encoded_as_srgb = true};
            output.write(
                index,
                shared
                    ? provider.evaluate(input, 1u, 0u)
                    : evaluate_surface_image_box(
                          input,
                          inline_sampler));
        }
    };
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};

    const auto inline_shape =
        make_repeated_sampling_kernel(false);
    const auto shared_shape =
        make_repeated_sampling_kernel(true);
    const auto inline_instructions =
        xir_instruction_count(inline_shape);
    const auto shared_instructions =
        xir_instruction_count(shared_shape);
    if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
        std::cout
            << "texture sampler 8x XIR: inline="
            << inline_instructions << ", shared="
            << shared_instructions << '\n';
    }
    if (shared_instructions >= inline_instructions) {
        std::cerr
            << "Shared texture sampler did not reduce repeated XIR: inline="
            << inline_instructions << ", shared="
            << shared_instructions << '\n';
        return EXIT_FAILURE;
    }

    const auto inline_box_shape =
        make_repeated_image_box_kernel(false);
    const auto shared_box_shape =
        make_repeated_image_box_kernel(true);
    const auto inline_box_instructions =
        xir_instruction_count(inline_box_shape);
    const auto shared_box_instructions =
        xir_instruction_count(shared_box_shape);
    if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
        std::cout
            << "image BOX 8x XIR: inline="
            << inline_box_instructions << ", shared="
            << shared_box_instructions << '\n';
    }
    if (shared_box_instructions >= inline_box_instructions) {
        std::cerr
            << "Shared Image BOX operation did not reduce repeated XIR: "
               "inline="
            << inline_box_instructions << ", shared="
            << shared_box_instructions << '\n';
        return EXIT_FAILURE;
    }

    // Two independently recorded, structurally identical callables must
    // collapse by their complete AST hash. This is the construction pattern
    // produced by independent surface-operation factories.
    const auto first = make_texture_2d_sampling_callables();
    const auto second = make_texture_2d_sampling_callables();
    Kernel1D hash_reuse = [first, second](
                              BindlessVar textures,
                              BufferFloat4 output) noexcept {
        CallableTexture2DSamplingProvider a{textures, first};
        CallableTexture2DSamplingProvider b{textures, second};
        output.write(0u, a.sample(
                             0u,
                             make_float2(0.25f),
                             1u,
                             0u));
        output.write(1u, b.sample(
                             0u,
                             make_float2(0.75f),
                             1u,
                             0u));
    };
    if (hash_reuse.function()->function()
            .custom_callables()
            .size() != 1u) {
        std::cerr
            << "Independently constructed texture callables were not "
               "deduplicated by complete hash\n";
        return EXIT_FAILURE;
    }

    const auto first_box = make_texture_2d_sampling_callables();
    const auto second_box = make_texture_2d_sampling_callables();
    Kernel1D box_hash_reuse = [first_box, second_box](
                                  BindlessVar textures,
                                  BufferFloat4 output) noexcept {
        CallableTexture2DSamplingProvider a{textures, first_box};
        CallableTexture2DSamplingProvider b{textures, second_box};
        const SurfaceImageBoxInput first_input{
            .coordinate = make_float3(0.25f),
            .signed_normal = make_float3(0.2f, -0.7f, 0.1f),
            .blend = 0.5f,
            .texture_handle = 0u,
            .unassociate_alpha = false,
            .encoded_as_srgb = false};
        const SurfaceImageBoxInput second_input{
            .coordinate = make_float3(0.75f),
            .signed_normal = make_float3(-0.4f, 0.3f, 0.6f),
            .blend = 0.25f,
            .texture_handle = 0u,
            .unassociate_alpha = true,
            .encoded_as_srgb = true};
        output.write(0u, a.evaluate(first_input, 1u, 0u));
        output.write(1u, b.evaluate(second_input, 1u, 0u));
    };
    // The kernel directly references one BOX callable; its raw sampler is a
    // nested dependency owned by that callable. Independently constructed BOX
    // copies must collapse by their complete AST hashes; a name-only or
    // address-only identity cannot pass. Raw-sampler hash reuse is checked by
    // the independent test above.
    if (box_hash_reuse.function()->function()
            .custom_callables()
            .size() != 1u) {
        std::cerr
            << "Independently constructed Image BOX callables were not "
               "deduplicated by complete hash; expected one direct "
               "definition, got "
            << box_hash_reuse.function()->function()
                   .custom_callables()
                   .size()
            << '\n';
        return EXIT_FAILURE;
    }

    const auto callables = make_texture_2d_sampling_callables();
    Kernel1D compare = [callables](
                           BindlessVar textures,
                           BufferFloat2 coordinates,
                           BufferFloat4 output) noexcept {
        const auto coordinate_index = dispatch_x();
        const auto uv = coordinates.read(coordinate_index);
        CallableTexture2DSamplingProvider provider{
            textures,
            callables};
        for (auto interpolation = std::uint32_t{0u};
             interpolation < source_interpolation_mode_count;
             ++interpolation) {
            for (auto extension = std::uint32_t{0u};
                 extension < source_extension_mode_count;
                 ++extension) {
                const auto variant =
                    interpolation * source_extension_mode_count +
                    extension;
                const auto base =
                    (variant * coordinate_count + coordinate_index) *
                    result_stride;
                output.write(
                    base,
                    sample_cycles_texture_2d(
                        textures,
                        0u,
                        uv,
                        interpolation,
                        extension));
                output.write(
                    base + 1u,
                    provider.sample(
                        0u,
                        uv,
                        interpolation,
                        extension));
            }
        }
    };
    if (compare.function()->function()
            .custom_callables()
            .size() !=
        texture_sampling_specialization_count) {
        std::cerr
            << "Texture sampler specialization regression: expected "
            << texture_sampling_specialization_count
            << " definitions, got "
            << compare.function()->function()
                   .custom_callables()
                   .size()
            << '\n';
        return EXIT_FAILURE;
    }

    Kernel1D compare_box = [callables](
                               BindlessVar textures,
                               BufferFloat3 box_coordinates,
                               BufferFloat3 box_normals,
                               BufferFloat box_blends,
                               BufferUInt box_flags,
                               BufferFloat4 output) noexcept {
        const auto input_index = dispatch_x();
        const auto flags = box_flags.read(input_index);
        const SurfaceImageBoxInput input{
            .coordinate = box_coordinates.read(input_index),
            .signed_normal = box_normals.read(input_index),
            .blend = box_blends.read(input_index),
            .texture_handle = 0u,
            .unassociate_alpha = (flags & 1u) != 0u,
            .encoded_as_srgb = (flags & 2u) != 0u};
        CallableTexture2DSamplingProvider provider{
            textures,
            callables};
        for (auto interpolation = std::uint32_t{0u};
             interpolation < source_interpolation_mode_count;
             ++interpolation) {
            for (auto extension = std::uint32_t{0u};
                 extension < source_extension_mode_count;
                 ++extension) {
                InlineImageBoxTextureSampler inline_sampler{
                    textures,
                    interpolation,
                    extension};
                const auto variant =
                    interpolation * source_extension_mode_count +
                    extension;
                const auto base =
                    (variant * dispatch_size_x() + input_index) *
                    result_stride;
                output.write(
                    base,
                    evaluate_surface_image_box(
                        input,
                        inline_sampler));
                output.write(
                    base + 1u,
                    provider.evaluate(
                        input,
                        interpolation,
                        extension));
            }
        }
    };
    // The complete authored 4 x 4 domain has exactly 12 semantic BOX
    // specializations: Cubic and Smart share a filter, while every extension
    // remains distinct. This is the converse of the independent-construction
    // reuse check above. It proves that the outer callable hash observes the
    // complete nested sampler dependency instead of merging equal outer
    // shapes whose texture filters differ.
    if (compare_box.function()->function()
            .custom_callables()
            .size() !=
        texture_sampling_specialization_count) {
        std::cerr
            << "Image BOX specialization quotient regression: expected "
            << texture_sampling_specialization_count
            << " distinct definitions, got "
            << compare_box.function()->function()
                   .custom_callables()
                   .size()
            << '\n';
        return EXIT_FAILURE;
    }

    // A zero BOX weight makes the corresponding texture read semantically
    // dead. Multiplying an eagerly fetched NaN by zero would nevertheless
    // poison the result, so this runtime-input counterexample distinguishes
    // Cycles' guarded face reads from an algebraically tempting but invalid
    // unconditional weighted sum.
    Kernel1D guarded_box = [](
                               BufferFloat3 box_coordinates,
                               BufferFloat3 box_normals,
                               BufferFloat box_blends,
                               BufferFloat4 output) noexcept {
        const auto input_index = dispatch_x();
        const SurfaceImageBoxInput input{
            .coordinate = box_coordinates.read(input_index),
            .signed_normal = box_normals.read(input_index),
            .blend = box_blends.read(input_index),
            .texture_handle = 0u,
            .unassociate_alpha = false,
            .encoded_as_srgb = false};
        // coordinate=(0.2, 0.3, 0.4), +X face -> UV=(0.3, 0.4),
        // then host-image row-order conversion -> (0.3, 0.6).
        PoisonUnusedFaceSampler sampler{make_float2(0.3f, 0.6f)};
        output.write(
            input_index,
            evaluate_surface_image_box(input, sampler));
    };

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    constexpr auto width = std::uint32_t{3u};
    constexpr auto height = std::uint32_t{2u};
    constexpr std::array pixels{
        luisa::float4{0.10f, 0.20f, 0.30f, 0.40f},
        luisa::float4{0.50f, 0.60f, 0.70f, 0.80f},
        luisa::float4{0.90f, 1.00f, 0.10f, 0.20f},
        luisa::float4{0.30f, 0.40f, 0.50f, 0.60f},
        luisa::float4{0.70f, 0.80f, 0.90f, 1.00f},
        luisa::float4{0.20f, 0.30f, 0.40f, 0.50f}};
    constexpr std::array coordinates{
        luisa::float2{0.10f, 0.10f},
        luisa::float2{0.45f, 0.25f},
        luisa::float2{0.90f, 0.80f},
        luisa::float2{-0.10f, 0.35f},
        luisa::float2{1.10f, 0.65f},
        luisa::float2{-1.00f, 1.00f},
        luisa::float2{0.50f, 1.25f}};

    struct BoxCase {
        luisa::float3 coordinate;
        luisa::float3 signed_normal;
        float blend;
    };
    constexpr std::array box_cases{
        BoxCase{{0.13f, 0.27f, 0.41f}, {1.0f, 0.0f, 0.0f}, 0.0f},
        BoxCase{{0.83f, 0.67f, 0.51f}, {-1.0f, 0.0f, 0.0f}, 0.2f},
        BoxCase{{-0.2f, 1.1f, 0.31f}, {0.0f, 1.0f, 0.0f}, 0.4f},
        BoxCase{{1.2f, -0.1f, 0.71f}, {0.0f, -1.0f, 0.0f}, 0.6f},
        BoxCase{{0.23f, 0.47f, 1.21f}, {0.0f, 0.0f, 1.0f}, 0.8f},
        BoxCase{{0.73f, 0.17f, -0.21f}, {0.0f, 0.0f, -1.0f}, 1.0f},
        BoxCase{{0.33f, 0.57f, 0.81f}, {0.7f, 0.3f, 0.0f}, 0.5f},
        BoxCase{{0.93f, 0.37f, 0.11f}, {0.0f, 0.35f, 0.65f}, 0.5f},
        BoxCase{{0.43f, 0.77f, 0.21f}, {0.4f, 0.0f, 0.6f}, 0.5f},
        BoxCase{{0.53f, 0.87f, 0.61f}, {1.0f, 1.0f, 1.0f}, 1.0f},
        BoxCase{{0.63f, 0.97f, 0.91f}, {0.0f, 0.0f, 0.0f}, 0.0f}};
    constexpr auto box_flag_count = std::uint32_t{4u};
    constexpr auto box_input_count =
        static_cast<std::uint32_t>(box_cases.size()) *
        box_flag_count;
    std::vector<luisa::float3> box_coordinates;
    std::vector<luisa::float3> box_normals;
    std::vector<float> box_blends;
    std::vector<std::uint32_t> box_flags;
    box_coordinates.reserve(box_input_count);
    box_normals.reserve(box_input_count);
    box_blends.reserve(box_input_count);
    box_flags.reserve(box_input_count);
    for (const auto &box_case : box_cases) {
        for (auto flags = std::uint32_t{0u};
             flags < box_flag_count;
             ++flags) {
            box_coordinates.emplace_back(box_case.coordinate);
            box_normals.emplace_back(box_case.signed_normal);
            box_blends.emplace_back(box_case.blend);
            box_flags.emplace_back(flags);
        }
    }

    auto image = device.create_image<float>(
        PixelStorage::FLOAT4, width, height);
    auto textures = device.create_bindless_array(1u);
    textures.emplace_on_update(
        0u, image, Sampler::linear_point_repeat());
    auto coordinate_buffer =
        device.create_buffer<luisa::float2>(coordinate_count);
    constexpr auto result_count =
        comparison_variant_count * coordinate_count * result_stride;
    auto output = device.create_buffer<luisa::float4>(result_count);
    auto box_coordinate_buffer =
        device.create_buffer<luisa::float3>(box_input_count);
    auto box_normal_buffer =
        device.create_buffer<luisa::float3>(box_input_count);
    auto box_blend_buffer =
        device.create_buffer<float>(box_input_count);
    auto box_flag_buffer =
        device.create_buffer<std::uint32_t>(box_input_count);
    constexpr auto box_result_count =
        comparison_variant_count * box_input_count * result_stride;
    auto box_output =
        device.create_buffer<luisa::float4>(box_result_count);
    auto shader = device.compile(compare);
    auto box_shader = device.compile(compare_box);
    auto guarded_box_shader = device.compile(guarded_box);
    std::vector<luisa::float4> actual(result_count);
    std::vector<luisa::float4> actual_box(box_result_count);
    constexpr std::array guarded_coordinate{
        luisa::float3{0.2f, 0.3f, 0.4f}};
    constexpr std::array guarded_normal{
        luisa::float3{1.0f, 0.0f, 0.0f}};
    constexpr std::array guarded_blend{0.0f};
    auto guarded_coordinate_buffer =
        device.create_buffer<luisa::float3>(1u);
    auto guarded_normal_buffer =
        device.create_buffer<luisa::float3>(1u);
    auto guarded_blend_buffer = device.create_buffer<float>(1u);
    auto guarded_output = device.create_buffer<luisa::float4>(1u);
    std::array<luisa::float4, 1u> actual_guarded{};
    stream << image.copy_from(luisa::span{pixels})
           << coordinate_buffer.copy_from(luisa::span{coordinates})
           << textures.update()
           << shader(textures, coordinate_buffer, output)
                  .dispatch(coordinate_count)
           << output.copy_to(luisa::span{actual})
           << box_coordinate_buffer.copy_from(
                  luisa::span{box_coordinates})
           << box_normal_buffer.copy_from(
                  luisa::span{box_normals})
           << box_blend_buffer.copy_from(
                  luisa::span{box_blends})
           << box_flag_buffer.copy_from(
                  luisa::span{box_flags})
           << box_shader(
                  textures,
                  box_coordinate_buffer,
                  box_normal_buffer,
                  box_blend_buffer,
                  box_flag_buffer,
                  box_output)
                  .dispatch(box_input_count)
           << box_output.copy_to(luisa::span{actual_box})
           << guarded_coordinate_buffer.copy_from(
                  luisa::span{guarded_coordinate})
           << guarded_normal_buffer.copy_from(
                  luisa::span{guarded_normal})
           << guarded_blend_buffer.copy_from(
                  luisa::span{guarded_blend})
           << guarded_box_shader(
                  guarded_coordinate_buffer,
                  guarded_normal_buffer,
                  guarded_blend_buffer,
                  guarded_output)
                  .dispatch(1u)
           << guarded_output.copy_to(luisa::span{actual_guarded})
           << synchronize();

    for (auto variant = std::uint32_t{0u};
         variant < comparison_variant_count;
         ++variant) {
        for (auto coordinate = std::uint32_t{0u};
             coordinate < coordinate_count;
             ++coordinate) {
            const auto base =
                (variant * coordinate_count + coordinate) *
                result_stride;
            if (!approximately_equal(
                    actual[base], actual[base + 1u], 2.0e-6f)) {
                const auto direct = actual[base];
                const auto shared = actual[base + 1u];
                std::cerr
                    << "Texture sampler mismatch on " << backend
                    << ", variant " << variant
                    << ", coordinate " << coordinate
                    << ": direct={" << direct.x << ", "
                    << direct.y << ", " << direct.z << ", "
                    << direct.w << "}, shared={" << shared.x
                    << ", " << shared.y << ", " << shared.z
                    << ", " << shared.w << "}\n";
                return EXIT_FAILURE;
            }
        }
    }
    for (auto variant = std::uint32_t{0u};
         variant < comparison_variant_count;
         ++variant) {
        for (auto input = std::uint32_t{0u};
             input < box_input_count;
             ++input) {
            const auto base =
                (variant * box_input_count + input) *
                result_stride;
            if (!approximately_equal(
                    actual_box[base],
                    actual_box[base + 1u],
                    2.0e-6f)) {
                const auto direct = actual_box[base];
                const auto shared = actual_box[base + 1u];
                std::cerr
                    << "Image BOX mismatch on " << backend
                    << ", variant " << variant
                    << ", input " << input
                    << ": direct={" << direct.x << ", "
                    << direct.y << ", " << direct.z << ", "
                    << direct.w << "}, shared={" << shared.x
                    << ", " << shared.y << ", " << shared.z
                    << ", " << shared.w << "}\n";
                return EXIT_FAILURE;
            }
        }
    }
    const auto guarded = actual_guarded.front();
    if (!std::isfinite(guarded.x) ||
        !std::isfinite(guarded.y) ||
        !std::isfinite(guarded.z) ||
        !std::isfinite(guarded.w) ||
        !approximately_equal(
            guarded,
            luisa::float4{0.25f, 0.5f, 0.75f, 1.0f},
            1.0e-6f)) {
        std::cerr
            << "Image BOX sampled a zero-weight poisoned face on "
            << backend << ": actual={" << guarded.x << ", "
            << guarded.y << ", " << guarded.z << ", "
            << guarded.w << "}\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
