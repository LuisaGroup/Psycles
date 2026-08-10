#include "path_tracer_surface_closure_setup.h"
#include "principled_base_component.h"

#include <psycles/luisa/cycles_bsdf_tables.h>

#include "luisa_surface_test_support.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
using psycles::test_support::approximately_equal;

class TableShaderServices final : public ShaderServices {

private:
    const BufferFloat &_cycles_bsdf_tables;

public:
    explicit TableShaderServices(
        const BufferFloat &cycles_bsdf_tables) noexcept
        : _cycles_bsdf_tables{cycles_bsdf_tables} {}

    [[nodiscard]] Float parameter_float(
        Expr<std::uint32_t>, Expr<std::uint32_t>) const noexcept override {
        return 0.0f;
    }

    [[nodiscard]] Float3 parameter_float3(
        Expr<std::uint32_t>, Expr<std::uint32_t>) const noexcept override {
        return make_float3(0.0f);
    }

    [[nodiscard]] ULong parameter_uint64(
        Expr<std::uint32_t>, Expr<std::uint32_t>) const noexcept override {
        return 0u;
    }

    [[nodiscard]] Float4 texture_2d(
        Expr<std::uint32_t>,
        Expr<luisa::float2>,
        Expr<luisa::float2>,
        Expr<luisa::float2>,
        std::uint32_t,
        std::uint32_t) const noexcept override {
        return make_float4(0.0f);
    }

    [[nodiscard]] ShaderAttribute attribute(
        Expr<luisa::ulong>, const SurfacePoint &) const noexcept override {
        return ShaderAttribute::missing();
    }

    [[nodiscard]] Float cycles_bsdf_data(
        Expr<std::uint32_t> index) const noexcept override {
        return _cycles_bsdf_tables->read(index);
    }

    [[nodiscard]] Float3 xyz_to_rgb(
        Expr<luisa::float3> value) const noexcept override {
        return Float3{value};
    }

    [[nodiscard]] Float3 rec709_to_rgb(
        Expr<luisa::float3> value) const noexcept override {
        return Float3{value};
    }

    [[nodiscard]] Float3 nishita_sky(
        Expr<std::uint32_t>,
        std::uint32_t,
        Expr<luisa::float3>,
        Expr<float>,
        Expr<float>,
        Expr<float>,
        Expr<float>) const noexcept override {
        return make_float3(0.0f);
    }
};

constexpr std::uint32_t case_count = 6u;
constexpr std::uint32_t records_per_result = 6u;
constexpr std::uint32_t records_per_case = 4u * records_per_result;

void write_result(
    const BufferFloat4 &output,
    UInt base,
    const PrincipledDielectricSetupResult &result) noexcept {
    output->write(
        base + 0u, make_float4(result.weight, result.allocation_weight));
    output->write(
        base + 1u, make_float4(result.albedo, result.sample_weight));
    output->write(base + 2u, make_float4(result.normal, result.ior));
    output->write(base + 3u, make_float4(result.color, 0.0f));
    output->write(
        base + 4u, make_float4(result.evaluation_scale, 0.0f));
    output->write(base + 5u, make_float4(result.lower_weight, 0.0f));
}

[[nodiscard]] bool finite(luisa::float4 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z) && std::isfinite(value.w);
}

}// namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
    const auto callables = make_surface_closure_setup_callables();

    Kernel1D compare = [callables](
                           BufferFloat cycles_bsdf_tables,
                           BufferFloat4 output) noexcept {
        const auto case_index = dispatch_x();
        const auto lower_weight = select(
            make_float3(0.73f, 0.41f, 0.19f),
            make_float3(8.0e-6f),
            case_index == 1u);
        const auto glossy_normal = normalize(
            select(make_float3(0.0f, 0.0f, 1.0f),
                   make_float3(0.2f, -0.1f, 0.97f),
                   case_index == 5u));
        const auto incoming_cosine = select(
            0.73f, 0.05f, case_index == 4u);
        const auto incoming = normalize(make_float3(
            sqrt(max(1.0f - incoming_cosine * incoming_cosine, 0.0f)),
            0.0f,
            incoming_cosine));
        const auto roughness = select(
            0.37f, 0.91f, case_index == 3u);
        const auto ior = select(
            select(1.47f, 1.0f, case_index == 2u),
            0.82f,
            case_index == 3u);
        const auto specular_ior_level = select(
            0.5f, 0.23f, case_index == 3u);
        const auto specular_tint = select(
            make_float3(0.91f, 0.62f, 0.27f),
            make_float3(0.19f, 0.77f, 1.0f),
            case_index == 5u);
        const auto reflective_caustics = case_index != 4u;
        const TableShaderServices direct_services{cycles_bsdf_tables};
        const CallableSurfaceClosureSetupProvider callable_provider{
            cycles_bsdf_tables, callables};
        const auto base = case_index * records_per_case;

        const auto evaluate = [&](bool preserve_ggx_energy,
                                  std::uint32_t direct_offset,
                                  std::uint32_t callable_offset) noexcept {
            const auto input = PrincipledDielectricSetupInput{
                .lower_weight = lower_weight,
                .normal = glossy_normal,
                .incoming = incoming,
                .surface_shading_normal =
                    make_float3(0.0f, 0.0f, 1.0f),
                .surface_geometric_normal =
                    make_float3(0.0f, 0.0f, 1.0f),
                .roughness = roughness,
                .ior = ior,
                .specular_ior_level = specular_ior_level,
                .specular_tint = specular_tint,
                .use_bump_map_correction = case_index == 5u,
                .preserve_ggx_energy = preserve_ggx_energy};
            write_result(
                output,
                base + direct_offset,
                setup_principled_dielectric(
                    direct_services,
                    populate_principled_dielectric(input),
                    reflective_caustics));
            write_result(
                output,
                base + callable_offset,
                callable_provider.principled_dielectric(
                    input, reflective_caustics));
        };
        evaluate(false, 0u, records_per_result);
        evaluate(true, 2u * records_per_result, 3u * records_per_result);
    };

    // One callable per immutable energy-preservation specialization must be
    // shared by every invocation in the kernel. This is an AST-shape guard;
    // the device comparison below guards the typed ABI and numerical result.
    if (compare.function()->function().custom_callables().size() != 2u) {
        std::cerr << "Principled setup callable reuse regression: expected 2 "
                     "specializations, got "
                  << compare.function()->function().custom_callables().size()
                  << '\n';
        return EXIT_FAILURE;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    std::vector<float> cycles_table(
        cycles45_tables::total_size, 0.5f);
    auto cycles_table_buffer =
        device.create_buffer<float>(cycles_table.size());
    auto output_buffer = device.create_buffer<luisa::float4>(
        case_count * records_per_case);
    auto shader = device.compile(compare);
    std::array<luisa::float4, case_count * records_per_case> actual{};
    stream << cycles_table_buffer.copy_from(luisa::span{cycles_table})
           << shader(cycles_table_buffer, output_buffer).dispatch(case_count)
           << output_buffer.copy_to(luisa::span{actual})
           << synchronize();

    const auto compare_pair = [&](std::uint32_t case_index,
                                  std::uint32_t direct_offset,
                                  std::uint32_t callable_offset) noexcept {
        for (auto record = 0u; record < records_per_result; ++record) {
            const auto direct = actual[
                case_index * records_per_case + direct_offset + record];
            const auto shared = actual[
                case_index * records_per_case + callable_offset + record];
            if (!finite(direct) || !finite(shared) ||
                !approximately_equal(direct, shared, 2.0e-6f)) {
                std::cerr << "Principled setup callable mismatch on "
                          << backend << ", case " << case_index
                          << ", record " << record << ": direct={"
                          << direct.x << ", " << direct.y << ", "
                          << direct.z << ", " << direct.w << "}, shared={"
                          << shared.x << ", " << shared.y << ", "
                          << shared.z << ", " << shared.w << "}\n";
                return false;
            }
        }
        return true;
    };
    for (auto case_index = 0u; case_index < case_count; ++case_index) {
        if (!compare_pair(case_index, 0u, records_per_result) ||
            !compare_pair(case_index,
                          2u * records_per_result,
                          3u * records_per_result)) {
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
