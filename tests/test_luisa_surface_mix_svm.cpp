#include "luisa_shader_shape_test_support.h"
#include "luisa_surface_test_support.h"
#include "path_tracer_surface_value_program.h"
#include "surface_mix.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
using psycles::test_support::approximately_equal;
using psycles::test_support::ParameterShaderServices;
using psycles::test_support::require_bounded_xir;

constexpr auto operation_count =
    static_cast<std::uint32_t>(BlendOperation::value) + 1u;
constexpr auto flag_count = 4u;
constexpr auto case_count = operation_count * flag_count;

[[nodiscard]] consteval auto make_immediate_domain() {
    std::array<std::uint16_t, case_count> result{};
    for (auto flags = std::uint32_t{0u}; flags < flag_count; ++flags) {
        for (auto operation = std::uint32_t{0u}; operation < operation_count;
             ++operation) {
            const auto index = flags * operation_count + operation;
            result[index] = static_cast<std::uint16_t>(
                operation |
                ((flags & 1u) != 0u ? surface_value_mix_factor_clamp_bit : 0u) |
                ((flags & 2u) != 0u ? surface_value_mix_result_clamp_bit : 0u));
        }
    }
    return result;
}

inline constexpr auto immediate_domain = make_immediate_domain();
inline constexpr std::array<std::uint16_t, 1u> mix_only_domain{0u};

using SvmStackMutationCallable =
    Callable<void(luisa::uint, SurfaceValueStackBank &)>;

[[nodiscard]] Kernel1D<Buffer<luisa::float4>, Buffer<luisa::ulong>>
make_svm_bank_reference_kernel() {
    return [](BufferFloat4 output, BufferULong exact_output) noexcept {
        Local<float> stack{SurfaceValueRuntime::stack_capacity};
        SvmStackMutationCallable mutate = [](
            UInt index, Var<SurfaceValueStackBank> &stack) noexcept {
            const auto storage =
                luisa::compute::detail::Ref<SurfaceValueStackBank>{
                    stack.expression()};
            SurfaceValueLocalScalarView scalars{stack.expression()};
            SurfaceValueLocalVectorView vectors{scalars};
            SurfaceValueLocalUnsignedIntegerView unsigned_integers{scalars};
            scalars.write(index, 3.25f);
            vectors.write(index + 1u, make_float3(1.0f, 2.0f, 3.0f));
            unsigned_integers.write(index + 4u,
                                    ULong{0x0102030405060708ull});
        };
        const auto storage =
            luisa::compute::detail::Ref<SurfaceValueStackBank>{
                stack.expression()};
        mutate(3u, storage);
        const SurfaceValueLocalScalarView scalars{stack.expression()};
        const SurfaceValueLocalVectorView vectors{scalars};
        const SurfaceValueLocalUnsignedIntegerView unsigned_integers{scalars};
        const auto vector = vectors.read(4u);
        output.write(
            0u,
            make_float4(
                scalars.read(3u), vector.x, vector.y, vector.z));
        exact_output.write(0u, unsigned_integers.read(7u));
    };
}

[[nodiscard]] Kernel1D<Buffer<luisa::float4>, Buffer<luisa::float3>>
make_callable_reuse_shape(bool include_distinct_domain) {
    return [include_distinct_domain](BufferFloat4 parameters,
                                     BufferFloat3 output) noexcept {
        ParameterShaderServices services{parameters};
        const auto first =
            evaluate_surface_mix_svm(services,
                                     0u,
                                     immediate_domain,
                                     0.25f,
                                     make_float3(0.1f),
                                     make_float3(0.9f));
        const auto second = evaluate_surface_mix_svm(
            services,
            0u,
            include_distinct_domain
                ? std::span<const std::uint16_t>{mix_only_domain}
                : std::span<const std::uint16_t>{immediate_domain},
            0.75f,
            make_float3(0.2f),
            make_float3(0.8f));
        output.write(0u, first + second);
    };
}

[[nodiscard]] Kernel1D<Buffer<luisa::float4>, Buffer<luisa::float3>>
make_runtime_kernel() {
    return [](BufferFloat4 parameters, BufferFloat3 output) noexcept {
        const auto index = dispatch_x();
        const auto operation = index % operation_count;
        const auto flags = index / operation_count;
        UInt immediate = operation;
        immediate |=
            select(0u, surface_value_mix_factor_clamp_bit, (flags & 1u) != 0u);
        immediate |=
            select(0u, surface_value_mix_result_clamp_bit, (flags & 2u) != 0u);
        ParameterShaderServices services{parameters};
        output.write(index,
                     evaluate_surface_mix_svm(services,
                                              immediate,
                                              immediate_domain,
                                              1.35f,
                                              make_float3(-0.2f, 0.3f, 1.4f),
                                              make_float3(0.8f, -0.1f, 0.6f)));
    };
}

[[nodiscard]] Kernel1D<Buffer<luisa::float4>, Buffer<luisa::float3>>
make_static_reference_kernel() {
    return [](BufferFloat4 parameters, BufferFloat3 output) noexcept {
        const auto index = dispatch_x();
        const auto operation = index % operation_count;
        const auto flags = index / operation_count;
        ParameterShaderServices services{parameters};
        auto factor = def(1.35f);
        factor = select(factor, clamp(factor, 0.0f, 1.0f), (flags & 1u) != 0u);
        auto result = def(make_float3(0.0f));
        luisa::compute::detail::SwitchStmtBuilder{operation} % [&] {
            for (auto mode = std::uint32_t{0u}; mode < operation_count;
                 ++mode) {
                luisa::compute::detail::SwitchCaseStmtBuilder{mode} %
                    [&, mode] {
                        result = evaluate_surface_mix_operation(
                            services,
                            static_cast<BlendOperation>(mode),
                            factor,
                            make_float3(-0.2f, 0.3f, 1.4f),
                            make_float3(0.8f, -0.1f, 0.6f));
                    };
            }
            luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                luisa::compute::dsl::unreachable(
                    "invalid static Mix regression operation");
            };
        };
        result = select(result, clamp(result, 0.0f, 1.0f), (flags & 2u) != 0u);
        output.write(index, result);
    };
}

} // namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();

    const auto bank_reference_kernel = make_svm_bank_reference_kernel();
    if (bank_reference_kernel.function()
            ->function()
            .custom_callables()
            .size() != 1u) {
        std::cerr << "whole-node SVM lane-stack mutation was not recorded as "
                     "one callable\n";
        return EXIT_FAILURE;
    }

    const auto reused_callable = make_callable_reuse_shape(false);
    if (reused_callable.function()->function().custom_callables().size() != 1u) {
        std::cerr << "independently constructed Mix SVM callables were not "
                     "deduplicated by complete AST hash\n";
        return EXIT_FAILURE;
    }
    const auto distinct_callable = make_callable_reuse_shape(true);
    if (distinct_callable.function()->function().custom_callables().size() !=
        2u) {
        std::cerr << "Mix SVM callable hashing omitted its reachable-operation "
                     "domain\n";
        return EXIT_FAILURE;
    }

    const auto runtime_kernel = make_runtime_kernel();
    if (!require_bounded_xir("surface Mix SVM", runtime_kernel, 4096u)) {
        return EXIT_FAILURE;
    }
    const ShaderOption uncached{.enable_cache = false};
    auto runtime_shader = device.compile(runtime_kernel, uncached);
    auto reference_shader =
        device.compile(make_static_reference_kernel(), uncached);
    auto parameters = device.create_buffer<luisa::float4>(1u);
    auto bank_output_buffer = device.create_buffer<luisa::float4>(1u);
    auto stack_uint64_output_buffer =
        device.create_buffer<luisa::ulong>(1u);
    auto actual_buffer = device.create_buffer<luisa::float3>(case_count);
    auto expected_buffer = device.create_buffer<luisa::float3>(case_count);
    const std::array parameter_data{luisa::make_float4(0.0f)};
    std::vector<luisa::float3> actual(case_count);
    std::vector<luisa::float3> expected(case_count);
    std::array<luisa::float4, 1u> bank_output{};
    std::array<luisa::ulong, 1u> stack_uint64_output{};
    stream << parameters.copy_from(luisa::span{parameter_data})
           << device.compile(bank_reference_kernel, uncached)(
                  bank_output_buffer, stack_uint64_output_buffer)
                  .dispatch(1u)
           << bank_output_buffer.copy_to(luisa::span{bank_output})
           << stack_uint64_output_buffer.copy_to(
                  luisa::span{stack_uint64_output})
           << runtime_shader(parameters, actual_buffer).dispatch(case_count)
           << reference_shader(parameters, expected_buffer).dispatch(case_count)
           << actual_buffer.copy_to(luisa::span{actual})
           << expected_buffer.copy_to(luisa::span{expected}) << synchronize();

    const auto &bank_vector = bank_output[0u];
    if (!approximately_equal(bank_vector.x, 3.25f) ||
        !approximately_equal(bank_vector.y, 1.0f) ||
        !approximately_equal(bank_vector.z, 2.0f) ||
        !approximately_equal(bank_vector.w, 3.0f) ||
        stack_uint64_output[0u] != 0x0102030405060708ull) {
        std::cerr << "whole-node SVM lane-stack reference ABI mismatch on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    for (auto index = std::uint32_t{0u}; index < case_count; ++index) {
        if (!approximately_equal(actual[index].x, expected[index].x) ||
            !approximately_equal(actual[index].y, expected[index].y) ||
            !approximately_equal(actual[index].z, expected[index].z)) {
            std::cerr << "surface Mix SVM mismatch on " << backend
                      << ", operation=" << index % operation_count
                      << ", flags=" << index / operation_count << '\n';
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
