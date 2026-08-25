#include "luisa_shader_shape_test_support.h"
#include "luisa_surface_test_support.h"
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

using ScalarSvmBank = std::array<float, 8u>;
using VectorSvmBank = std::array<luisa::float3, 12u>;
using SvmBankMutationCallable =
    Callable<void(luisa::uint, ScalarSvmBank &, VectorSvmBank &,
                  luisa::ulong &)>;

[[nodiscard]] Kernel1D<Buffer<luisa::float4>>
make_svm_bank_reference_kernel() {
    return [](BufferFloat4 output) noexcept {
        Local<float> scalars{std::tuple_size_v<ScalarSvmBank>};
        Local<luisa::float3> vectors{std::tuple_size_v<VectorSvmBank>};
        Local<luisa::ulong> unsigned_integers{1u};
        SvmBankMutationCallable mutate = [](
            UInt index,
            Var<ScalarSvmBank> &scalar_bank,
            Var<VectorSvmBank> &vector_bank,
            ULong &unsigned_integer_bank) noexcept {
            scalar_bank[index] = 3.25f;
            vector_bank[index + 1u] = make_float3(1.0f, 2.0f, 3.0f);
            unsigned_integer_bank = 0x0102030405060708ull;
        };
        mutate(3u,
               luisa::compute::detail::Ref<ScalarSvmBank>{
                   scalars.expression()},
               luisa::compute::detail::Ref<VectorSvmBank>{
                   vectors.expression()},
               luisa::compute::detail::Ref<luisa::ulong>{
                   unsigned_integers.expression()});
        output.write(
            0u,
            make_float4(
                scalars.read(3u),
                vectors.read(4u).x,
                vectors.read(4u).y,
                vectors.read(4u).z));
        output.write(
            1u,
            make_float4(
                cast<float>(unsigned_integers.read(0u) & 0xffull),
                0.0f,
                0.0f,
                0.0f));
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
        std::cerr << "whole-node SVM local-bank mutation was not recorded as "
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
    auto bank_output_buffer = device.create_buffer<luisa::float4>(2u);
    auto actual_buffer = device.create_buffer<luisa::float3>(case_count);
    auto expected_buffer = device.create_buffer<luisa::float3>(case_count);
    const std::array parameter_data{luisa::make_float4(0.0f)};
    std::vector<luisa::float3> actual(case_count);
    std::vector<luisa::float3> expected(case_count);
    std::array<luisa::float4, 2u> bank_output{};
    stream << parameters.copy_from(luisa::span{parameter_data})
           << device.compile(bank_reference_kernel, uncached)(
                  bank_output_buffer)
                  .dispatch(1u)
           << bank_output_buffer.copy_to(luisa::span{bank_output})
           << runtime_shader(parameters, actual_buffer).dispatch(case_count)
           << reference_shader(parameters, expected_buffer).dispatch(case_count)
           << actual_buffer.copy_to(luisa::span{actual})
           << expected_buffer.copy_to(luisa::span{expected}) << synchronize();

    const auto &bank_vector = bank_output[0u];
    if (!approximately_equal(bank_vector.x, 3.25f) ||
        !approximately_equal(bank_vector.y, 1.0f) ||
        !approximately_equal(bank_vector.z, 2.0f) ||
        !approximately_equal(bank_vector.w, 3.0f) ||
        !approximately_equal(bank_output[1u].x, 8.0f)) {
        std::cerr << "whole-node SVM local-bank reference ABI mismatch on "
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
