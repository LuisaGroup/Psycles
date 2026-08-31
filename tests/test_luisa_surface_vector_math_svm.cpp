#include "luisa_shader_shape_test_support.h"
#include "luisa_surface_test_support.h"
#include "surface_vector_math.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler;
using namespace psycles::luisa_backend::detail;
using psycles::test_support::approximately_equal;
using psycles::test_support::require_bounded_xir;

[[nodiscard]] consteval auto make_immediate_domain() {
    std::array<std::uint16_t, vector_math_operation_count> result{};
    for (auto index = std::uint32_t{0u};
         index < vector_math_operation_count;
         ++index) {
        result[index] = static_cast<std::uint16_t>(index);
    }
    return result;
}

inline constexpr auto immediate_domain = make_immediate_domain();
inline constexpr std::array<std::uint16_t, 1u> add_only_domain{
    static_cast<std::uint16_t>(VectorMathOperation::add)};
inline constexpr std::array<std::uint16_t, 2u> power_round_domain{
    static_cast<std::uint16_t>(VectorMathOperation::power),
    static_cast<std::uint16_t>(VectorMathOperation::round)};
inline constexpr std::array<std::uint16_t, 1u> cycles_normalize_domain{
    static_cast<std::uint16_t>(VectorMathOperation::cycles_normalize)};
inline constexpr auto input_a = luisa::float3{0.75f, -0.4f, 1.2f};
inline constexpr auto input_b = luisa::float3{0.5f, 1.25f, -0.8f};
inline constexpr auto input_c = luisa::float3{-0.2f, 0.7f, 0.3f};
inline constexpr auto input_scale = 0.65f;

[[nodiscard]] Kernel1D<Buffer<float>> make_value_callable_reuse_shape(
    bool include_distinct_domain) {
    return [include_distinct_domain](BufferFloat output) noexcept {
        output.write(
            0u,
            evaluate_surface_vector_math_value_svm(
                static_cast<std::uint32_t>(VectorMathOperation::length),
                immediate_domain,
                input_a,
                input_b,
                input_c,
                input_scale) +
                evaluate_surface_vector_math_value_svm(
                    static_cast<std::uint32_t>(
                        include_distinct_domain
                            ? VectorMathOperation::add
                            : VectorMathOperation::dot_product),
                    include_distinct_domain
                        ? std::span<const std::uint16_t>{add_only_domain}
                        : std::span<const std::uint16_t>{immediate_domain},
                    input_a,
                    input_b,
                    input_c,
                    input_scale));
    };
}

[[nodiscard]] Kernel1D<Buffer<luisa::float3>>
make_vector_callable_reuse_shape(bool include_distinct_domain) {
    return [include_distinct_domain](BufferFloat3 output) noexcept {
        output.write(
            0u,
            evaluate_surface_vector_math_vector_svm(
                static_cast<std::uint32_t>(VectorMathOperation::refract),
                immediate_domain,
                input_a,
                input_b,
                input_c,
                input_scale) +
                evaluate_surface_vector_math_vector_svm(
                    static_cast<std::uint32_t>(
                        include_distinct_domain
                            ? VectorMathOperation::add
                            : VectorMathOperation::power),
                    include_distinct_domain
                        ? std::span<const std::uint16_t>{add_only_domain}
                        : std::span<const std::uint16_t>{immediate_domain},
                    input_a,
                    input_b,
                    input_c,
                    input_scale));
    };
}

[[nodiscard]] Kernel1D<Buffer<luisa::float4>> make_runtime_kernel() {
    return [](BufferFloat4 output) noexcept {
        const UInt operation = dispatch_x();
        const auto value = evaluate_surface_vector_math_value_svm(
            operation,
            immediate_domain,
            input_a,
            input_b,
            input_c,
            input_scale);
        const auto vector = evaluate_surface_vector_math_vector_svm(
            operation,
            immediate_domain,
            input_a,
            input_b,
            input_c,
            input_scale);
        output.write(operation, make_float4(vector, value));
    };
}

[[nodiscard]] Kernel1D<Buffer<luisa::float4>>
make_static_reference_kernel() {
    return [](BufferFloat4 output) noexcept {
        const UInt operation = dispatch_x();
        Float value = 0.0f;
        Float3 vector = make_float3(0.0f);
        luisa::compute::detail::SwitchStmtBuilder{operation} % [&] {
            for (auto index = std::uint32_t{0u};
                 index < vector_math_operation_count;
                 ++index) {
                luisa::compute::detail::SwitchCaseStmtBuilder{index} %
                    [&, index] {
                        const auto result =
                            evaluate_surface_vector_math_operation(
                                static_cast<VectorMathOperation>(index),
                                input_a,
                                input_b,
                                input_c,
                                input_scale);
                        value = result.value;
                        vector = result.vector;
                    };
            }
            luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                luisa::compute::dsl::unreachable(
                    "invalid static Vector Math regression operation");
            };
        };
        output.write(operation, make_float4(vector, value));
    };
}

[[nodiscard]] Kernel1D<Buffer<luisa::float3>>
make_cycles_boundary_kernel() {
    return [](BufferFloat3 output) noexcept {
        output.write(
            0u,
            evaluate_surface_vector_math_vector_svm(
                static_cast<std::uint32_t>(VectorMathOperation::power),
                power_round_domain,
                make_float3(0.0f, -2.0f, -2.0f),
                make_float3(-2.0f, 3.0f, 0.5f),
                make_float3(0.0f),
                1.0f));
        output.write(
            1u,
            evaluate_surface_vector_math_vector_svm(
                static_cast<std::uint32_t>(VectorMathOperation::round),
                power_round_domain,
                make_float3(-1.5f, 1.5f, 2.49f),
                make_float3(0.0f),
                make_float3(0.0f),
                1.0f));
    };
}

[[nodiscard]] Kernel1D<Buffer<luisa::float3>>
make_cycles_normalize_zero_kernel() {
    return [](BufferFloat3 output) noexcept {
        output.write(
            0u,
            evaluate_surface_vector_math_vector_svm(
                static_cast<std::uint32_t>(
                    VectorMathOperation::cycles_normalize),
                cycles_normalize_domain,
                make_float3(0.0f),
                make_float3(0.0f),
                make_float3(0.0f),
                1.0f));
    };
}

} // namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};

    const auto reused_value = make_value_callable_reuse_shape(false);
    const auto distinct_value = make_value_callable_reuse_shape(true);
    if (reused_value.function()->function().custom_callables().size() != 1u ||
        distinct_value.function()->function().custom_callables().size() != 2u) {
        std::cerr << "Vector Math value callable hash does not exactly include "
                     "its reachable-mode domain\n";
        return EXIT_FAILURE;
    }
    const auto reused_vector = make_vector_callable_reuse_shape(false);
    const auto distinct_vector = make_vector_callable_reuse_shape(true);
    if (reused_vector.function()->function().custom_callables().size() != 1u ||
        distinct_vector.function()->function().custom_callables().size() !=
            2u) {
        std::cerr << "Vector Math vector callable hash does not exactly include "
                     "its reachable-mode domain\n";
        return EXIT_FAILURE;
    }

    const auto runtime_kernel = make_runtime_kernel();
    // The unchecked Cycles NormalNode needs an explicit non-finite zero-domain
    // projection because fallback's native rsqrt is zero-safe. The four XIR
    // instructions are structural, not material-mode dispatch growth.
    if (!require_bounded_xir(
            "surface Vector Math SVM", runtime_kernel, 4100u)) {
        return EXIT_FAILURE;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    const ShaderOption uncached{.enable_cache = false};
    auto runtime_shader = device.compile(runtime_kernel, uncached);
    auto reference_shader =
        device.compile(make_static_reference_kernel(), uncached);
    auto cycles_boundary_shader =
        device.compile(make_cycles_boundary_kernel(), uncached);
    auto cycles_normalize_zero_shader =
        device.compile(make_cycles_normalize_zero_kernel(), uncached);
    auto actual_buffer =
        device.create_buffer<luisa::float4>(vector_math_operation_count);
    auto expected_buffer =
        device.create_buffer<luisa::float4>(vector_math_operation_count);
    std::vector<luisa::float4> actual(vector_math_operation_count);
    std::vector<luisa::float4> expected(vector_math_operation_count);
    std::array<luisa::float3, 2u> cycles_boundary{};
    auto cycles_boundary_buffer =
        device.create_buffer<luisa::float3>(cycles_boundary.size());
    std::array<luisa::float3, 1u> cycles_normalize_zero{};
    auto cycles_normalize_zero_buffer =
        device.create_buffer<luisa::float3>(cycles_normalize_zero.size());
    stream << runtime_shader(actual_buffer).dispatch(vector_math_operation_count)
           << reference_shader(expected_buffer)
                  .dispatch(vector_math_operation_count)
           << cycles_boundary_shader(cycles_boundary_buffer).dispatch(1u)
           << cycles_normalize_zero_shader(cycles_normalize_zero_buffer)
                  .dispatch(1u)
           << actual_buffer.copy_to(luisa::span{actual})
           << expected_buffer.copy_to(luisa::span{expected})
           << cycles_boundary_buffer.copy_to(luisa::span{cycles_boundary})
           << cycles_normalize_zero_buffer.copy_to(
                  luisa::span{cycles_normalize_zero})
           << synchronize();

    for (auto index = std::uint32_t{0u};
         index < vector_math_operation_count;
         ++index) {
        if (!approximately_equal(actual[index], expected[index], 1.0e-6f)) {
            std::cerr << "surface Vector Math SVM mismatch on " << backend
                      << ", operation=" << index << '\n';
            return EXIT_FAILURE;
        }
    }
    constexpr std::array expected_cycles_boundary{
        luisa::float3{0.0f, -8.0f, 0.0f},
        luisa::float3{-1.0f, 2.0f, 2.0f}};
    for (auto index = std::size_t{0u};
         index < cycles_boundary.size();
         ++index) {
        const auto &actual_boundary = cycles_boundary[index];
        const auto &expected_boundary = expected_cycles_boundary[index];
        if (actual_boundary.x != expected_boundary.x ||
            actual_boundary.y != expected_boundary.y ||
            actual_boundary.z != expected_boundary.z) {
            std::cerr << "Cycles Vector Math boundary mismatch on " << backend
                      << ", case=" << index << '\n';
            return EXIT_FAILURE;
        }
    }
    const auto &zero = cycles_normalize_zero.front();
    if (!std::isnan(zero.x) || !std::isnan(zero.y) ||
        !std::isnan(zero.z)) {
        std::cerr << "Cycles unchecked normalize lost its zero-vector "
                     "non-finite boundary on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
