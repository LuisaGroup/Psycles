#include "luisa_shader_shape_test_support.h"
#include "luisa_surface_test_support.h"
#include "surface_math.h"

#include <array>
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
    std::array<std::uint16_t, math_operation_count> result{};
    for (auto index = std::uint32_t{0u}; index < math_operation_count;
         ++index) {
        result[index] = static_cast<std::uint16_t>(index);
    }
    return result;
}

inline constexpr auto immediate_domain = make_immediate_domain();
inline constexpr std::array<std::uint16_t, 1u> add_only_domain{
    static_cast<std::uint16_t>(MathOperation::add)};

[[nodiscard]] Kernel1D<Buffer<float>> make_callable_reuse_shape(
    bool include_distinct_domain) {
    return [include_distinct_domain](BufferFloat output) noexcept {
        output.write(
            0u,
            evaluate_surface_math_svm(
                static_cast<std::uint32_t>(MathOperation::power),
                immediate_domain,
                0.75f,
                1.25f,
                0.2f) +
                evaluate_surface_math_svm(
                    static_cast<std::uint32_t>(MathOperation::add),
                    include_distinct_domain
                        ? std::span<const std::uint16_t>{add_only_domain}
                        : std::span<const std::uint16_t>{immediate_domain},
                    0.75f,
                    1.25f,
                    0.2f));
    };
}

[[nodiscard]] Kernel1D<Buffer<float>> make_runtime_kernel() {
    return [](BufferFloat output) noexcept {
        const UInt operation = dispatch_x();
        output.write(operation,
                     evaluate_surface_math_svm(
                         operation,
                         immediate_domain,
                         0.75f,
                         1.25f,
                         0.2f));
    };
}

[[nodiscard]] Kernel1D<Buffer<float>> make_static_reference_kernel() {
    return [](BufferFloat output) noexcept {
        const UInt operation = dispatch_x();
        Float result = 0.0f;
        luisa::compute::detail::SwitchStmtBuilder{operation} % [&] {
            for (auto index = std::uint32_t{0u};
                 index < math_operation_count;
                 ++index) {
                luisa::compute::detail::SwitchCaseStmtBuilder{index} %
                    [&, index] {
                        result = evaluate_surface_math_operation(
                            static_cast<MathOperation>(index),
                            0.75f,
                            1.25f,
                            0.2f);
                    };
            }
            luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                luisa::compute::dsl::unreachable(
                    "invalid static Math regression operation");
            };
        };
        output.write(operation, result);
    };
}

} // namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};

    const auto reused_callable = make_callable_reuse_shape(false);
    if (reused_callable.function()->function().custom_callables().size() !=
        1u) {
        std::cerr << "independent Math SVM handlers were not deduplicated by "
                     "complete AST hash\n";
        return EXIT_FAILURE;
    }
    const auto distinct_callable = make_callable_reuse_shape(true);
    if (distinct_callable.function()->function().custom_callables().size() !=
        2u) {
        std::cerr << "Math SVM callable hashing omitted its reachable-mode "
                     "domain\n";
        return EXIT_FAILURE;
    }

    const auto runtime_kernel = make_runtime_kernel();
    if (!require_bounded_xir("surface Math SVM", runtime_kernel, 4096u)) {
        return EXIT_FAILURE;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    const ShaderOption uncached{.enable_cache = false};
    auto runtime_shader = device.compile(runtime_kernel, uncached);
    auto reference_shader =
        device.compile(make_static_reference_kernel(), uncached);
    auto actual_buffer = device.create_buffer<float>(math_operation_count);
    auto expected_buffer = device.create_buffer<float>(math_operation_count);
    std::vector<float> actual(math_operation_count);
    std::vector<float> expected(math_operation_count);
    stream << runtime_shader(actual_buffer).dispatch(math_operation_count)
           << reference_shader(expected_buffer).dispatch(math_operation_count)
           << actual_buffer.copy_to(luisa::span{actual})
           << expected_buffer.copy_to(luisa::span{expected}) << synchronize();

    for (auto index = std::uint32_t{0u}; index < math_operation_count;
         ++index) {
        if (!approximately_equal(actual[index], expected[index], 1.0e-6f)) {
            std::cerr << "surface Math SVM mismatch on " << backend
                      << ", operation=" << index << ", actual="
                      << actual[index] << ", expected=" << expected[index]
                      << '\n';
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
