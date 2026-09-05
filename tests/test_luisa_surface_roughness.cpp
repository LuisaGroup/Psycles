#include "cycles_surface_roughness_cases.h"
#include "luisa_surface_test_support.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend;
using psycles::test_support::roughness_cases;

// Captured from tools/cycles_surface_roughness_oracle.hip, calling Cycles
// 5.2.1 surface_shader_average_roughness() on HIP with these exact inputs.
// No local CPU shading implementation supplies the expected values.
constexpr std::array<float, roughness_cases.size()> expected{
    1.0f, 1.0f, 0.400000006f, 0.400000006f, 0.400000036f, 1.0f,
    0.574999988f, 0.0f, 0.449999988f, 1.0f,
    1.0f, 1.0f, 1.0f, 0.400000006f, 0.400000006f, 0.400000006f,
    0.400000006f, 1.0f, 1.0f, 1.0f, 0.400000006f, 0.400000006f,
    0.400000006f, 1.0f, 0.400000006f, 0.400000006f, 0.400000006f,
    1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f};

} // namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
    const auto aov = make_surface_closure_aov_callable();
    Kernel1D test = [aov](BufferFloat4 input, BufferFloat output) noexcept {
        const auto point = psycles::test_support::make_surface_point();
        SurfaceAovVisitor visitor{point, 2u, aov};
        visitor.begin(point.shading_normal);
        for (auto i = 0u; i < 2u; ++i) {
            const auto value = input.read(dispatch_x() * 4u + i * 2u);
            const auto meta = input.read(dispatch_x() * 4u + i * 2u + 1u);
            auto closure = SurfaceClosureRecord::zero();
            // Runtime inputs keep the classification from being specialized
            // to a host-known type. Allocation flags describe a retained
            // prefix, including closures whose weight became zero in setup.
            closure.closure_type = cast<luisa::uint>(meta.x);
            closure.allocation_weight = meta.y;
            closure.setup_valid = meta.y != 0.0f;
            closure.weight = value.xyz();
            closure.roughness = value.w;
            visitor.add(closure);
        }
        visitor.finish();
        output.write(dispatch_x(), visitor.result().roughness.x);
    };

    std::vector<luisa::float4> inputs;
    for (const auto &scenario : roughness_cases) {
        for (auto i = 0u; i < 2u; ++i) {
            const auto &closure = scenario.closures[i];
            inputs.emplace_back(closure.weight[0], closure.weight[1],
                                closure.weight[2], closure.roughness);
            inputs.emplace_back(static_cast<float>(closure.type),
                                i < scenario.count ? 1.0f : 0.0f, 0.0f, 0.0f);
        }
    }
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto input = device.create_buffer<luisa::float4>(inputs.size());
    auto output = device.create_buffer<float>(expected.size());
    auto shader = device.compile(test);
    std::array<float, expected.size()> actual{};
    stream << input.copy_from(inputs.data())
           << shader(input, output).dispatch(expected.size())
           << output.copy_to(actual.data()) << synchronize();
    auto failed = false;
    for (auto i = 0u; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 2.0e-6f) {
            std::cerr << "roughness case " << i << " on " << backend
                      << ": Cycles " << expected[i] << ", actual " << actual[i] << '\n';
            failed = true;
        }
    }
    if (failed) { return EXIT_FAILURE; }
    std::cout << "Cycles roughness oracle: " << expected.size()
              << " cases passed on " << backend << '\n';
    return EXIT_SUCCESS;
}
