#include "path_tracer_surface_closure_point.h"

#include "luisa_surface_test_support.h"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
using psycles::test_support::approximately_equal;
using psycles::test_support::compile_named_kernel;
using psycles::test_support::make_surface_point;

constexpr std::uint32_t case_count = 8u;
constexpr std::uint32_t records_per_case = 3u;

static_assert(sizeof(SurfaceClosurePointCall) == 48u);
static_assert(offsetof(
                  SurfaceClosurePointCall,
                  shading_yz_and_incoming_xy) == 16u);
static_assert(offsetof(SurfaceClosurePointCall, incoming_z) == 32u);
static_assert(offsetof(
                  SurfaceClosurePointCall,
                  ray_visibility) == 36u);
static_assert(offsetof(SurfaceClosurePointCall, flags) == 40u);

using SurfaceClosurePointRoundTripCallable =
    Callable<SurfaceClosurePointCall(SurfaceClosurePointCall)>;

[[nodiscard]] luisa::float4 expected_record(
    std::uint32_t case_index,
    std::uint32_t record) noexcept {
    const auto value = static_cast<float>(case_index);
    if (record == 0u) {
        return {
            0.11f + value,
            -0.23f - 2.0f * value,
            0.37f + 3.0f * value,
            (case_index & 1u) == 0u ? 1.0f : 0.0f};
    }
    if (record == 1u) {
        return {
            -0.41f - value,
            0.53f + 2.0f * value,
            -0.67f - 3.0f * value,
            (case_index & 2u) != 0u ? 1.0f : 0.0f};
    }
    return {
        0.71f + 4.0f * value,
        -0.83f - 5.0f * value,
        0.97f + 6.0f * value,
        static_cast<float>(3u + 17u * case_index)};
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};

    SurfaceClosurePointRoundTripCallable round_trip =
        [](Var<SurfaceClosurePointCall> packed) noexcept {
            return pack_surface_closure_point(
                unpack_surface_closure_point(packed));
        };
    round_trip.set_name("surface_closure_point_round_trip");

    Kernel1D test = [round_trip](BufferFloat4 output) noexcept {
        const auto case_index = dispatch_x();
        const auto value = cast<float>(case_index);
        auto point = make_surface_point();
        point.geometric_normal = make_float3(
            0.11f + value,
            -0.23f - 2.0f * value,
            0.37f + 3.0f * value);
        point.shading_normal = make_float3(
            -0.41f - value,
            0.53f + 2.0f * value,
            -0.67f - 3.0f * value);
        point.incoming = make_float3(
            0.71f + 4.0f * value,
            -0.83f - 5.0f * value,
            0.97f + 6.0f * value);
        point.ray_visibility = 3u + 17u * case_index;
        point.use_bump_map_correction = (case_index & 1u) == 0u;
        point.back_facing = (case_index & 2u) != 0u;

        // These sentinels belong to material population and deliberately do
        // not cross the post-population closure boundary.
        point.position = make_float3(101.0f + value);
        point.uv = make_float2(211.0f + value, 307.0f - value);
        point.parameter_block = 401u + case_index;

        const SurfaceClosurePoint projected{point};
        const auto unpacked = unpack_surface_closure_point(
            round_trip(pack_surface_closure_point(projected)));
        const auto base = case_index * records_per_case;
        output->write(
            base,
            make_float4(
                unpacked.geometric_normal,
                select(0.0f,
                       1.0f,
                       unpacked.use_bump_map_correction)));
        output->write(
            base + 1u,
            make_float4(
                unpacked.shading_normal,
                select(0.0f, 1.0f, unpacked.back_facing)));
        output->write(
            base + 2u,
            make_float4(
                unpacked.incoming,
                cast<float>(unpacked.ray_visibility)));
    };

    if (test.function()->function().custom_callables().size() != 1u) {
        std::cerr << "SurfaceClosurePoint callable reuse regression: "
                     "expected one definition, got "
                  << test.function()->function().custom_callables().size()
                  << '\n';
        return EXIT_FAILURE;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto output = device.create_buffer<luisa::float4>(
        case_count * records_per_case);
    auto shader = compile_named_kernel(
        device, "surface closure point ABI", test);
    std::array<luisa::float4, case_count * records_per_case> actual{};
    stream << shader(output).dispatch(case_count)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    for (auto case_index = 0u; case_index < case_count; ++case_index) {
        for (auto record = 0u; record < records_per_case; ++record) {
            const auto observed =
                actual[case_index * records_per_case + record];
            const auto expected = expected_record(case_index, record);
            if (!approximately_equal(observed, expected, 1.0e-6f)) {
                std::cerr << "SurfaceClosurePoint ABI mismatch on "
                          << backend << ", case " << case_index
                          << ", record " << record << ": got {"
                          << observed.x << ", " << observed.y << ", "
                          << observed.z << ", " << observed.w
                          << "}, expected {" << expected.x << ", "
                          << expected.y << ", " << expected.z << ", "
                          << expected.w << "}\n";
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}
