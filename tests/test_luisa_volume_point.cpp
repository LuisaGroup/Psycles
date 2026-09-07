#include "cycles_svm_volume_stack_fixture.h"
#include "cycles_svm_volume_scene_fixture.h"

#include <cmath>
#include <fstream>
#include <iostream>

namespace {
using namespace luisa::compute;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
namespace f = psycles::test_support::volume_stack_fixture;
namespace abi = psycles::compiler::cycles_svm;

bool run(const char *program, const char *backend) {
    Context context{program};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto scene = std::make_shared<LuisaSceneData>();
    scene->device = Device{device.impl_shared()};
    abi::CompiledShaderTable table;
    table.table.valid = true;
    table.table.words = f::words();
    table.table.shader_count = f::shader_count;
    table.table.peak_stack_usage = 4u;
    table.kernel_features = cycles_svm::kernel_feature_volume;
    table.kernel_shaders.resize(f::shader_count);
    for (unsigned s = 0; s < f::shader_count; ++s) { table.kernel_shaders[s].flags = f::shader_flags(s); }
    for (auto node : {abi::NODE_SHADER_JUMP, abi::NODE_END, abi::NODE_CLOSURE_SET_WEIGHT,
                     abi::NODE_CLOSURE_VOLUME, abi::NODE_VOLUME_COEFFICIENTS, abi::NODE_TEX_COORD,
                     abi::NODE_GEOMETRY, abi::NODE_LIGHT_PATH, abi::NODE_EMISSION_WEIGHT,
                     abi::NODE_CLOSURE_EMISSION}) { table.table.node_types_used[node] = true; }
    std::vector<abi::KernelObject> objects(4u);
    std::vector<unsigned> flags(4u);
    for (unsigned i = 0; i < 4u; ++i) {
        objects[i].volume_density = f::densities[i];
        objects[i].visibility = i == 2u ? abi::PATH_RAY_VISIBILITY_CAMERA : abi::PATH_RAY_VISIBILITY_ALL;
        objects[i].tfm = {{2, 0, 0, 0.5f}, {0, -4, 0, -0.25f}, {0, 0, 0.5f, 1}};
        objects[i].itfm = {{0.5f, 0, 0, -0.25f}, {0, -0.25f, 0, -0.0625f}, {0, 0, 2, -2}};
        flags[i] = i % 2u ? abi::SD_OBJECT_NEGATIVE_SCALE : 0u;
    }
    psycles::test_support::initialize_volume_fixture(scene, stream, std::move(table), objects, flags);
    std::vector<luisa::uint2> entries, states;
    for (unsigned row = 0; row < f::stacks.size(); ++row) {
        for (unsigned e = 0; e < 3u; ++e) {
            entries.emplace_back(f::object(row, e), unsigned(f::stacks[row][e]));
        }
    }
    for (unsigned mode = 0; mode < f::state_count; ++mode) {
        states.emplace_back(f::visibility(mode), f::flag(mode));
    }
    auto entry_buffer = psycles::test_support::upload_volume_fixture(device, stream, entries);
    auto state_buffer = psycles::test_support::upload_volume_fixture(device, stream, states);
    constexpr unsigned stride = 9u + 1u + 8u * 8u;
    auto output = device.create_buffer<float>(f::count * stride);
    std::ifstream oracle{PSYCLES_VOLUME_STACK_ORACLE};
    for (unsigned capacity : f::capacities) {
        scene->cycles_svm->compilation.max_closures = capacity;
        Kernel1D kernel = [scene, stride](BufferUInt2 entries, BufferUInt2 states, BufferFloat out) {
            const auto i = dispatch_x(), mode = i % f::state_count, row = i / f::state_count;
            const auto s = states.read(mode);
            const cycles_svm::PathState state{s.x, s.y, 3u};
            VolumeStack stack{4u};
            for (unsigned e = 0; e < 3u; ++e) {
                const auto pair = entries.read(row * 3u + e);
                const VolumeStackEntry entry{pair.x, pair.y, ~0u, 0u, ~0u, volume_sample_distance, pair.y != ~0u};
                stack.initialize_background(entry, entry.valid);
            }
            Var<RenderKernelParameters> parameters;
            parameters.camera_transform = parameters.camera_inverse_transform = make_float4x4(1.0f);
            const auto evaluate = [&](bool shadow) {
                const PathCyclesSvmVolumeShader shader{
                    scene, parameters, make_float3(1.0f, -2.0f, 3.0f),
                    make_float3(0.0f, 0.0f, 1.0f), 0.25f, 0.375f, ~0u, state, 0u, shadow};
                // A coefficient-only extrema query must not consume the next
                // stack evaluation's closure allocation budget.
                static_cast<void>(shader.evaluate_entry(stack.entry(0u), make_float3(7.0f)));
                VolumePhaseSet phases{8u};
                const auto value = shader.evaluate(stack, make_float3(1.0f, -2.0f, 3.25f), &phases);
                const auto put = [&](unsigned j, Float v) { out.write(i * stride + j, v); };
                const auto put3 = [&](unsigned j, Float3 v) { put(j, v.x); put(j + 1u, v.y); put(j + 2u, v.z); };
                put3(0, value.sigma_t); put3(3, value.sigma_s); put3(6, value.emission);
                put(9, phases.count().cast<float>());
                for (unsigned p = 0; p < 8u; ++p) {
                    const auto phase = phases.entry(p);
                    put(10u + p * 8u, phase.type.cast<float>());
                    put3(11u + p * 8u, phase.weight);
                    put(14u + p * 8u, phase.sample_weight);
                    put3(15u + p * 8u, phase.parameters);
                }
            };
            $if(mode == 2u) { evaluate(true); } $else { evaluate(false); };
        };
        auto shader = device.compile(kernel, ShaderOption{.enable_fast_math = true});
        std::vector<float> actual(f::count * stride);
        stream << shader(entry_buffer, state_buffer, output).dispatch(f::count)
               << output.copy_to(actual.data()) << synchronize();
        for (unsigned row = 0; row < f::count; ++row) {
            char tag{}; unsigned c{}, index{}; f::Output expected;
            oracle >> tag >> c >> index;
            for (auto &v : expected.meta) { oracle >> v; }
            for (auto &v : expected.setup) { oracle >> v; }
            for (auto &v : expected.coefficients) { oracle >> v; }
            for (auto &v : expected.closures) { for (auto &x : v) { oracle >> x; } }
            for (auto &v : expected.phases) { for (auto &x : v) { oracle >> x; } }
            if (!oracle || tag != 'R' || c != capacity || index != row) { return false; }
            for (unsigned j = 0; j < stride; ++j) {
                float want = j < 9u ? expected.coefficients[j] :
                             j == 9u ? float(expected.meta[7]) : expected.phases[(j - 10u) / 8u][(j - 10u) % 8u];
                const auto got = actual[row * stride + j];
                if (!std::isfinite(got) || std::abs(got - want) > 1.0e-6f + 1.0e-4f * std::abs(want)) {
                    std::cerr << "native volume consumer capacity=" << capacity << " row=" << row
                              << " lane=" << j << " got=" << got << " expected=" << want << '\n';
                    return false;
                }
            }
        }
    }
    return true;
}
} // namespace

int main(int argc, char **argv) {
    return run(argv[0], argc > 1 ? argv[1] : "fallback") ? 0 : 1;
}
