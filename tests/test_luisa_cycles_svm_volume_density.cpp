#include "cycles_svm_volume_density_fixture.h"
#include "cycles_svm_volume_scene_fixture.h"
#include <cmath>
#include <fstream>
#include <iostream>

namespace {
using namespace luisa::compute;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
namespace f = psycles::test_support::volume_density_fixture;
namespace abi = psycles::compiler::cycles_svm;

bool run(const char *program, const char *backend) {
    Context context{program};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto scene = std::make_shared<LuisaSceneData>();
    scene->device = Device{device.impl_shared()};
    abi::CompiledShaderTable table;
    table.table.valid = true; table.table.words = f::words();
    table.table.shader_count = f::shader_count; table.table.peak_stack_usage = 4u;
    table.kernel_features = cycles_svm::kernel_feature_volume;
    table.kernel_shaders.resize(f::shader_count);
    for (unsigned s = 0; s < f::shader_count; ++s) { table.kernel_shaders[s].flags = f::shader_flags(s); }
    for (auto node : {abi::NODE_SHADER_JUMP, abi::NODE_END, abi::NODE_TEX_COORD,
                     abi::NODE_GEOMETRY, abi::NODE_LIGHT_PATH, abi::NODE_EMISSION_WEIGHT,
                     abi::NODE_CLOSURE_EMISSION}) { table.table.node_types_used[node] = true; }
    std::vector<abi::KernelObject> objects(f::inputs.size());
    std::vector<unsigned> flags(f::inputs.size());
    std::vector<luisa::uint2> inputs;
    for (unsigned i = 0; i < objects.size(); ++i) {
        objects[i].volume_density = 2.0f; objects[i].visibility = abi::PATH_RAY_VISIBILITY_ALL;
        objects[i].tfm = {{2, 0, 0, 0.5f}, {0, -4, 0, -0.25f}, {0, 0, 0.5f, 1}};
        objects[i].itfm = {{0.5f, 0, 0, -0.25f}, {0, -0.25f, 0, -0.0625f}, {0, 0, 2, -2}};
        flags[i] = f::inputs[i].flags;
        inputs.emplace_back(f::inputs[i].shader, f::inputs[i].cell);
    }
    psycles::test_support::initialize_volume_fixture(scene, stream, std::move(table), objects, flags);
    psycles::test_support::volume_generated_fixture(scene, stream,
        {{{0.2f, 0, 0, 0.2f}, {0, 0.25f, 0, -0.5f}, {0, 0, 0.2f, -0.6f}}});
    auto input = psycles::test_support::upload_volume_fixture(device, stream, inputs);
    auto output = device.create_buffer<luisa::float2>(f::inputs.size());
    Kernel1D kernel = [scene](BufferUInt2 inputs, BufferVar<luisa::float2> out) {
        const auto i = dispatch_x();
        const auto v = inputs.read(i);
        const VolumeStackEntry entry{i, v.x, ~0u, 0u, ~0u, volume_sample_distance, true};
        Var<RenderKernelParameters> parameters;
        parameters.camera_transform = parameters.camera_inverse_transform = make_float4x4(1.0f);
        const VolumeMajorantGrid grid{make_float3(-1, 2, 3), make_float3(4, 6, 8),
                                     make_float4x4(1.0f), 128u};
        const auto value = evaluate_cycles_svm_volume_density_cell(scene, parameters, entry, grid, v.y);
        out.write(i, make_float2(value.minimum, value.maximum));
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false, .enable_fast_math = true});
    std::vector<luisa::float2> actual(f::inputs.size());
    stream << shader(input, output).dispatch(f::inputs.size()) << output.copy_to(actual.data()) << synchronize();
    std::ifstream oracle{PSYCLES_VOLUME_DENSITY_ORACLE};
    for (unsigned i = 0; i < actual.size(); ++i) {
        unsigned row{}; float lo{}, hi{}; oracle >> row >> lo >> hi;
        const auto equal = [](float a, float b) {
            return std::isfinite(a) && std::abs(a - b) <= 2.0e-6f + 2.0e-5f * std::abs(b);
        };
        if (!oracle || row != i || !equal(actual[i].x, lo) || !equal(actual[i].y, hi)) {
            std::cerr << "Cycles density bake row " << i << ": got " << actual[i].x << ','
                      << actual[i].y << " expected " << lo << ',' << hi << '\n';
            return false;
        }
    }
    return true;
}
} // namespace
int main(int argc, char **argv) { return run(argv[0], argc > 1 ? argv[1] : "fallback") ? 0 : 1; }
