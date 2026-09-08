#include <psycles/luisa/cycles_noise.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/shader.h>
#include "inputs.h"
#include <cstdio>
#include <string_view>

int main(int argc, char **argv) {
    using namespace luisa::compute;
    namespace noise = psycles::luisa_backend::cycles_noise;
    if (argc != 2) { return 2; }
    const bool is_fbm = std::string_view{argv[1]} == "fbm";
    if (!is_fbm && std::string_view{argv[1]} != "noise") { return 2; }
    Context context{"/home/mike/Projects/Psycles-surface-svm/build/bin/psycles_render_blender_scene"};
    auto device = context.create_device("hip");
    auto stream = device.create_stream();
    std::array<luisa::float4, probe_count> coordinates, parameters;
    const auto inputs = probe_inputs();
    for (unsigned i = 0; i < probe_count; ++i) {
        const auto &p = inputs[i].coordinate;
        const auto &q = inputs[i].parameters;
        coordinates[i] = {p[0], p[1], p[2], p[3]};
        parameters[i] = {q[0], q[1], q[2], q[3]};
    }
    Kernel1D kernel = [&](BufferFloat4 coords, BufferFloat4 params, BufferFloat output) {
        set_block_size(256u);
        const auto i = dispatch_x();
        const auto p = coords.read(i).xyz();
        const auto q = params.read(i);
        output.write(i, is_fbm ? noise::fbm(p, q.x, q.y, q.z, q.w != 0.0f)
                               : noise::signed_noise(p));
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false, .enable_fast_math = true});
    auto device_coordinates = device.create_buffer<luisa::float4>(probe_count);
    auto device_parameters = device.create_buffer<luisa::float4>(probe_count);
    auto device_output = device.create_buffer<float>(probe_count);
    std::array<float, probe_count> output;
    stream << device_coordinates.copy_from(luisa::span{coordinates})
           << device_parameters.copy_from(luisa::span{parameters})
           << shader(device_coordinates, device_parameters, device_output).dispatch(probe_count)
           << device_output.copy_to(luisa::span{output}) << synchronize();
    for (unsigned i = 0; i < probe_count; ++i) {
        std::printf("result %u %.9g\n", i, output[i]);
    }
}
