#include "cycles_svm_internal.h"
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/shader.h>
#include <luisa/dsl/sugar.h>
#include "node_inputs.h"
#include <cstdio>

int main(int argc, char **) {
    using namespace luisa::compute;
    namespace abi = psycles::compiler::cycles_svm;
    namespace svm = psycles::luisa_backend::cycles_svm::detail;
    const unsigned work_count = argc > 1 ? 1048576u : probe_count;
    const unsigned repeats = argc > 1 ? 18u : 1u;
    Context context{"/home/mike/Projects/Psycles-surface-svm/build/bin/psycles_render_blender_scene"};
    auto device = context.create_device("hip");
    auto stream = device.create_stream();
    const auto nodes = node_inputs<abi::SVMNodeTexNoise>();
    static_assert(sizeof(abi::SVMNodeTexNoise) == 48u);
    const auto words = std::bit_cast<std::array<std::uint32_t, probe_count * 12u>>(nodes);
    const auto inputs = probe_inputs();
    std::array<luisa::float4, probe_count> coordinates;
    for (unsigned i = 0; i < probe_count; ++i) {
        const auto &p = inputs[i].coordinate;
        coordinates[i] = {p[0], p[1], p[2], inputs[i].parameters[1]};
    }
    Kernel1D kernel = [&](BufferFloat4 coords, BufferUInt words, BufferFloat4 output,
                          BufferUInt end_offsets) {
        set_block_size(256u);
        const auto i = dispatch_x();
        const auto input_index = i % probe_count;
        const auto p = coords.read(input_index);
        svm::Stack stack{33u};
        stack[0u] = p.x; stack[1u] = p.y; stack[2u] = p.z; stack[3u] = p.w;
        UInt pc = input_index * 12u;
        svm::Cursor cursor{words, pc};
        svm::node_tex_noise(cursor, stack);
        const auto packed = words.read(input_index * 12u + 11u);
        const auto value_offset = (packed >> 8u) & 255u;
        const auto color_offset = (packed >> 16u) & 255u;
        Float3 color = make_float3(0.0f);
        $if(color_offset != 255u) { color = svm::stack_load_float3(stack, color_offset); };
        output.write(i, make_float4(color, svm::stack_load_float(stack, value_offset)));
        end_offsets.write(i, pc - input_index * 12u);
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false, .enable_fast_math = true});
    auto device_coordinates = device.create_buffer<luisa::float4>(probe_count);
    auto device_nodes = device.create_buffer<std::uint32_t>(probe_count * 12u);
    auto device_output = device.create_buffer<luisa::float4>(work_count);
    auto device_offsets = device.create_buffer<std::uint32_t>(work_count);
    std::array<luisa::float4, probe_count> output;
    std::array<std::uint32_t, probe_count> offsets;
    stream << device_coordinates.copy_from(luisa::span{coordinates})
           << device_nodes.copy_from(luisa::span{words});
    for (unsigned repeat = 0; repeat < repeats; ++repeat) {
        stream << shader(device_coordinates, device_nodes, device_output, device_offsets).dispatch(work_count);
    }
    stream << device_output.view(0u, probe_count).copy_to(luisa::span{output})
           << device_offsets.view(0u, probe_count).copy_to(luisa::span{offsets}) << synchronize();
    for (unsigned i = 0; i < probe_count; ++i) {
        if (offsets[i] != 12u) { return 1; }
        const auto v = output[i];
        std::printf("result %u %.9g %.9g %.9g %.9g\n", i, v.x, v.y, v.z, v.w);
    }
}
