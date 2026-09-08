#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/svm/noisetex.h"
#include "node_inputs.h"
#include <cstdio>
#include <cstdlib>

void check(hipError_t status) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s\n", hipGetErrorString(status));
        std::exit(1);
    }
}

__global__ __launch_bounds__(256, 1)
void node_probe(const float4 *coordinates, const SVMNodeTexNoise *nodes,
                float4 *output, unsigned *end_offsets, unsigned count) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const auto input_index = i % probe_count;
    const auto p = coordinates[input_index];
    float stack[33];
    stack[0] = p.x; stack[1] = p.y; stack[2] = p.z; stack[3] = p.w;
    const auto &node = nodes[input_index];
    svm_node_tex_noise(stack, node);
    const auto color = node.color_offset == SVM_STACK_INVALID ? zero_float3()
                                                             : stack_load_float3(stack, node.color_offset);
    output[i] = make_float4(color.x, color.y, color.z, stack_load_float(stack, node.value_offset));
    end_offsets[i] = sizeof(SVMNodeTexNoise) / sizeof(unsigned);
}

int main(int argc, char **) {
    const unsigned work_count = argc > 1 ? 1048576u : probe_count;
    const unsigned repeats = argc > 1 ? 18u : 1u;
    const auto nodes = node_inputs<SVMNodeTexNoise>();
    static_assert(sizeof(SVMNodeTexNoise) == 48u);
    const auto inputs = probe_inputs();
    std::array<float4, probe_count> coordinates;
    for (unsigned i = 0; i < probe_count; ++i) {
        const auto &p = inputs[i].coordinate;
        coordinates[i] = make_float4(p[0], p[1], p[2], inputs[i].parameters[1]);
    }
    float4 *device_coordinates{}, *device_output{};
    SVMNodeTexNoise *device_nodes{};
    unsigned *device_offsets{};
    check(hipMalloc(&device_coordinates, sizeof(coordinates)));
    check(hipMalloc(&device_nodes, sizeof(nodes)));
    check(hipMalloc(&device_output, work_count * sizeof(float4)));
    check(hipMalloc(&device_offsets, work_count * sizeof(unsigned)));
    check(hipMemcpy(device_coordinates, coordinates.data(), sizeof(coordinates), hipMemcpyHostToDevice));
    check(hipMemcpy(device_nodes, nodes.data(), sizeof(nodes), hipMemcpyHostToDevice));
    for (unsigned repeat = 0; repeat < repeats; ++repeat) {
        node_probe<<<(work_count + 255u) / 256u, 256>>>(device_coordinates, device_nodes, device_output, device_offsets, work_count);
        check(hipGetLastError());
    }
    std::array<float4, probe_count> output;
    check(hipMemcpy(output.data(), device_output, sizeof(output), hipMemcpyDeviceToHost));
    for (unsigned i = 0; i < probe_count; ++i) {
        const auto v = output[i];
        std::printf("result %u %.9g %.9g %.9g %.9g\n", i, v.x, v.y, v.z, v.w);
    }
    check(hipFree(device_output));
    check(hipFree(device_offsets));
    check(hipFree(device_nodes));
    check(hipFree(device_coordinates));
}
