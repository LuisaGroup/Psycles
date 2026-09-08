#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/svm/fractal_noise.h"
#include "inputs.h"
#include <cstdio>
#include <cstdlib>
#include <string_view>

void check(hipError_t status) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s\n", hipGetErrorString(status));
        std::exit(1);
    }
}

__global__ __launch_bounds__(256, 1)
void noise_probe(const float4 *coordinates, const float4 *, float *output, unsigned count) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const auto p = coordinates[i];
    output[i] = snoise_3d(make_float3(p.x, p.y, p.z));
}

__global__ __launch_bounds__(256, 1)
void fbm_probe(const float4 *coordinates, const float4 *parameters, float *output, unsigned count) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const auto p = coordinates[i];
    const auto q = parameters[i];
    output[i] = noise_fbm(make_float3(p.x, p.y, p.z), q.x, q.y, q.z, q.w != 0.0f);
}

int main(int argc, char **argv) {
    if (argc != 2) { return 2; }
    std::array<float4, probe_count> coordinates, parameters;
    const auto inputs = probe_inputs();
    for (unsigned i = 0; i < probe_count; ++i) {
        const auto &p = inputs[i].coordinate;
        const auto &q = inputs[i].parameters;
        coordinates[i] = make_float4(p[0], p[1], p[2], p[3]);
        parameters[i] = make_float4(q[0], q[1], q[2], q[3]);
    }
    float4 *device_coordinates{}, *device_parameters{};
    float *device_output{};
    check(hipMalloc(&device_coordinates, sizeof(coordinates)));
    check(hipMalloc(&device_parameters, sizeof(parameters)));
    check(hipMalloc(&device_output, probe_count * sizeof(float)));
    check(hipMemcpy(device_coordinates, coordinates.data(), sizeof(coordinates), hipMemcpyHostToDevice));
    check(hipMemcpy(device_parameters, parameters.data(), sizeof(parameters), hipMemcpyHostToDevice));
    if (std::string_view{argv[1]} == "noise") {
        noise_probe<<<1, 256>>>(device_coordinates, device_parameters, device_output, probe_count);
    } else if (std::string_view{argv[1]} == "fbm") {
        fbm_probe<<<1, 256>>>(device_coordinates, device_parameters, device_output, probe_count);
    } else { return 2; }
    check(hipGetLastError());
    std::array<float, probe_count> output;
    check(hipMemcpy(output.data(), device_output, sizeof(output), hipMemcpyDeviceToHost));
    for (unsigned i = 0; i < probe_count; ++i) {
        std::printf("result %u %.9g\n", i, output[i]);
    }
    check(hipFree(device_output));
    check(hipFree(device_parameters));
    check(hipFree(device_coordinates));
}
