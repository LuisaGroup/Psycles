// Executes original Cycles kernel_volume_density_evaluate, not a host evaluator.
#include "../tests/cycles_svm_volume_density_fixture.h"
// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/geom/primitive.h"
#include "kernel/geom/shader_data.h"
#include "kernel/integrator/state_util.h"
#include "kernel/integrator/volume_stack.h"
#include "kernel/integrator/surface_shader.h"
#include "kernel/bake/bake.h"
// clang-format on
#include <cstdio>
#include <cstdlib>

namespace f = psycles::test_support::volume_density_fixture;
void check(hipError_t e) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(e)); std::exit(1); }
}
template<typename T> T *upload(const std::vector<T> &v) {
    T *p{}; check(hipMalloc(&p, v.size() * sizeof(T)));
    check(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice)); return p;
}
__global__ void probe(const KernelShaderEvalInput *input, float *output, uint *miss, int cell) {
    kernel_volume_density_evaluate(nullptr, input, output, miss, cell);
}
int main() {
    KernelParamsHIP params{};
    params.svm_nodes = upload(f::words());
    std::vector<KernelShader> shaders(f::shader_count);
    for (unsigned s = 0; s < shaders.size(); ++s) { shaders[s].flags = f::shader_flags(s); }
    params.shaders = upload(shaders);
    std::vector<KernelObject> objects(f::inputs.size());
    std::vector<unsigned> flags(f::inputs.size());
    for (unsigned i = 0; i < objects.size(); ++i) {
        auto &o = objects[i]; flags[i] = f::inputs[i].flags;
        o.volume_density = 2.0f; o.visibility = PATH_RAY_VISIBILITY_ALL;
        o.tfm = {make_float4(2, 0, 0, 0.5f), make_float4(0, -4, 0, -0.25f), make_float4(0, 0, 0.5f, 1)};
        o.itfm = {make_float4(0.5f, 0, 0, -0.25f), make_float4(0, -0.25f, 0, -0.0625f), make_float4(0, 0, 2, -2)};
    }
    params.objects = upload(objects); params.object_flag = upload(flags);
    params.attributes_map = upload(std::vector<AttributeMap>{
        {ATTR_STD_GENERATED_TRANSFORM, 0, ATTR_ELEMENT_MESH, NODE_ATTR_MATRIX, 0}, {}});
    params.attributes_float4 = upload(std::vector<float4>{
        make_float4(0.2f, 0, 0, 0.2f), make_float4(0, 0.25f, 0, -0.5f), make_float4(0, 0, 0.2f, -0.6f)});
    params.data.kernel_features = KERNEL_FEATURE_VOLUME;
    params.data.background.object_index = OBJECT_NONE;
    check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
    constexpr unsigned cells = 128u * 128u * 128u;
    KernelShaderEvalInput *input{}; float *output{}; uint *miss{};
    check(hipMalloc(&input, cells * 2u * sizeof(*input)));
    check(hipMalloc(&output, cells * 2u * sizeof(float)));
    check(hipMalloc(&miss, sizeof(uint))); check(hipMemset(miss, 0, sizeof(uint)));
    const float3 cell_size = make_float3(5, 4, 5) / 128.0f;
    for (unsigned i = 0; i < f::inputs.size(); ++i) {
        const auto v = f::inputs[i];
        const float3 coord = make_float3(v.cell % 128u, v.cell / 128u % 128u, v.cell / 16384u);
        const float3 minimum = make_float3(-1, 2, 3) + coord * cell_size - cell_size * 0.2f;
        const float3 size = cell_size + 2.0f * cell_size * 0.2f;
        KernelShaderEvalInput in[2]{};
        in[0].object = i; in[0].prim = std::bit_cast<int>(minimum.x); in[0].u = minimum.y; in[0].v = minimum.z;
        in[1].object = v.shader; in[1].prim = std::bit_cast<int>(size.x); in[1].u = size.y; in[1].v = size.z;
        check(hipMemcpy(input + v.cell * 2u, in, sizeof(in), hipMemcpyHostToDevice));
        probe<<<1, 1>>>(input, output, miss, v.cell); check(hipGetLastError());
        float out[2]{};
        check(hipMemcpy(out, output + v.cell * 2u, sizeof(out), hipMemcpyDeviceToHost));
        std::printf("%u %.9g %.9g\n", i, out[0], out[1]);
    }
    check(hipDeviceReset());
}
