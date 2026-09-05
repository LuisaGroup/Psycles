// Original Cycles 5.2.1 affine inversion and camera NDC service.
// hipcc --offload-arch=gfx1201 -DHIPCC -std=c++20 -O3 \
//   -I include -I /path/to/blender/intern/cycles \
//   tools/cycles_camera_transform_oracle.hip -o /tmp/camera-transform-oracle
#include "../tests/cycles_camera_transform_fixture.h"

#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/camera/camera.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace {
void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(error));
    std::exit(1);
  }
}

__global__ void inverse_probe(Transform camera_to_world, Transform *output) {
  *output = transform_inverse(camera_to_world);
}

__global__ void ndc_probe(float3 position, float4 *output) {
  ShaderDataTinyStorage storage{};
  auto *sd = AS_SHADER_DATA(&storage);
  sd->P = position;
  sd->object = OBJECT_NONE;
  const auto ndc = camera_world_to_ndc(nullptr, sd, position);
  *output = make_float4(ndc, 0.0f);
}
} // namespace

int main() {
  Transform *inverse{};
  float4 *output{};
  check(hipMalloc(&inverse, sizeof(Transform)));
  check(hipMalloc(&output, sizeof(float4)));
  auto params = std::make_unique<KernelParamsHIP>();
  unsigned index{};
  for (const auto &input : psycles::test_support::camera_transform_cases) {
    const auto &e = input.camera_to_world.elements;
    const Transform forward{make_float4(e[0], e[4], e[8], e[12]),
                            make_float4(e[1], e[5], e[9], e[13]),
                            make_float4(e[2], e[6], e[10], e[14])};
    inverse_probe<<<1, 1>>>(forward, inverse);
    check(hipGetLastError());
    auto &cam = params->data.cam;
    check(hipMemcpy(&cam.worldtocamera, inverse, sizeof(Transform),
                    hipMemcpyDeviceToHost));
    cam.cameratoworld = forward;
    cam.type = CAMERA_ORTHOGRAPHIC;
    // Unit aspect, vertical span 2, no shift. This is an input projection,
    // evaluated by Cycles camera_world_to_ndc below.
    const auto &m = cam.worldtocamera;
    cam.worldtondc = {
        make_float4(0.5f * m.x.x, 0.5f * m.x.y, 0.5f * m.x.z, 0.5f * m.x.w + 0.5f),
        make_float4(0.5f * m.y.x, 0.5f * m.y.y, 0.5f * m.y.z, 0.5f * m.y.w + 0.5f),
        make_float4(-m.z.x, -m.z.y, -m.z.z, -m.z.w), make_float4(0, 0, 0, 1)};
    check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), params.get(), sizeof(*params)));
    ndc_probe<<<1, 1>>>(make_float3(input.position.x, input.position.y,
                                  input.position.z), output);
    check(hipGetLastError());
    std::array<float4, 1u> values{};
    check(hipMemcpy(values.data(), output, sizeof(values), hipMemcpyDeviceToHost));
    const std::array<float4, 4u> columns{
        make_float4(m.x.x, m.y.x, m.z.x, 0.0f),
        make_float4(m.x.y, m.y.y, m.z.y, 0.0f),
        make_float4(m.x.z, m.y.z, m.z.z, 0.0f),
        make_float4(m.x.w, m.y.w, m.z.w, 1.0f)};
    std::printf("%u", index++);
    for (const auto &v : columns) {
      std::printf(" %.9g %.9g %.9g %.9g", v.x, v.y, v.z, v.w);
    }
    for (const auto &v : values) {
      std::printf(" %.9g %.9g %.9g %.9g", v.x, v.y, v.z, v.w);
    }
    std::printf("\n");
  }
  check(hipFree(output));
  check(hipFree(inverse));
}
