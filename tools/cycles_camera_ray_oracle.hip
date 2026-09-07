// hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC -std=c++20 \
//   -O3 -ffast-math -I include -I /path/to/blender/intern/cycles \
//   tools/cycles_camera_ray_oracle.hip -o /tmp/camera-ray-oracle
#include "../tests/cycles_camera_projection_fixture.h"
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/camera/camera.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>

namespace {
void check(hipError_t result) {
  if (result != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(result));
    std::exit(1);
  }
}
__global__ void probe(float2 raster, float2 lens, bool orthographic, float4 *out) {
  Ray ray{};
  ray.time = 0.5f;
  if (orthographic) { camera_sample_orthographic(nullptr, raster, lens, &ray); }
  else { camera_sample_perspective(nullptr, raster, lens, &ray); }
  out[0] = make_float4(ray.P, ray.dP);
  out[1] = make_float4(ray.D, ray.dD);
  out[2] = make_float4(ray.tmin, ray.tmax, ray.time, 0);
  ShaderDataTinyStorage storage{};
  auto *sd = AS_SHADER_DATA(&storage);
  sd->object = 0;
  // Window coordinates at a surface along the ray, not exactly on the near
  // plane. At a 0.001 near plane, reprojecting rounded world-space ray.P is
  // ill-conditioned and amplifies ordinary float roundoff by about 1000x.
  out[3] = make_float4(camera_world_to_ndc(nullptr, sd, ray.P + 2.0f * ray.D), 0);
  sd->object = OBJECT_NONE;
  out[4] = make_float4(camera_world_to_ndc(nullptr, sd, ray.D), 0);
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 2) { return 1; }
  std::ifstream source{argv[1]};
  auto params = std::make_unique<KernelParamsHIP>();
  float4 *output{};
  check(hipMalloc(&output, 5u * sizeof(float4)));
  unsigned index{};
  using namespace psycles::test_support;
  for (const auto &input : camera_projection_inputs) {
    unsigned scenario{};
    source >> scenario;
    if (scenario != index++) { return 2; }
    std::array<float4, 10u> rows{};
    for (auto &v : rows) { source >> v.x >> v.y >> v.z >> v.w; }
    if (!source) { return 3; }
    auto &cam = params->data.cam;
    cam.rastertocamera = {rows[0], rows[1], rows[2], rows[3]};
    cam.worldtondc = {rows[6], rows[7], rows[8], rows[9]};
    cam.type = input.orthographic ? CAMERA_ORTHOGRAPHIC : CAMERA_PERSPECTIVE;
    const auto &e = input.transform.elements;
    cam.cameratoworld = {make_float4(e[0], e[4], -e[8], e[12]),
                          make_float4(e[1], e[5], -e[9], e[13]),
                          make_float4(e[2], e[6], -e[10], e[14])};
    cam.dx = rows[4]; cam.dy = rows[5]; cam.differential_scale = 1;
    cam.nearclip = input.near_clip;
    cam.cliplength = input.far_clip - input.near_clip;
    cam.aperturesize = input.aperture; cam.focaldistance = input.focus;
    cam.blades = input.blades < 3u ? 0.0f : float(input.blades);
    cam.bladesrotation = input.rotation;
    cam.inv_aperture_ratio = 1.0f / input.aperture_ratio;
    check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), params.get(), sizeof(*params)));
    unsigned sample{};
    for (const auto raster : camera_raster_fractions) {
      const auto xy = make_float2(raster.x * input.width + 0.3f,
                                  raster.y * input.height + 0.7f);
      probe<<<1, 1>>>(xy, make_float2(camera_lens_sample.x, camera_lens_sample.y),
                       input.orthographic, output);
      check(hipGetLastError());
      std::array<float4, 5u> values{};
      check(hipMemcpy(values.data(), output, sizeof(values), hipMemcpyDeviceToHost));
      std::printf("%u %u %.9g %.9g", scenario, sample++, xy.x, xy.y);
      for (const auto v : values) {
        std::printf(" %.9g %.9g %.9g %.9g", v.x, v.y, v.z, v.w);
      }
      std::puts("");
    }
  }
  check(hipFree(output));
}
