// Execute original Cycles consumers on original scene-owned KernelLight fields.
// hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC -std=c++20 -O3 \
//   -ffast-math -I /path/to/blender/intern/cycles THIS -o /tmp/light-oracle
// /tmp/light-oracle tests/data/cycles_light_parameters.txt > /tmp/state.txt
// Capture the FTZ companion with the same command plus
// -fgpu-flush-denormals-to-zero (fallback's explicit FTZ/DAZ environment).
#include "../tests/cycles_light_parameters_fixture.h"
// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/light/light.h"
// clang-format on
#include <cstdio>
#include <cstdlib>

namespace f = psycles::test_support::light_parameters;
static void check(hipError_t e) {
  if (e != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(e)); std::exit(1); }
}

__device__ void save_sample(float *out, const KernelLight *k, bool valid, const LightSample &ls) {
  out[0] = float(valid);
  // On false, the native function may have returned before defining fields.
  // Only validity is part of the comparison for rejected samples.
  if (!valid) { return; }
  out[1] = ls.eval_fac; out[2] = ls.pdf; out[3] = ls.t;
  out[4] = ls.D.x; out[5] = ls.D.y; out[6] = ls.D.z;
  out[7] = ls.P.x; out[8] = ls.P.y; out[9] = ls.P.z;
  out[10] = ls.Ng.x; out[11] = ls.Ng.y; out[12] = ls.Ng.z;
  const auto uv = k->type == LIGHT_AREA ? area_light_uv(k, ls.P) :
      spot_light_uv(spot_light_to_local(nullptr, k, -ls.D), k->spot.half_cot_half_spot_angle);
  out[13] = uv.x; out[14] = uv.y;
}

__global__ void probe(float *output, unsigned row, float3 offset, float2 random,
                      float3 local_ray, float3 N, float3 interval_direction, unsigned ordinal) {
  const auto k = &kernel_data_fetch(lights, row);
  auto out = output + ordinal * f::outputs;
  for (unsigned i = 0; i < f::outputs; ++i) { out[i] = 0; }
  const auto P = k->co + offset;
  if (k->type == LIGHT_SPOT) {
    out[0] = spot_light_attenuation(&k->spot, local_ray);
    const auto uv = spot_light_uv(local_ray, k->spot.half_cot_half_spot_angle);
    out[1] = uv.x; out[2] = uv.y;
    out[3] = 1.0f - k->spot.cos_half_larger_spread;
    LightSample s{};
    const bool valid = spot_light_sample<false>(nullptr, k, random, P, N, 0, &s);
    save_sample(out + 4, k, valid, s);
    LightSample v{};
    // Original light_sample_from_volume_segment supplies transmission support.
    const bool volume_valid = spot_light_sample<true>(nullptr, k, random, P, N,
                                                     SD_BSDF_HAS_TRANSMISSION, &v);
    save_sample(out + 20, k, volume_valid, v);
    Interval<float> interval{0, 10};
    const bool interval_valid = spot_light_valid_ray_segment(nullptr, k, P, interval_direction, &interval);
    out[36] = float(interval_valid);
    if (interval_valid) { out[37] = interval.min; out[38] = interval.max; }
  } else {
    const bool spread = k->area.normalize_spread > 0;
    out[0] = spread ? area_light_spread_attenuation(-local_ray, make_float3(0,0,1),
                         k->area.tan_half_spread, k->area.normalize_spread) : 1.0f;
    out[3] = float(spread);
    LightSample s{};
    const bool valid = area_light_sample<false>(k, random, P, &s);
    save_sample(out + 4, k, valid, s);
    LightSample v{};
    const bool volume_valid = area_light_sample<true>(k, random, P, &v);
    save_sample(out + 20, k, volume_valid, v);
    Interval<float> interval{0, 10};
    const bool interval_valid = area_light_valid_ray_segment(&k->area, P - k->co, interval_direction, &interval);
    out[36] = float(interval_valid);
    if (interval_valid) { out[37] = interval.min; out[38] = interval.max; }
  }
}

__global__ void invert_objects(KernelObject *objects, unsigned count) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) { objects[i].itfm = transform_inverse(objects[i].itfm); }
}

int main(int argc, char **argv) {
  if (argc != 2) { return 2; }
  const auto records = f::read(argv[1]);
  std::vector<KernelLight> lights(records.size());
  std::vector<KernelObject> objects(records.size());
  for (unsigned i = 0; i < records.size(); ++i) {
    const auto &r = records[i]; const auto &v = r.v;
    const auto vec = [&](unsigned offset) { return make_float3(v[offset], v[offset+1], v[offset+2]); };
    auto &k = lights[i]; k.type = r.spot ? LIGHT_SPOT : LIGHT_AREA;
    k.co = vec(9); k.object_id = i;
    // Original scene transform inversion for the native object table.
    Transform tfm{make_float4(v[0], v[3], v[6], v[9]),
                  make_float4(v[1], v[4], v[7], v[10]),
                  make_float4(v[2], v[5], v[8], v[11])};
    objects[i].itfm = tfm;
    if (r.spot) {
      k.spot.cos_half_spot_angle = v[16]; k.spot.half_cot_half_spot_angle = v[17];
      k.spot.spot_smooth = v[18]; k.spot.cos_half_larger_spread = v[19];
      k.spot.ray_segment_dp = v[20]; k.spot.eval_fac = v[21];
      k.spot.radius = v[22]; k.spot.is_sphere = r.sphere && v[22] != 0;
      k.spot.dir = vec(25);
    } else {
      k.area.tan_half_spread = v[16]; k.area.normalize_spread = v[17];
      k.area.invarea = v[21]; k.area.len_u = v[23]; k.area.len_v = v[24];
      k.area.dir = vec(25); k.area.axis_u = vec(28); k.area.axis_v = vec(31);
    }
  }
  KernelLight *device_lights{}; KernelObject *device_objects{}; float *output{};
  check(hipMalloc(&device_lights, lights.size() * sizeof(KernelLight)));
  check(hipMalloc(&device_objects, objects.size() * sizeof(KernelObject)));
  const auto count = records.size() * f::probes * f::outputs;
  check(hipMalloc(&output, count * sizeof(float)));
  check(hipMemcpy(device_lights, lights.data(), lights.size()*sizeof(KernelLight), hipMemcpyHostToDevice));
  check(hipMemcpy(device_objects, objects.data(), objects.size()*sizeof(KernelObject), hipMemcpyHostToDevice));
  invert_objects<<<(objects.size()+63u)/64u, 64u>>>(device_objects, objects.size());
  check(hipGetLastError());
  KernelParamsHIP params{}; params.lights = device_lights; params.objects = device_objects;
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
  for (unsigned i = 0; i < records.size(); ++i) {
    for (unsigned j = 0; j < f::probes; ++j) {
      const auto p = f::input(records[i],j);
      const auto vec = [](const auto &v) { return make_float3(v[0],v[1],v[2]); };
      probe<<<1, 1>>>(output, i, vec(p.offset), make_float2(p.random[0],p.random[1]),
          vec(p.local_ray), vec(p.normal), vec(p.interval_direction), i*f::probes+j);
      check(hipGetLastError());
    }
  }
  std::vector<float> result(count);
  check(hipMemcpy(result.data(), output, count*sizeof(float), hipMemcpyDeviceToHost));
  for (unsigned i = 0; i < records.size()*f::probes; ++i) {
    std::printf("%u", i);
    for (unsigned j = 0; j < f::outputs; ++j) {
      std::printf(" %u", std::bit_cast<std::uint32_t>(result[i*f::outputs+j]));
    }
    std::puts("");
  }
  check(hipFree(output)); check(hipFree(device_objects)); check(hipFree(device_lights));
}
