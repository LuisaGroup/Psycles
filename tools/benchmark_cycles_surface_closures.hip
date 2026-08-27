// External Cycles 5.2 oracle for tools/benchmark_surface_closures.cpp.
//
// This file deliberately calls Cycles' public device closure routines through
// its source headers. It does not duplicate or translate their implementation.

#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/closure/bsdf_diffuse.h"
#include "kernel/closure/bsdf_microfacet.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <utility>

namespace {

constexpr auto block_size = 256u;
constexpr auto warmup_count = 3u;

enum class ProbeKind {
  diffuse,
  glossy_ggx_regular,
  glossy_ggx_singular,
};

template<ProbeKind kind>
__device__ float probe_roughness(const float4 value)
{
  if constexpr (kind == ProbeKind::glossy_ggx_singular) {
    return 0.001f + 0.001f * fabsf(value.z);
  }
  return 0.2f + 0.6f * fabsf(value.z);
}

template<ProbeKind kind>
__device__ void run_probe(const float4 value,
                          const unsigned int iterations,
                          float4 *result)
{
  const float3 normal = normalize(
      make_float3(value.x * 0.25f, value.y * 0.25f, 1.0f));
  const float3 incoming = normalize(make_float3(0.13f + value.x * 0.1f,
                                                 -0.21f + value.y * 0.1f,
                                                 0.968f));
  float2 random = make_float2(fminf(fmaxf(value.z, 0.001f), 0.999f),
                              fminf(fmaxf(value.w, 0.001f), 0.999f));
  float4 accumulated = zero_float4();

  if constexpr (kind == ProbeKind::diffuse) {
    DiffuseBsdf bsdf{};
    bsdf.N = normal;
    for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
      Spectrum evaluation = zero_spectrum();
      float3 outgoing = zero_float3();
      float pdf = 0.0f;
      const int label = bsdf_diffuse_sample(
          reinterpret_cast<const ShaderClosure *>(&bsdf),
          normal,
          incoming,
          random,
          &evaluation,
          &outgoing,
          &pdf);
      accumulated += make_float4(evaluation.x,
                                 outgoing.x + outgoing.y,
                                 pdf,
                                 static_cast<float>(label));
      random += make_float2(0.61803398875f, 0.41421356237f);
      if (random.x >= 1.0f) {
        random.x -= 1.0f;
      }
      if (random.y >= 1.0f) {
        random.y -= 1.0f;
      }
    }
  }
  else {
    const float roughness = probe_roughness<kind>(value);
    MicrofacetBsdf bsdf{};
    bsdf.N = normal;
    bsdf.alpha_x = roughness * roughness;
    bsdf.alpha_y = bsdf.alpha_x;
    bsdf.ior = 1.0f;
    bsdf.energy_scale = 1.0f;
    bsdf.fresnel_type = MicrofacetFresnel::NONE;
    bsdf.fresnel = nullptr;
    bsdf.T = normalize(make_float3(1.0f, 0.23f, 0.0f));
    bsdf.type = CLOSURE_BSDF_MICROFACET_GGX_ID;
    for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
      Spectrum evaluation = zero_spectrum();
      float3 outgoing = zero_float3();
      float pdf = 0.0f;
      float2 sampled_roughness = zero_float2();
      float eta = 1.0f;
      const int label = bsdf_microfacet_ggx_sample(
          nullptr,
          reinterpret_cast<const ShaderClosure *>(&bsdf),
          normal,
          incoming,
          make_float3(random.x, random.y, random.x),
          &evaluation,
          &outgoing,
          &pdf,
          &sampled_roughness,
          &eta);
      accumulated += make_float4(evaluation.x,
                                 outgoing.x + outgoing.y,
                                 pdf,
                                 static_cast<float>(label));
      random += make_float2(0.61803398875f, 0.41421356237f);
      if (random.x >= 1.0f) {
        random.x -= 1.0f;
      }
      if (random.y >= 1.0f) {
        random.y -= 1.0f;
      }
    }
  }
  *result = accumulated;
}

template<ProbeKind kind>
__device__ void probe_kernel_body(const float4 *input,
                                  float4 *output,
                                  const unsigned int count,
                                  const unsigned int iterations)
{
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  run_probe<kind>(input[index], iterations, &output[index]);
}

extern "C" __global__ __launch_bounds__(block_size)
void cycles_diffuse_probe(const float4 *input,
                          float4 *output,
                          const unsigned int count,
                          const unsigned int iterations)
{
  probe_kernel_body<ProbeKind::diffuse>(input, output, count, iterations);
}

extern "C" __global__ __launch_bounds__(block_size)
void cycles_glossy_ggx_regular_probe(const float4 *input,
                                     float4 *output,
                                     const unsigned int count,
                                     const unsigned int iterations)
{
  probe_kernel_body<ProbeKind::glossy_ggx_regular>(
      input, output, count, iterations);
}

extern "C" __global__ __launch_bounds__(block_size)
void cycles_glossy_ggx_singular_probe(const float4 *input,
                                      float4 *output,
                                      const unsigned int count,
                                      const unsigned int iterations)
{
  probe_kernel_body<ProbeKind::glossy_ggx_singular>(
      input, output, count, iterations);
}

extern "C" __global__ __launch_bounds__(block_size)
void initialize_surface_closure_probe(float4 *input,
                                      const unsigned int count)
{
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const float lane = static_cast<float>(index & 255u) * (1.0f / 256.0f);
    input[index] = make_float4(0.13f,
                               -0.17f,
                               0.001f + 0.997f * lane,
                               0.999f - 0.997f * lane);
  }
}

void check(const hipError_t error, const char *operation)
{
  if (error != hipSuccess) {
    std::fprintf(stderr,
                 "%s failed: %s\n",
                 operation,
                 hipGetErrorString(error));
    std::exit(EXIT_FAILURE);
  }
}

unsigned int parse_u32(const char *text, const char *label)
{
  errno = 0;
  char *end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' ||
      value > std::numeric_limits<unsigned int>::max()) {
    std::fprintf(stderr, "%s must be uint32: %s\n", label, text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<unsigned int>(value);
}

template<typename Launch>
void benchmark(const char *name,
               const unsigned int count,
               const unsigned int iterations,
               const unsigned int repeats,
               Launch &&launch)
{
  float4 *input = nullptr;
  float4 *output = nullptr;
  check(hipMalloc(&input, sizeof(float4) * count), "hipMalloc(input)");
  check(hipMalloc(&output, sizeof(float4) * count), "hipMalloc(output)");
  const unsigned int grid_size =
      (count + block_size - 1u) / block_size;
  initialize_surface_closure_probe<<<grid_size, block_size>>>(input, count);
  for (unsigned int warmup = 0; warmup < warmup_count; ++warmup) {
    launch(grid_size, input, output, count, iterations);
  }
  check(hipDeviceSynchronize(), "warmup");

  hipEvent_t begin{};
  hipEvent_t end{};
  check(hipEventCreate(&begin), "hipEventCreate(begin)");
  check(hipEventCreate(&end), "hipEventCreate(end)");
  for (unsigned int repeat = 0; repeat < repeats; ++repeat) {
    check(hipEventRecord(begin), "hipEventRecord(begin)");
    launch(grid_size, input, output, count, iterations);
    check(hipEventRecord(end), "hipEventRecord(end)");
    check(hipEventSynchronize(end), "hipEventSynchronize(end)");
    float milliseconds = 0.0f;
    check(hipEventElapsedTime(&milliseconds, begin, end),
          "hipEventElapsedTime");
    const double ns_per_handler =
        milliseconds * 1.0e6 / static_cast<double>(count) /
        static_cast<double>(iterations);
    std::printf("repeat=%u milliseconds=%.6f ns_per_handler=%.9f\n",
                repeat,
                milliseconds,
                ns_per_handler);
  }
  float4 checksum{};
  check(hipMemcpy(&checksum,
                  output,
                  sizeof(checksum),
                  hipMemcpyDeviceToHost),
        "hipMemcpy(checksum)");
  std::printf(
      "checksum=%.9g,%.9g,%.9g,%.9g count=%u iterations=%u "
      "warmups=%u contract=selected_closure_sample probe=%s\n",
      checksum.x,
      checksum.y,
      checksum.z,
      checksum.w,
      count,
      iterations,
      warmup_count,
      name);

  check(hipEventDestroy(begin), "hipEventDestroy(begin)");
  check(hipEventDestroy(end), "hipEventDestroy(end)");
  check(hipFree(input), "hipFree(input)");
  check(hipFree(output), "hipFree(output)");
}

}  // namespace

int main(int argc, char **argv)
{
  if (argc < 2 || argc > 5) {
    std::fprintf(stderr,
                 "usage: %s PROBE [COUNT [ITERATIONS [REPEATS]]]\n"
                 "probes: diffuse glossy_ggx_regular glossy_ggx_singular\n",
                 argv[0]);
    return EXIT_FAILURE;
  }
  const std::string_view probe{argv[1]};
  const unsigned int count =
      argc > 2 ? parse_u32(argv[2], "count") : 1u << 22u;
  const unsigned int iterations =
      argc > 3 ? parse_u32(argv[3], "iterations") : 32u;
  const unsigned int repeats =
      argc > 4 ? parse_u32(argv[4], "repeats") : 5u;
  if (count == 0u || iterations == 0u || repeats == 0u) {
    std::fprintf(stderr, "count, iterations, and repeats must be positive\n");
    return EXIT_FAILURE;
  }

  if (probe == "diffuse") {
    benchmark(argv[1], count, iterations, repeats,
              [](const unsigned int grid_size,
                 const float4 *input,
                 float4 *output,
                 const unsigned int launch_count,
                 const unsigned int launch_iterations) {
                cycles_diffuse_probe<<<grid_size, block_size>>>(
                    input, output, launch_count, launch_iterations);
              });
  }
  else if (probe == "glossy_ggx_regular") {
    benchmark(argv[1], count, iterations, repeats,
              [](const unsigned int grid_size,
                 const float4 *input,
                 float4 *output,
                 const unsigned int launch_count,
                 const unsigned int launch_iterations) {
                cycles_glossy_ggx_regular_probe<<<grid_size, block_size>>>(
                    input, output, launch_count, launch_iterations);
              });
  }
  else if (probe == "glossy_ggx_singular") {
    benchmark(argv[1], count, iterations, repeats,
              [](const unsigned int grid_size,
                 const float4 *input,
                 float4 *output,
                 const unsigned int launch_count,
                 const unsigned int launch_iterations) {
                cycles_glossy_ggx_singular_probe<<<grid_size, block_size>>>(
                    input, output, launch_count, launch_iterations);
              });
  }
  else {
    std::fprintf(stderr, "unknown probe: %s\n", argv[1]);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
