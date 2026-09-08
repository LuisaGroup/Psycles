// Original Cycles GPU Voronoi; no copied shader implementation.
// hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC -std=c++20 -O3 \
//   -ffast-math -I /path/to/blender/intern/cycles this_file -o ORACLE
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/features.h"
#include "kernel/svm/voronoi.h"
#include "../tests/cycles_svm_voronoi_octave_test_data.h"
#include <cstdio>
#include <cstdlib>

namespace fixture = psycles::test_support::voronoi_octave;
namespace {
using Input = fixture::Case<SVMNodeTexVoronoi>;
void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(error));
    std::exit(1);
  }
}
template<unsigned Mask>
__global__ void probe(const Input *inputs, float *output) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= fixture::case_count) { return; }
  float stack[fixture::stack_size];
  for (unsigned slot = 0; slot < 8u; ++slot) { stack[slot] = inputs[i].inputs[slot]; }
  svm_node_tex_voronoi<Mask>(stack, inputs[i].node);
  for (unsigned slot = 0; slot < fixture::output_count; ++slot) {
    output[i * fixture::output_count + slot] = stack[fixture::output_begin + slot];
  }
}
} // namespace

int main() {
  const auto inputs = fixture::cases<SVMNodeTexVoronoi>();
  Input *device_inputs{};
  float *device_output{};
  std::array<float, fixture::case_count * fixture::output_count> output;
  check(hipMalloc(&device_inputs, sizeof(inputs)));
  check(hipMalloc(&device_output, sizeof(output)));
  check(hipMemcpy(device_inputs, inputs.data(), sizeof(inputs), hipMemcpyHostToDevice));
  for (unsigned extra = 0; extra < 2u; ++extra) {
    if (extra != 0u) {
      probe<KERNEL_FEATURE_NODE_VORONOI_EXTRA><<<1, 128>>>(device_inputs, device_output);
    } else {
      probe<0u><<<1, 128>>>(device_inputs, device_output);
    }
    check(hipGetLastError());
    check(hipMemcpy(output.data(), device_output, sizeof(output), hipMemcpyDeviceToHost));
    for (unsigned i = 0; i < fixture::case_count; ++i) {
      std::printf("%u", extra * fixture::case_count + i);
      for (unsigned slot = 0; slot < fixture::output_count; ++slot) {
        std::printf(" %.9g", output[i * fixture::output_count + slot]);
      }
      std::printf("\n");
    }
  }
  check(hipFree(device_output));
  check(hipFree(device_inputs));
}
