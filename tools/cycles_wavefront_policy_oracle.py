"""Run the actual Cycles host queue-policy functions with instrumented storage.

This extracts function bodies verbatim from the external Cycles tree. The
stand-ins only hold counters and record device-operation requests; they do not
implement scheduling, compaction decisions, GPU transport, or a CPU renderer.
The generated C++ and source hashes are retained in the explicit build folder.

Example:
  python tools/cycles_wavefront_policy_oracle.py /path/to/blender \
      --build-dir /var/tmp/wavefront-policy-oracle
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess


def function(source: str, signature: str) -> str:
    """Cycles definitions close at column zero; nested blocks are indented."""
    start = source.index(signature + "\n{")
    end = source.index("\n}\n", start) + 3
    return source[start:end]


def enum(source: str, prefix: str) -> str:
    start = source.index(prefix)
    end = source.index("\n};", start) + 3
    return source[start:end]


_HARNESS = r"""
struct IntegratorQueueCounter {
  int num_queued[DEVICE_GPU_KERNEL_INTEGRATOR_NUM]{};
};
template<typename T> struct Storage {
  T value{};
  T *data() { return &value; }
  const T *data() const { return &value; }
};
struct Queue {
  unsigned uploads{};
  template<typename T> void copy_to_device(Storage<T> &) { ++uploads; }
};
struct PathTraceWorkGPU {
  Storage<IntegratorQueueCounter> integrator_queue_counter_;
  Storage<int> integrator_next_shadow_path_index_;
  Queue queue_storage_;
  Queue *queue_ = &queue_storage_;
  unsigned compactions{};
  DeviceKernel get_most_queued_kernel() const;
  void compact_shadow_paths();
  void compact_paths(int, int, DeviceKernel, DeviceKernel, DeviceKernel) {
    ++compactions;
  }
};
"""

_MAIN = r"""
int main() {
  constexpr int stages[] = {
    DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_NEE,
    DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW,
    DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW,
    DEVICE_KERNEL_INTEGRATOR_SHADOW_PATH_MNEE_PENDING};
  std::printf("kernels %d %d %d %d %d %d\n",
    DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST,
    DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE,
    stages[0], stages[1], stages[2], stages[3]);
  // Boundaries of empty reset, minimum extent, overhead factor, and float
  // representability. These are inputs, never manually chosen expectations.
  constexpr std::array<int, 2> inputs[] = {
    {0, 0}, {1, 0}, {31, 0}, {32, 0}, {1048576, 0},
    {1, 1}, {19, 6}, {19, 13}, {31, 1}, {31, 15},
    {32, 16}, {32, 17}, {33, 16}, {33, 17},
    {63, 31}, {63, 32}, {64, 31}, {64, 32}, {64, 33},
    {65, 32}, {65, 33}, {1048575, 524287}, {1048575, 524288},
    {1048576, 524288}, {1048576, 524289},
    {33554431, 16777216}, {33554432, 16777217},
    {33554433, 16777217}, {33554434, 16777217}};
  for (auto input : inputs) {
    for (int distribution = 0; distribution < 5; ++distribution) {
      PathTraceWorkGPU work;
      auto &counts = work.integrator_queue_counter_.value.num_queued;
      const int extent = input[0], live = input[1];
      work.integrator_next_shadow_path_index_.value = extent;
      for (int i = 0; i < 4; ++i) {
        counts[stages[i]] = distribution == 4 ?
          live / 4 + int(i < live % 4) : (i == distribution ? live : 0);
      }
      work.compact_shadow_paths();
      std::printf("compact %d %d %d %d %d %d %u %u\n", extent,
        counts[stages[0]], counts[stages[1]], counts[stages[2]], counts[stages[3]],
        work.integrator_next_shadow_path_index_.value,
        work.compactions, work.queue_storage_.uploads);
    }
  }
  constexpr int queues[] = {
    DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST,
    DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE,
    DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_NEE,
    DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW,
    DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW,
    DEVICE_KERNEL_INTEGRATOR_SHADOW_PATH_MNEE_PENDING};
  for (int a : queues) {
    for (int b : queues) {
      for (int delta : {-1, 0, 1}) {
        PathTraceWorkGPU work;
        auto &counts = work.integrator_queue_counter_.value.num_queued;
        counts[a] = 8;
        counts[b] = 8 + delta;
        std::printf("select %d %d %d %d %d\n", a, counts[a], b, counts[b],
          int(work.get_most_queued_kernel()));
      }
    }
  }
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=pathlib.Path)
    parser.add_argument("--build-dir", required=True, type=pathlib.Path)
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()
    root = args.source_root.resolve()
    types = (root / "intern/cycles/kernel/types.h").read_text()
    implementation = (root / "intern/cycles/integrator/path_trace_work_gpu.cpp").read_text()
    pieces = [
        enum(types, "enum DeviceKernel : int {"),
        enum(types, "enum {\n  /* Megakernel"),
        function(implementation, "DeviceKernel PathTraceWorkGPU::get_most_queued_kernel() const"),
        function(implementation, "void PathTraceWorkGPU::compact_shadow_paths()"),
    ]
    digest = hashlib.sha256("\n".join(pieces).encode()).hexdigest()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    args.build_dir.mkdir(parents=True, exist_ok=True)
    source = args.build_dir.resolve() / "oracle.cpp"
    binary = args.build_dir.resolve() / "oracle"
    source.write_text(
        f"// Source revision: {revision}\n// Extracted source SHA-256: {digest}\n"
        + "#include <array>\n#include <cstdio>\n"
        + "\n".join(pieces[:2]) + _HARNESS + "\n".join(pieces[2:]) + _MAIN
    )
    subprocess.run([args.cxx, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)
    print(f"source {revision} {digest}", flush=True)
    subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
