#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 CYCLES_SOURCE GPU_ARCH [OUTPUT]" >&2
    exit 1
fi

cycles_source=$1
gpu_arch=$2
output=${3:-build/bin/cycles_benchmark_surface_closures}
cycles_include="${cycles_source}/intern/cycles"

if [[ ! -f "${cycles_include}/kernel/closure/bsdf_microfacet.h" ]]; then
    echo "error: '${cycles_source}' is not a Blender source tree with Cycles" >&2
    exit 1
fi

mkdir -p "$(dirname "${output}")"
hipcc \
    --offload-arch="${gpu_arch}" \
    -DHIPCC \
    -I "${cycles_include}" \
    -Wno-parentheses-equality \
    -Wno-unused-value \
    -ffast-math \
    -std=c++17 \
    -O3 \
    tools/benchmark_cycles_surface_closures.hip \
    -o "${output}"
