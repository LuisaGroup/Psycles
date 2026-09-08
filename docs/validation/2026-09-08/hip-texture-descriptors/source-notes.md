# Descriptor audit provenance and reproduction

Delivery is repository Markdown, following the user's existing documentation
choice. Technical summary, findings, definitions, methods, limitations and
next/further questions are all present in the README; definitions precede the
time table, and next checks/further questions share the final section. Compact
tables are used for exact variant-by-access-pattern lookup and stage mapping;
there is no invented timeline, decorative chart or implication that these
microsecond-level kernels directly predict frame-render time.

Evidence root:
/var/tmp/psycles-native-volume-svm-06XnDX/texture-descriptor-audit-9taPHx.
The parent evidence directory contains the original-scene profile and images.
Source/API inspection, generated IR/ISA and GPU controls are distinguished
from synthetic timings and renderer timings. This follows the metric-diagnostic
and technical-report workflows; negative experiments are retained.

## Authoritative sources

- Cycles 5.2.1: /home/mike/Projects/blender-cycles-trace-5.2,
  cb168525138fecc792cc393f94afc39582b0103c. The retained light.cpp/svm.cpp
  diagnostic edits do not change image sampling.
- `intern/cycles/device/hip/device_impl.cpp`, filter/address construction;
  `kernel/device/gpu/image.h`, native and four-bilinear bicubic fetches;
  `util/types_image.h`, KernelImageTexture/KernelImageInfo indirection.
- Installed HIP 7.2.53211: /opt/rocm/include/hip/texture_types.h and
  amd_detail/texture_indirect_functions.h; ROCclr
  [7.2.0 source](https://github.com/ROCm/rocm-systems/blob/rocm-7.2.0/projects/clr/hipamd/src/hip_texture.cpp),
  `__hip_texture` prefix and `ihipCreateTextureObject` allocation request.
- HSA effective allocation properties come from the live
  [allocation probe](../../../../tools/hip_texture_allocation_probe.hip), not
  inference from the requested ROCclr SVM flag.
- Current child revision 9ea3b720f: src/backends/hip/hip_texture.{h,cpp},
  hip_bindless_array.{h,cpp}, llvm_codegen/hip_codegen_llvm_impl_resource.cpp.
  Child origin/next advanced independently to 686015e2b during the read-only
  audit; the tested gitlink/binary was not changed.
- Published renderer e2d38fc0; test-only fallback fix 251fc767. All six
  executable/library hashes still match the prior bound-sampler canaries.

## Commands

All HIP compilations use all 32 threads. Run from the verified repository root,
with output paths in a newly created diagnostic directory. The commands below
use the existing evidence directory so they identify the actual run; use new
output names when repeating, preserving the archived results.

```bash
/opt/rocm/bin/hipcc -parallel-jobs=32 --offload-arch=gfx1201 \
  -std=c++20 -O3 -ffast-math tools/hip_texture_descriptor_probe.hip \
  -o /var/tmp/psycles-native-volume-svm-06XnDX/texture-descriptor-audit-9taPHx/texture-layout-waterfall-probe

/var/tmp/psycles-native-volume-svm-06XnDX/texture-descriptor-audit-9taPHx/texture-layout-waterfall-probe \
  > /var/tmp/psycles-native-volume-svm-06XnDX/texture-descriptor-audit-9taPHx/layout-waterfall-timings.csv \
  2> /var/tmp/psycles-native-volume-svm-06XnDX/texture-descriptor-audit-9taPHx/layout-waterfall-validation.log

/opt/rocm/bin/hipcc -parallel-jobs=32 --offload-arch=gfx1201 \
  -std=c++20 -O3 tools/hip_texture_allocation_probe.hip \
  -L/opt/rocm/lib -lhsa-runtime64 \
  -o /var/tmp/psycles-native-volume-svm-06XnDX/texture-descriptor-audit-9taPHx/texture-allocation-probe
```

The executed source paths were the corresponding temporary `.hip` files in
that directory, byte-identical to the committed tools. The source and binary
hashes are in audit-summary.json. To generate assembly, use the same HIP
compiler options with `--cuda-device-only -S` and an `.s` output. The original
Cycles image oracle uses tools/cycles_svm_image_binding_oracle.hip with
`-DHIPCC -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles`; use
`--cuda-device-only -S -emit-llvm` for LLVM IR.

Actual Luisa integration capture:

```bash
env LUISA_DUMP_LLVM_IR=1 \
  LUISA_DUMP_HIP_ISA=/var/tmp/psycles-native-volume-svm-06XnDX/texture-descriptor-audit-9taPHx \
  /home/mike/Projects/Psycles-surface-svm/build/bin/psycles_luisa_cycles_svm_image_binding_tests hip
```

That command ran **inside the isolated evidence directory** so its LLVM dumps
did not overwrite the user's existing root-worktree dumps. It passed 640 checks.
The code object was disassembled using /opt/rocm/llvm/bin/llvm-objdump with
`--disassemble --mcpu=gfx1201`.

The original-scene profile's exact renderer command and profiler prefix are in
audit-summary.json. Total samples and requested count are both 64; otherwise
it matches the prior bound-sampler canary command. No own build, benchmark or
test overlaps its rendering, or the synthetic timing runs. This remains an
ordinary desktop session, not an exclusively controlled GPU environment.

The fifteen-pass profile-image comparison uses the retained
barbershop-matched-kernel-profile-cycles.exr and its matching `.json` metadata,
plus the current export's scene.json. Both report build 9e2066aef7ef. An initial
comparison command omitted metadata and was rejected before comparison; the
verified retry supplies both identities, without weakening the comparator.
All pass metrics and source hashes are retained in audit-summary.json. The
analyzer additionally verifies finite values in all 46 actual channels,
including Combined alpha, not just the RGB pass comparisons.

Reproduce the schema/count/order/finite/checksum audit with:

```bash
python3 docs/validation/2026-09-08/hip-texture-descriptors/analyze_evidence.py \
  /var/tmp/psycles-native-volume-svm-06XnDX/texture-descriptor-audit-9taPHx --summary-only
```

## Retained data and experiment boundaries

The `data/` directory retains every timed sample and validation receipt from
the two six-variant process runs and the final seven-variant run. The latter
adds explicit index grouping. The initial exploratory version used a generic
pointer load (flat address space) and is not used in the reported table;
its source/log/CSV remains in the evidence directory. Later versions use the
same explicit global address space as Luisa's image-SRD loads. All artifacts
are separate files, not overwritten captures.

An initial harness build warned that floating `isfinite` is invalid under
fast math. Before measurement, the comparator was changed to integer exponent
checks; the production renderer and its fast-math options were never changed.
Neither this setup correction nor the rejected layout variants is claimed as
a compiler bug or a production regression fix.

Native-prefix clones copy all 80 prefix bytes, including the image metadata
used by HIP's device wrapper, into a 96-byte aligned record. Clones are not
registered runtime objects and are freed only as their `hipMalloc` backing
array. Real texture objects and pixel arrays remain alive through all kernels.
The compact controls instead copy the eight/four dwords consumed by the native
instruction. Both controls validate against untouched native `tex2D` objects.

The explicit grouping experiment is correct because equal descriptor indices
identify equal immutable image/sampler pairs in this fixture. It preserves
per-lane coordinates and makes progress by removing at least one active lane
per iteration. The same argument would need to include custom sampler state,
mips and resource ownership in a generic backend; no such backend change is
made. Its poor measured result is retained rather than hidden behind a lower
instruction or descriptor-load count.
