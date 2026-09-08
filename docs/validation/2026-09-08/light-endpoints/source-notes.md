# Sources and reproduction

The selected delivery is repository Markdown, continuing the user's explicit
documentation choice. The technical summary, evidence, definitions, methods,
limitations and next questions map to the technical-report structure. Scope
precedes the count table; next work and further questions are combined.
Tables are exact stage/engine and before/after lookups, not a padded grid-size
chart or an invented time series. The diagnostic, report and visualization
workflows keep active work, dispatches, GPU time, render wall time and image
error as separate measurements.

Evidence directory: `/var/tmp/psycles-active-work-NtKLIR`.
Psycles baseline: `d9146fef`, production renderer from `e2d38fc0`.
Published correction: `7419301e` on `origin/main`.
Luisa stays at `9ea3b720f`; no child source or gitlink changes are included.
Hardware is RX 9070 XT / gfx1201, HIP 7.2.53211, with 32 host threads.

## Source-of-truth boundaries

- External Cycles source:
  `/home/mike/Projects/blender-cycles-trace-5.2`, revision
  `cb168525138fecc792cc393f94afc39582b0103c`. The existing diagnostic edits in
  `intern/cycles/scene/light.cpp` and `svm.cpp` are preserved. The GPU probe
  includes original device headers; it does not compile either dirty host file.
- `kernel/light/light.h::lights_intersect_impl<true>` owns intersection
  eligibility, with `point.h`, `spot.h`, `area.h` owning the geometry/evaluation
  separation. `common.h::is_light_shader_visible_to_path` owns visibility.
- `kernel/integrator/shade_light.h::integrate_light_forward` advances ray tmin
  before empty/visibility returns. `integrator_shade_light_forward` increments
  transparent bounce and schedules INTERSECT_CLOSEST even for empty emission.
- `device/queue.cpp::debug_enqueue_begin` logs the exact work_size passed by
  `integrator/path_trace_work_gpu.cpp::enqueue_path_iteration`. The GPU shading
  kernels guard `global_index < work_size` and operate on queued indices.
- Camera initialization is different: `kernel/device/gpu/kernel.h` launches
  max_tile_work_size times num_tiles and rejects each tile's padding. Its
  114,982,144 logged items are not 112,459,776 generated camera paths.
- Luisa `include/luisa/coro/schedulers/wavefront.h` records executed_count in
  the same host resume operation that dispatches exactly count frames. Entry,
  continuations, pre-resume extensions and auxiliary work are separate records.
- Performance/image oracle executable:
  `/home/mike/Projects/blender-install-5.2-hiprt/blender`, build `9e2066aef7ef`.
  The trace-build executable is not substituted for this original-scene run.

## Exact work-count commands

These commands ran serially before any subsequent build. For repetition use a
new output directory; the analyzer intentionally refuses to overwrite prior
image-comparison reports. The second Psycles command below was rerun after the
fix with `psycles-fixed.ppm` / `psycles-fixed.log` output names.

```bash
env -u CYCLES_DEBUG_PER_KERNEL_PERFORMANCE \
  /home/mike/Projects/blender-install-5.2-hiprt/blender \
  /var/tmp/psycles-official-redownload-20260814/barbershop_interior.blend \
  --background --debug-cycles --log cycles --log-level trace \
  --python-exit-code 1 \
  --python /home/mike/Projects/Psycles-surface-svm/tools/render_cycles_golden.py -- \
  /var/tmp/psycles-active-work-NtKLIR/cycles.exr 2048 858 64 \
  --cycles-device HIP --device-name 'Radeon RX 9070 XT' \
  > /var/tmp/psycles-active-work-NtKLIR/cycles.log 2>&1

env PSYCLES_DISABLE_SHADER_CACHE=1 LUISA_CORO_SHADER_MAP=1 \
  LUISA_CORO_WAVEFRONT_STATS=1 \
  /home/mike/Projects/Psycles-surface-svm/build/bin/psycles_render_blender_scene \
  /var/tmp/psycles-four-scene-hip-0PRq34/barbershop/export \
  /var/tmp/psycles-active-work-NtKLIR/psycles.ppm \
  hip 2048 858 64 64 - 1024 429 0 0 64 - 1 0 wavefront-staged \
  32 32768 32 1 1 1 4 2 0 0 0 0 1 1048576 \
  > /var/tmp/psycles-active-work-NtKLIR/psycles.log 2>&1
```

Both sample arguments are 64. The fresh Cycles metadata preserves authored
seed 0, animated effective seed 1267069554, frame 1, Tabulated Sobol,
scrambling distance 1, no automatic scrambling, no adaptive sampling and no
denoising. The exported scene hash matches the retained equal-pass manifest.
The existing host statistics do not insert shader counters, modify local
array bounds or create a profile-driven scene specialization.

Analyze with:

```bash
python3 docs/validation/2026-09-08/light-endpoints/analyze_work.py \
  /var/tmp/psycles-active-work-NtKLIR \
  --output /var/tmp/psycles-active-work-NtKLIR/work-summary.json
```

For repeat count-only audits use `--skip-images` and a new summary filename.
The full audit checks exact sums, input/build identity, 46-channel finiteness
and all fifteen pass comparisons. Raw log/image hashes and all per-dispatch
records are in work-summary.json. No profiler was attached; logged elapsed
times are not substituted for the earlier rocprof GPU-duration profile.

## Reduced red, permanent oracle and fix

The first unmodified production-stage regression produced 46 failures out of
240 components, retained as `endpoint-red-hip.log`. These are missing hits,
wrong endpoint distances and missing conditional PDFs; continuous roundoff is
not their cause. The initial fixture already used all twelve geometry rows
and five visibility modes. The final oracle appends 64 packed visibility masks
representing 2,048 additional predicates; it does not rewrite the original
sixty endpoint rows.

```bash
/opt/rocm/bin/hipcc -parallel-jobs=32 --offload-arch=gfx1201 \
  -DHIPCC -std=c++20 -O3 -ffast-math \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles \
  tools/cycles_light_endpoint_oracle.hip \
  -o /var/tmp/psycles-active-work-NtKLIR/light-endpoint-oracle-final

set -o pipefail
/var/tmp/psycles-active-work-NtKLIR/light-endpoint-oracle-final \
  | cmp - tests/data/cycles_light_endpoints.txt

cmake --build build --parallel 32
./build/bin/psycles_luisa_light_endpoints_tests hip
```

The first harness build tried to put the inverse transform in KernelLight;
inspection corrected that input setup to original KernelObject before oracle
generation. This was a harness API correction, not a compiler bug or a
production fix. The oracle's final output is byte-identical to the checked-in
fixture, with the final formatted source rebuilt independently.

## Full-resolution follow-up protocol

The canary runner preserves the original equal-pass campaign manifests,
resolution, sampler, integrator limits, frame, source bundle and command line.
Only the output path changes. It renders all four scenes at 256 spp, with
Barbershop repeated three times, checks six binary hashes after every run and
compares to the retained production Cycles 256-spp images. Queue logging is
off during canary timing. Own tests/builds do not overlap these canary renders;
this is still an ordinary desktop, not an exclusively reserved GPU.

```bash
python3 docs/validation/2026-09-08/light-endpoints/run_canaries.py \
  /var/tmp/psycles-matched-pass-hip-2c98Q0 \
  /var/tmp/psycles-active-work-NtKLIR
```

`canaries.json` retains commands, source/output/manifest checksums, scene
compilation, session initialization and render times, frame layout and all
pass errors. Initialization includes setup/baking as well as JIT; it is not
pure compiler time. These are follow-up canaries, not fresh Cycles/Psycles
timing pairs, and their 256-spp image errors must not be compared directly to
the 64-spp work-count diagnostic's errors.

The six completed runs give Barbershop render times 39.2464, 39.2392 and
39.2907 s (median 39.2464). Lone Monk, Monster and Classroom take 13.4148,
14.8794 and 18.0993 s respectively. The report includes every run, not a
selection based on minimum render or initialization time.

Monster's unexpectedly long first link was followed by this extra unchanged
renderer invocation after all six canaries completed:

```bash
env PSYCLES_DISABLE_SHADER_CACHE=1 \
  /home/mike/Projects/Psycles-surface-svm/build/bin/psycles_render_blender_scene \
  /var/tmp/psycles-four-scene-hip-0PRq34/monster/export \
  /var/tmp/psycles-active-work-NtKLIR/monster-warm.ppm \
  hip 1080 1080 256 64 - 540 540 0 0 256 - 1 0 wavefront-staged \
  32 32768 32 1 1 1 4 2 0 0 0 0 1 1048576 \
  > /var/tmp/psycles-active-work-NtKLIR/monster-warm.log 2>&1

python3 docs/validation/2026-09-08/light-endpoints/analyze_warm_control.py \
  /var/tmp/psycles-active-work-NtKLIR \
  --output /var/tmp/psycles-active-work-NtKLIR/monster-warm-summary.json
```

The invocation exited zero. The separate audit verifies frozen binary hashes,
46 finite channels, all fifteen pass comparisons, completed sample range and
unchanged frame. It records both sets of link times and does not merge the
extra repeat into the six-canary campaign. Initialization falls from 62.649
to 22.159 s while render time stays 14.8794 versus 14.8515 s. The unchanged
main-cache switch does not disable every lower-level HIP linker cache; no
cache was purged or forced to attribute the difference causally.

## Validation gates

The all-target build passes with 32 threads. Registered HIP tests pass 181/181
and fallback 183/183, including the former dispatch-film assertion. Host tests
pass 156/157; only the existing source-size check fails, naming
cycles_svm_nodes.cpp (2026), test_cycles_svm_compiler.cpp (2088),
test_luisa_compact_surface_preparation.cpp (2114) and test_luisa_cycles_svm.cpp
(2038), against the unchanged 2000-line limit. Full HIP and fallback suites
overlapped as correctness runs; their elapsed times are not benchmarks. All
tests finished before the 256-spp canary runner started.

Strict native Vulkan verification used:

```bash
env LUISA_VULKAN_USE_XIR=1 LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
  LUISA_VULKAN_DISABLE_DXC=1 LD_DEBUG=libs \
  ./build/bin/psycles_luisa_light_endpoints_tests vk \
  > /var/tmp/psycles-active-work-NtKLIR/endpoint-native-vulkan.log 2>&1
```

Both native SPIR-V modules compiled (5770 and 877 words after optimization),
all 2288 checks passed, and dynamic-loader tracing contains no DXC/DXIL load.
This is a fresh endpoint/visibility gate, not a claim that every Vulkan test
was rerun. No Luisa compiler/backend correction was needed.

Recreate the final checksummed report record after both audits finish:

```bash
python3 docs/validation/2026-09-08/light-endpoints/summarize_evidence.py \
  /var/tmp/psycles-active-work-NtKLIR
```

The script independently reparses exact counts, verifies retained raw source
and binary hashes, checks six canary records, both original-GPU red/green
gates and the separate Monster control. Its output is the checked-in
`evidence-summary.json`; raw EXRs and complete scene logs remain in the
named evidence directories and are not embedded in the repository.
