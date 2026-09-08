# Psycles validation index

Updated 2026-09-08. This is the current evidence index, not a cumulative
roadmap. Older timing tables and legacy-executor claims have been removed
from this page; their dated reports remain under docs/validation and in Git
history.

## Current full-scene baseline

**Pass-workload qualification:** the Cycles setup script added the requested
passes without clearing the source view layer's other passes. Actual EXR
inspection finds extra Mist/Depth/sample-count output in Monk, Depth/sample
count in Monster, AO/Depth/Object Index/sample count in Classroom, and
AO/Depth/sample count in Barbershop. AO enables a separate Cycles shadow
path. Therefore these archived timings are not equal-pass workload
comparisons, and the earlier Classroom speed-lead interpretation is withdrawn.
A corrected pass contract and new matched campaign are required. The raw
timings and common-pass correctness evidence remain available, with this
qualification; do not use the ratios as performance-goal completion evidence.

The [four-scene HIP campaign](docs/validation/2026-09-08/four-scene-hip/README.md)
completed **12 paired runs**, three per original scene, at fixed 256 spp.
Psycles eaa7c72e / Luisa 9ea3b720f is compared against original Blender
Cycles 5.2.1 LTS build 9e2066aef7ef on the RX 9070 XT. This baseline precedes
the subsequent per-ShaderJump-entry specialization change; it must not be
relabeled as a benchmark of later code.

Seconds below are medians. Cycles is its original main-loop wall interval;
Psycles is render-only wall time. These are not whole render-call durations
or summed profiled kernel times.

| Scene | Extent / seed | Cycles HIP | Psycles HIP | Psycles / Cycles |
| --- | --- | ---: | ---: | ---: |
| Lone Monk | 1440x1080 / 0 | 13.4183 | 13.9693 | 1.0411 |
| Monster | 1080x1080 / 0 | 14.4289 | 15.2304 | 1.0555 |
| Classroom | 1920x1080 / 1 | 20.7624 | 18.8207 | 0.9065 |
| Barbershop | 2048x858 / 0 | 28.7312 | 41.0285 | 1.4280 |

Classroom has the lower observed time, but its unequal AO workload prevents
an equivalent-work speedup conclusion. Session-initialization medians
(the CLI's `shader_jit_seconds`, including JIT and setup/baking) are
18.8151 / 22.3434 / 18.1598 / 27.4798 s respectively, with main shader
caching disabled and auxiliary/OS caches retaining normal policy.
Coroutine frames are 220 / 284 / 264 / 416 B. Cycles loads precompiled
kernels; these JIT observations are not a symmetric compiler comparison.

All 46 Psycles channels are finite in all 12 images, and all 15 comparison
passes completed. Combined relative RMSE is approximately
1.241% / 0.548% / 0.353% / 1.080%; DiffInd remains
12.881% / 2.553% / 17.820% / 7.451%. These residuals remain correctness
work, not a noise-based exemption. Original Cycles Classroom contains
25 non-finite DiffDir and 27 non-finite GlossDir pixels in every pair;
affected metrics exclude the union of invalid pixels explicitly.

The campaign report retains exact commands, cache policy, ranges, manifests,
build/device and source/export/output hashes, all pass metrics, and the
location of original-resolution EXRs and reviewed triptychs. Native fast
math is enabled. No profiler or concurrent build/render overlapped these
performance runs. Use the [schema-v2 runner](docs/scene-benchmark.md);
schema-v1 Cycles whole-call ratios are not comparable.

## Published compiler and backend gate

The later [entry-specific specialization](docs/validation/2026-09-08/svm-entry-usage/README.md)
adds independent original-word, AST and GPU regressions. Its completed
gates are full 32-thread build, host 153/154, HIP 177/177, fallback 178/179
and strict native Vulkan 5/5; only the same source-size and fallback-film
failures remain. Twelve subsequent Psycles canaries use the unchanged
original Cycles reference images, not new paired reference renders.
Render medians are 13.9472 / 15.2102 / 18.9568 / 39.5579 s, with unchanged
frames and residual image differences. First-run session initialization
and later repetitions are reported separately, including final-link delays.
The complete paired baseline above retains its original revision and scope.

The [read-only-reference checkpoint](docs/validation/2026-09-08/coro-readonly-forwarding/README.md)
records the complete suites for Psycles 2d89cf1d / Luisa 9ea3b720f. The later
eaa7c72e change corrects benchmark timing/resume contracts, with 22 Python
tests and its focused CTest passing; it does not change renderer code.

| Gate | Result | Qualification |
| --- | --- | --- |
| Full Psycles and Luisa builds | Passed | All 32 hardware threads |
| Psycles HIP | 177/177 | Correctness suite, not performance timing |
| Psycles fallback | 178/179 | Existing sample-dispatch film light_ng.z mismatch |
| Psycles host | 152/153 | Existing four source-size violations |
| Luisa registered non-device-specialized tests | 140/140 | Includes 69 XIR/coroutine tests |
| Strict native Vulkan focused Psycles tests | 5/5 | XIR -> SPIR-V, DXC disabled |
| New read-only coroutine runtime oracle | 1031 assertions per backend | HIP, fallback and strict native Vulkan |
| Existing ordinary coroutine initialization | 22 tests / 306 assertions | HIP and fallback retain default-zero semantics |

The fallback mismatch is expected 0xbf1f8bfd versus actual 0xbf1f8a50.
It has not been hidden by a tolerance change or slow arithmetic. The four
source-size violations are cycles_svm_nodes.cpp, test_cycles_svm_compiler.cpp,
test_luisa_compact_surface_preparation.cpp and test_luisa_cycles_svm.cpp.
Do not describe these full suites as entirely green or relax their limits
to manufacture a pass.

The native Vulkan loader audit records native SPIR-V compilation without
loading DXC/DXIL. A separate initial CUDA-interop loader failure is preserved;
only that local build's CUDA-interop option was disabled. No system package
or toolchain changes were made.

## Native semantic and scheduling evidence

These are scoped checkpoints, not interchangeable full-render certificates.

| Contract | Original-source / regression evidence |
| --- | --- |
| Default native SVM, geometry, curve and light state | [Native default](docs/validation/2026-09-07/native-default/README.md), [surface state](docs/validation/2026-09-07/native-surface-state/README.md) |
| Omitted node cases, feature guards and static array bounds | [Static pruning](docs/validation/2026-09-07/native-static-pruning/README.md), [scene-local extents](docs/validation/2026-09-07/scene-local-extents/README.md), [entry usage](docs/validation/2026-09-08/svm-entry-usage/README.md), [closure budget](docs/validation/2026-09-07/native-closure-budget/README.md) |
| Native volume words, ordered stack and consumers | [Volume SVM](docs/validation/2026-09-07/native-volume-svm/README.md), [volume consumers](docs/validation/2026-09-08/native-volume-consumers/README.md) |
| World/background and camera-dependent baking | [Native background](docs/validation/2026-09-08/native-background/README.md) |
| Scene admission, attribute residency and deferred volume emission | [Native admission](docs/validation/2026-09-08/native-scene-admission/README.md) |
| Assigned-but-failed images | [Missing-image state](docs/validation/2026-09-08/native-missing-image/README.md) |
| Shared surface/volume closure weights | [Original full words and GPU allocator state](docs/validation/2026-09-08/shared-closure-weights/README.md) |
| Local lifetime, generic scheduler policy and subsurface continuation | [Luisa publication](docs/validation/2026-09-07/luisa-local-coro-publication/README.md), [coroutine boundaries](docs/validation/2026-09-07/coroutine-boundaries/README.md), [surface-sort handler](docs/validation/2026-09-07/surface-sort-handler/README.md) |
| Sampler properties and same-path diagnosis | [Sampler contract](docs/validation/2026-09-07/sampler-contract/README.md), [Monk residual](docs/validation/2026-09-07/lone-monk-residual/README.md) |

The repaired Barbershop shared closure matches the first four surface events
and all 45 sampled random fields at the diagnosed pixel. A later NEE event
still selects an adjacent emitter triangle. Monk retains the original
coincident leaf geometry and a visibility divergence. Neither observation
establishes global RNG/path parity or permits deduplication and bit-matching
intersection emulation.

## Reproduction and completion gates

Use the designated worktree and nested Luisa submodule, inspect both Git
states, and preserve unrelated changes. Build with all hardware threads.
Validate HIP first, then fallback; native Vulkan canaries require all three:

```bash
LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
ctest --test-dir build --output-on-failure -j32 -R '<focused-native-Vulkan-tests>'
```

Compiler corrections require formal cause, a minimal failing regression, a
generic fix and validation of the full original module. Expected shading
state comes from version-pinned Cycles source/GPU execution, never a second
host renderer. Do not use profile/prerender observations as local-array
bounds. Separate exact structural contracts from harmless native arithmetic
rounding, and keep fast math enabled.

The requested goal remains open: nine native semantic opcodes, private
legacy displacement removal, indirect/path structural differences and
cross-scene rendering efficiency are not complete. See
[compatibility status](docs/cycles-compatibility.md) for current scope and
[DEVELOP.md](DEVELOP.md) for mandatory implementation/publication rules.
