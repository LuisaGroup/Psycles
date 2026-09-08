# Psycles validation index

Updated 2026-09-08. This is the current evidence index, not a cumulative
roadmap. Older timing tables and legacy-executor claims have been removed
from this page; their dated reports remain under docs/validation and in Git
history.

## Current full-scene baseline

The [equal-pass HIP campaign](docs/validation/2026-09-08/matched-pass-hip/README.md)
completed **12 pairs**, three per original scene, at Psycles c5bf9247 /
Luisa 9ea3b720f and fixed 256 spp. Both engines produce exactly 15 linear
passes / 46 channels, validated from actual EXR headers. Production Cycles is
Blender 5.2.1 LTS build 9e2066aef7ef on the same RX 9070 XT.

Times are medians in seconds: original Cycles' main-loop wall interval versus
Psycles' render-only interval. Neither is an enclosing process timer or a
sum of profiled kernels.

| Scene | Extent / authored seed | Cycles HIP | Psycles HIP | Psycles / Cycles |
| --- | --- | ---: | ---: | ---: |
| Lone Monk | 1440x1080 / 0 | 13.2425 | 13.6826 | 1.0332 |
| Monster | 1080x1080 / 0 | 14.2865 | 15.1328 | 1.0592 |
| Classroom | 1920x1080 / 1 | 18.0319 | 18.7626 | 1.0405 |
| Barbershop | 2048x858 / 0 | 25.3775 | 39.2753 | 1.5476 |

The performance goal is not complete. At this paired checkpoint Barbershop
is 54.8% slower; the other three are 3.3%–5.9% slower. The previous script left
source-only outputs,
including Cycles AO shadow work, enabled. Its unequal-pass timing ratios and
Classroom speed-lead interpretation are superseded, not carried into this
baseline. The [pass-contract regression](docs/validation/2026-09-08/pass-contract/README.md)
and dated old reports preserve that measurement history.

Session-initialization medians (JIT plus setup/baking, not compiler-only)
are 18.7327 / 22.1471 / 18.1196 / 25.6044 s. Main shader caching is disabled;
auxiliary/downstream/OS caches retain normal policy. Frames are
220 / 284 / 264 / 416 B. Native fast math is enabled. No profiler, concurrent
render or build overlapped the pairs. Scene order rotates; Cycles always
runs before Psycles within a pair. Exact commands, hashes, observed ranges,
seed/frame semantics and initialization timings are in the campaign report.

All 46 actual channels are finite in every run, with all 15 comparisons
complete. First-pair Combined relative RMSE is approximately
1.241% / 0.548% / 0.353% / 1.080%; DiffInd remains
12.882% / 2.552% / 17.820% / 7.454%. These are unresolved image differences,
not a sampling-noise exemption or a count of unnecessary paths. Original
Cycles Classroom retains 25 invalid DiffDir and 27 invalid GlossDir pixels;
affected comparisons explicitly exclude the union of invalid pixels.

## Published compiler and backend gate

The earlier
[native descriptor sampler correction](docs/validation/2026-09-08/bound-image-sampler/README.md)
at e2d38fc0. Its six follow-up 256-spp canaries retain finite outputs and the
same frame sizes; Barbershop's new Psycles median is 38.8902 s, 1.0% below
the paired baseline above. These use retained Cycles references, not new
paired timings, and do not close the surface-shading gap. First-use versus
warm downstream HIP link timings are recorded separately.

The follow-up [HIP descriptor audit](docs/validation/2026-09-08/hip-texture-descriptors/README.md)
confirms Luisa already copies native SRDs into GPU memory. Extra inline-layout
and manual grouping experiments are not adopted: fewer indirections did not
give a consistent general improvement. A new 64-spp Barbershop kernel profile
still places the main gap in surface shading (5.6336 s versus retained Cycles
2.9628 s), not volume (0.4289 s versus 0.3536 s). This is instrumented GPU
stage time, not a new paired 256-spp render benchmark.

The latest [light-endpoint correction](docs/validation/2026-09-08/light-endpoints/README.md)
fixes a structural error found using exact queue counts: a geometrically hit
lamp remains a transparent endpoint even when spot/spread evaluation is zero
or its shader excludes an indirect ray class. Original Cycles GPU functions
provide the permanent regression (240 endpoint components and 2048 visibility
predicates). At 64 spp, corrected Barbershop surface/volume visit totals differ
from Cycles by only 196 / 70 out of 332.3M / 85.1M. Surface was already within
0.04% before this fix; an excess of stage visits did not explain the large
surface GPU-time gap. Post-lamp closest routing and a 1.49% shadow-intersection
surplus remain open. These are work-count results, not a new speedup claim.
Its six completed 256-spp canaries retain finite output in all four scenes.
Barbershop's median is 39.2464 s, 0.92% longer than the preceding sampler
checkpoint, with DiffInd relative RMSE improved from 7.45% to 7.13%. Lone
Monk, Monster and Classroom render in 13.4148 / 14.8794 / 18.0993 s. These are
follow-ups against retained references, not fresh paired Cycles timings.
The report retains Monster's first 62.649 s session initialization and its
separate 22.159 s repeat; link-time variation is not a rendering speedup.

The [scheduler trace comparison correction](docs/validation/2026-09-08/dispatch-trace-comparison/README.md)
fixes the outstanding fallback assertion without changing renderer binaries
or any captured trace bits. Continuous intermediate values have a separate
1e-4 bound; RNG/discrete state/written lanes are now exact gates. Film's 2e-5
bound and serial bit-exact chunking remain unchanged. The earlier
[entry specialization](docs/validation/2026-09-08/svm-entry-usage/README.md) and
[generic read-only coroutine correction](docs/validation/2026-09-08/coro-readonly-forwarding/README.md)
remain covered by the complete suites.

| Gate | Result | Qualification |
| --- | --- | --- |
| Full build | Passed | All 32 hardware threads |
| Psycles HIP | 181/181 | Registered suite, including new endpoint/visibility regression |
| Psycles fallback | 183/183 | Includes repaired semantic dispatch trace comparison |
| Psycles host | 156/157 | Existing four source-size violations; comparator has 11,016 checks |
| Strict native Vulkan endpoint gate | 2288 checks | Two native SPIR-V modules; no DXC/DXIL load |
| Benchmark protocol focused host gate | 6/6 | Actual Blender pass reset, header/resume and comparator tests |

The former fallback mismatch was expected 0xbf1f8bfd versus actual 0xbf1f8a50
in a spherical-light intermediate normal. The full captured traces establish
unchanged discrete/RNG state and small continuous roundoff; no slow arithmetic
or renderer fix was introduced. The remaining source-size failures are
cycles_svm_nodes.cpp, test_cycles_svm_compiler.cpp,
test_luisa_compact_surface_preparation.cpp and test_luisa_cycles_svm.cpp.
Do not call these full suites entirely green or relax their limits.
The earlier five-test native Vulkan image gate remains revision-pinned in the
descriptor audit; it is not represented as a newly rerun image suite here.

At the published Luisa 9ea3b720f checkpoint, 140/140 registered
non-device-specialized tests passed, including 69 XIR/coroutine tests.
The read-only coroutine runtime oracle passes 1031 assertions on HIP,
fallback and strict native Vulkan; 22 ordinary-initialization tests /
306 assertions preserve default-zero semantics. These are revision-pinned
child results, not an assertion that a new child suite ran with every report.

The strict Vulkan loader audit uses native XIR -> SPIR-V with all three
guards and records no loaded DXC/DXIL library. No system package/toolchain
changes or inlining-policy changes were made for these checkpoints.

## Native semantic and scheduling evidence

These are scoped checkpoints, not interchangeable full-render certificates.

| Contract | Original-source / regression evidence |
| --- | --- |
| Default native SVM, geometry, curve and light state | [Native default](docs/validation/2026-09-07/native-default/README.md), [surface state](docs/validation/2026-09-07/native-surface-state/README.md) |
| Omitted node cases, feature guards and static array bounds | [Static pruning](docs/validation/2026-09-07/native-static-pruning/README.md), [scene-local extents](docs/validation/2026-09-07/scene-local-extents/README.md), [entry usage](docs/validation/2026-09-08/svm-entry-usage/README.md), [closure budget](docs/validation/2026-09-07/native-closure-budget/README.md) |
| Native volume words, ordered stack and consumers | [Volume SVM](docs/validation/2026-09-07/native-volume-svm/README.md), [volume consumers](docs/validation/2026-09-08/native-volume-consumers/README.md) |
| World/background and camera-dependent baking | [Native background](docs/validation/2026-09-08/native-background/README.md) |
| Scene admission, attribute residency and deferred volume emission | [Native admission](docs/validation/2026-09-08/native-scene-admission/README.md) |
| ImageManager sampler descriptors and assigned-but-failed images | [Native bound samplers](docs/validation/2026-09-08/bound-image-sampler/README.md), [missing-image state](docs/validation/2026-09-08/native-missing-image/README.md) |
| Shared surface/volume closure weights | [Original full words and GPU allocator state](docs/validation/2026-09-08/shared-closure-weights/README.md) |
| Surface NEE work eligibility | [BSDF-gated random/selection work, full backend suites and four-scene canaries](docs/validation/2026-09-08/surface-nee-work/README.md) |
| Homogeneous volume work eligibility | [Original rejection/technique branches and the Barbershop kernel diagnosis](docs/validation/2026-09-08/volume-work/README.md) |
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
