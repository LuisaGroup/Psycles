# Psycles validation index

Updated 2026-09-09. This is the current evidence index, not a cumulative
roadmap. Older timing tables and legacy-executor claims have been removed
from this page; their dated reports remain under docs/validation and in Git
history.

## Revision-pinned paired full-scene baseline

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

## Latest four-scene follow-up

Psycles `239cade6` / Luisa `85e5300f1` complete six further 256-spp renders,
using fresh socket metadata with the exact earlier geometry/texture bytes.
These use retained equal-pass Cycles references, **not fresh timing pairs**.
All 46 channels are finite and all 15 pass comparisons complete. Exact six-run
data, images' provenance and all implementation hashes are in the
[latest report](docs/validation/2026-09-08/svm-group-contexts/README.md).

| Scene | Latest render seconds | Session init seconds | Current frame |
| --- | ---: | ---: | ---: |
| Lone Monk, one run | 13.3748 | 19.2910 | 220 B |
| Monster, one run | 14.7691 | 22.4903 | 280 B |
| Classroom, one run | 18.2831 | 19.4370 | 260 B |
| Barbershop, three-run median | 40.1352 | 25.9988 / 26.5395 / 25.9027 | 416 B |

Barbershop's individual times are 40.1352 / 40.0687 / 40.9465 s. Its median
is 0.29% below the preceding 40.2529 s, not an isolated causal improvement
estimate, and 58.2% slower than the retained Cycles median. Initialization is
JIT plus setup, not compiler-only or cold JIT: the separate profile had warmed
downstream caches before these six runs. It is never included in render time.
Current first-run DiffInd relative RMSE is 12.882% / 2.552% / 17.820% / 7.126%
in the table's scene order. The structural and efficiency goals remain open.

## Current compiler and backend gate

The [group-context repair](docs/validation/2026-09-08/svm-group-contexts/README.md)
at Psycles `239cade6` / Luisa `85e5300f1` is the latest structural checkpoint
and the source of the complete 256-spp four-scene campaign above.
The preceding lamp-routing report records Psycles `773f1aca` / Luisa
`4284e8cb9`, which publish the
post-lamp closest-intersection boundary, native miss-distance normalization
and generic loop-epoch CFG repair. The original 1,685-block native module
now restructures; the 11-block reduced input, transactional rejection case
and remaining formal proof obligations are retained in the child audit.
Post-lamp closest routing is no longer listed as an unfixed discrepancy.
The roughly 1.49% shadow-intersection surplus and residual DiffInd remain.

The main SVM interpreter and most handlers are already inlined. Retaining
more HIP callable boundaries in a full-scene A/B/A control made Barbershop
surface GPU time 35.5% slower and render wall 20% slower, so that experiment
is completely reverted. Static spill counts are not dynamic spill traffic;
the report identifies actual stage symbols and outlined Cycles callees.
No inlining policy, register limit or shader-specific backend option changed.

A separate [ordinary microfacet callable A/B/A control](docs/validation/2026-09-08/microfacet-boundaries/README.md)
confirms that the current microfacet bodies are already inlined in final ISA.
Adding source callable boundaries grows fixed private storage from 2,496 to
94,896 bytes and surface time from 5.90 to 38.30 seconds; restored A returns
to 5.896 seconds with an identical code object. This experiment is also
fully reverted and does not change the production benchmark checkpoint.

The earlier hidden-input, Vector Math, bump-edge/domain and procedural-output
repairs have 105 original-Cycles material images. Fifty-three additional
exact images now constrain native texture socket types, conversion identity,
closure input declaration order and unavailable Voronoi defaults. Another
34 original images constrain linked/primitive forwarding through groups,
reroutes and muted links, plus separate Blender luma/Gamma folding. Of these,
33 are raw-exact and one differs only in three-ULP typed float literals; no
device arithmetic or expected words change. Eighteen further exact original
images constrain Math clamp expansion after native conversion links;
fifteen fail before the repair. Another eighteen exact images constrain
Mapping's native POINT declarations and shared conversions; five fail before
that repair, including a five-authored-node 47-versus-44-word reduction.
Thirty further exact original images constrain lazy group input evaluation,
persistent per-instance output caches and parent-context forwarding. Twelve
fail before the repair, including a five-node 25-word chain incorrectly
collapsed to 22 words. This removes the last six known Barbershop schedules:
all 279 used-shader images have identical node/payload layouts, stack
addresses and jump/domain structure; 115 are raw-exact and 164 differ only
in declared resource-ID fields.
Resource binding equivalence is unresolved; no words are normalized away.
Static stack capacity remains 33 floats; the surface ELF `.text` remains
byte-identical and resources stay at 256 VGPRs / 2496 private bytes.
The new 64-spp surface GPU total is 5.872749 s, versus the
retained original Cycles 2.962829 s, with essentially unchanged surface visits.
These structural counts are not a measured speedup or complete shader parity.
Host-only Blender folding domains remain separate from later Cycles folding;
no slow bit-matching device arithmetic is added.

The full parallel fallback suite additionally exposed a real worker-pool
lost-wakeup race. Luisa `85e5300f1`, published to `origin/next`, fixes both
completion and shutdown predicate publication under their mutex. Two minimal
production-queue counterexamples fail before the fix and pass afterwards,
with 100 repetitions each. The failed 183/184 run is retained, not replaced
by an earlier successful result; its final full rerun passes 184/184.

| Gate | Result | Qualification |
| --- | --- | --- |
| Full build | Passed | All 32 hardware threads |
| Psycles HIP | 182/182 | Complete registered suite, 145.30 s |
| Psycles fallback | 184/184 | Complete parallel suite, 91.40 s |
| Psycles host | 168/168 | 258 original-Cycles images across the new families; source-size gate fully green |
| Luisa child | 155/155 | Retained unchanged child checkpoint; both queue races also repeated 100 times |
| Strict native Vulkan | 2/2 | Lamp routing and bump state; 31 native SPIR-V compilations, no DXC/DXIL load |
| Benchmark protocol focused host gate | 6/6 | Actual Blender pass reset, header/resume and comparator tests |

The prior [dispatch-trace comparator fix](docs/validation/2026-09-08/dispatch-trace-comparison/README.md)
remains test-only: discrete/RNG state is exact, continuous intermediates have
a separate 1e-4 bound, and film's 2e-5 bound is unchanged. It is distinct from
the newly fixed queue race. The four previous source-size violations are
resolved by cohesive ConvertNode, color-test, AST-visitor and fixture-builder
modules, with all previous test bodies/assertions retained. No size limit is
relaxed and no exception is added.
The earlier five-test native Vulkan image gate remains revision-pinned in the
descriptor audit; it is not represented as a newly rerun image suite here.

Earlier [descriptor sampling](docs/validation/2026-09-08/bound-image-sampler/README.md),
[native SRD investigation](docs/validation/2026-09-08/hip-texture-descriptors/README.md),
[light endpoints](docs/validation/2026-09-08/light-endpoints/README.md), and
[read-only coroutine state](docs/validation/2026-09-08/coro-readonly-forwarding/README.md)
retain their revision-pinned measurements and full evidence. They are not
substituted for current timings. Ordinary scalar/vector initialization still
uses default-zero semantics.

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
