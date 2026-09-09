# Psycles validation index

Updated 2026-09-10. This is the current evidence index, not a cumulative
roadmap. Older timing tables and legacy-executor claims have been removed
from this page; their dated reports remain under docs/validation and in Git
history.

## Apple Metal / Metal4 checkpoint

These observations pin the source/SDK snapshots in their report; they do not
validate subsequent HIP continuation changes described below.

The [Lone Monk Metal campaign](docs/validation/2026-09-09/matched-pass-metal/README.md)
records published Luisa backend repairs, focused regressions and complete
1920x1080 original-scene canaries on both backends. The staged main frame is
55 fields / 220 B; shadow queues are separate. The first 256-sample staged
matrix completes: Cycles Metal 50.4312 s, Psycles Metal 350.511 s and Metal4
105.569 s (render-only). These are single observations with different
unchanged backend batching policies, not repeated performance conclusions.
All 46 channels are finite and all 15 pass comparisons complete. The original
graph attempt failed generic XIR restructuring. Luisa `afdc139ec` repairs it
with a reduced counterexample, 97 CFG tests / 2740 assertions and both original
scene gates. Luisa `03a0f5158` also repairs swizzle reference lowering; complete
Metal / Metal4 codegen fixtures pass, and both graph scene gates pass again.
Graph uses 90 fields / 456 B with inline shadow state. The fresh repeated
256-sample matrix uses this published SDK and upstream `287bc520` / film
implementation `9e3ba165`, integrated before restarting measurement. There
are no local Psycles implementation changes.

The first film-aligned graph pair completes, but **Metal4 output is invalid**:
Combined relative RMSE is 0.2304 and mean luminance is 0.7757 of Cycles, versus
0.007605 Combined relative RMSE on Metal. Observed render times are 264.027 s
(Metal) and 291.785 s (invalid Metal4 output), with Cycles 50.7783 s. These are
single observations, not a valid Metal4 performance score. Both current graph
frames have 91 fields / 456 B. The film-aligned staged control also completes:
Metal 424.505 s, Metal4 133.884 s, Cycles Metal 50.8303 s. Metal4 staged Combined
relative RMSE is 0.007759739, without graph's coherent darkening. All six
first-round EXRs have 46 finite channels; this does not waive graph's numerical
failure. The exact Metal4 graph repeat fails again (294.441 s, luminance ratio
0.783365179); it is also excluded from performance scores. All eight EXRs
across these three matrices have 46 finite channels. A reduced same-sample
replay fails when sorting and tail are enabled together, while either option
alone passes the reduced check. The reduction now identifies generic Luisa
split code read-modifying an undefined packed-Boolean word at entry. The
correction is published as Luisa `6e58928d8`; permanent regressions pass host XIR suites and Metal/Metal4
scheduler suites; the original reduced replay passes without disabling
sorting/tail or clearing the pool. The full 1080p/256 Cycles gate now passes:
Combined relative RMSE 0.007764527, luminance ratio 0.999891523, all 46 channels
finite; 328.193 s versus Cycles Metal 50.778 s (one correctness-gate observation).
See the full report for retained failed evidence, batching caveats and source
identities.

## Whole-surface structural audit and native repairs

The [whole-surface audit](docs/validation/2026-09-09/surface-semantic-audit/README.md)
separates interpreter coverage from production consumers. Seven fresh tiny
original-Cycles-HIP/production-HIP pairs exposed **three failures**: two pure
volume slabs exhaust an unrelated outer path budget; selected Ray Portal
closures never perform portal continuation; packed object holdout is not
applied before emission. Four controls agree, including increasing only the
unused transparent budget to recover the volume scene. Both captured replay
gate families exit 2. The [native lifetime repair](docs/validation/2026-09-09/native-path-lifetime/README.md)
now removes that aggregate path budget and its ABI member. A permanent
original-HIP full-render regression first fails both schedulers on the two-slab
case, then passes all six case/scheduler combinations after correction.
The [native continuation repair](docs/validation/2026-09-09/native-surface-continuation/README.md)
also restores selected Portal continuation and persistent portal depth, with
native labels consumed directly. Its original-GPU state and complete-render
fixtures check the production boundary, not only opcode population. The
[Holdout repair](docs/validation/2026-09-09/native-holdout/README.md) restores the
object/node pre-emission boundary, with original word, GPU-state and complete-film
regressions. The captured audit stays a historical red baseline, not a claim
that all three remain unfixed.

The 110-tag catalog has 100 implemented handlers, eight missing semantic
opcodes and two sentinels; this is not 100 proved end-to-end features. Missing
transparent-glass controls, data-pass predicates,
shadow-catcher integration and private displacement dependencies remain.
Work-placement/ownership leads are recorded separately from measured
GPU costs. The continuation repair moves first-bounce weights, transparent-ray
work and the surface-BSDF tuple to their native predicates. The audited
Barbershop word identity is unchanged by these runtime-only edits. The
portal/holdout defects are inactive in the benchmark scenes; path-budget
exhaustion there is not demonstrated. The performance and finite-value
limitations below remain open; a same-binary Monk repeat also exposes local
run-to-run image variation, without an established cause.

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

## Historical four-scene HIP follow-up (SDK 6e)

The native-Holdout repair on Luisa `6e58928d8` completes six 256-spp renders with the exact
earlier geometry/image bytes and refreshed socket metadata. These use
retained equal-pass Cycles references, **not fresh timing pairs**. Five
renders have all 46 channels finite; Classroom's separated-pass finite gate
fails and remains explicitly recorded. All 15 comparisons complete on their
reported finite domains. See the [native Holdout report](docs/validation/2026-09-09/native-holdout/README.md).

| Scene | Frozen render seconds | Session init seconds | Frozen frame |
| --- | ---: | ---: | ---: |
| Lone Monk, one run | 12.4505 | 43.9887 | 200 B |
| Monster, one run | 13.7470 | 52.6500 | 272 B |
| Classroom, one run | 17.2434 | 38.3884 | 252 B |
| Barbershop, three-run median | 37.8379 | 58.1651 / 19.1775 / 20.5389 | 416 B |

Barbershop's observations are 37.8379 / 37.8313 / 37.8524 s; none is
discarded. The median remains 49.1% above retained Cycles and is 0.91% above
the previous S2 median. Earlier campaigns remain in their dated reports, not
mixed into this median. These temporal changes are not isolated causal
estimates or a demonstrated Holdout performance effect. Initialization
includes JIT, setup and baking, excludes render time, and is not a matched
cold-JIT comparison. Main shader caching is disabled; downstream/OS caches
retain ordinary policy. The main HIPRTC link takes 27,000.95 ms initially
and 92.18 / 108.99 ms on repeats, producing an 862,656-byte code object;
that localizes the observed first/repeat
delay without independently proving a specific cache mechanism.

Classroom reproduces 18 invalid channel values at eight pixels with the same
NaN/+Inf/-Inf masks as the prior capture: actual DiffDir/GlossDir have seven invalid pixels each, and
DiffInd/GlossInd two each. Combined is finite. Original DiffDir/GlossDir
have 25/27 invalid pixels at different coordinates; comparisons exclude
and report the union. A dynamic-input original GPU/actual DSL observer
reproduces the same tiny-BSDF fastmath NaN/Inf classes with matching
zero/normal controls. This supports the shared arithmetic boundary, not a
proof of every affected path or an all-finite pass. No epsilon/slow-math
workaround or numeric waiver is introduced; the keep-going runner exits 2.
First-run DiffInd relative RMSE remains 12.882% / 2.553% / 17.820% / 7.126%
in the table's scene order. The earlier S1 SDK isolation control and its
same-binary repeat exposed local Monk changes, including Normal. The frozen
S2-to-Holdout comparison also retains local changes: Combined RMSE 5.49e-4
and Normal RMSE 1.47e-4, with maximum errors 0.283 and 0.0535. Their cause is
unresolved; the changes are not attributed to Holdout or path-bound
exhaustion, or waived as floating-point noise. Correctness,
repeatability and efficiency goals remain open.

## Current generic shading-pruning checkpoint

The [bounded native shading-terminator repair](docs/validation/2026-09-10/shading-terminator-pruning/README.md)
uses ordinary C++ recording guards only when the complete finalized object
image proves every native `frequency > 1` predicate false. Mixed, enabled or
unknown images retain the native device path; no scene names or profiles are
used. An original six-row HIP oracle also exposes the pre-existing standard
ACOS/native fast-ACOS mismatch. Reusing the existing native fast helper at
that one site passes the oracle without introducing a precision workaround.

The final `86fe7274…` runtime, on SDK `5c7de2bb9` (unchanged 98f production
libraries), passes full host **186/186**, HIP **193/193**, fallback **195/195**
and focused strict native Vulkan **4/4**. The Vulkan trace has six successful
native SPIR-V compilations, all three route guards, and no DXC/DXIL load.
Full builds use all 32 threads. These suite durations are correctness evidence,
not performance measurements. The separate new coroutine reachability candidate
is not included in these binaries.

The [fresh four-scene Phase A capture](docs/validation/2026-09-10/shading-terminator-pruning/PHASE_A.md)
completes four renders and sixty pass comparisons at native extents and 256
fixed samples, main shader cache off and native fast math on. All 32 source
pins, eight binaries, four input/reference sets and 310 loaded texture entries
remain unchanged across execution.

| Scene / extent | Render seconds | Session/JIT seconds | Frame | DiffInd relative RMSE |
| --- | ---: | ---: | ---: | ---: |
| Barbershop / 2048x858 | 37.1113 | 55.0679 | 416 B | 7.12617% |
| Lone Monk / 1440x1080 | 12.7550 | 42.9222 | 200 B | 12.88149% |
| Monster / 1080x1080 | 13.6912 | 55.2632 | 272 B | 2.55253% |
| Classroom / 1920x1080 | 17.5279 | 40.2551 | 252 B | 17.82034% |

These are isolated single observations against retained original Cycles HIP
images, **not fresh timing pairs or a demonstrated speedup**. Session/JIT
includes compilation, allocation and upload/setup, not compiler-only time.
All 46 channels are finite for the first three actual/reference pairs.
Classroom still fails: **18 actual non-finite lanes / eight pixels**, with
70 reference lanes / 27 pixels. Combined is finite; separated-pass metrics
report the valid domain and do not waive the failure. DiffInd and previously
observed localized Monk variation remain unresolved. Fresh current-exporter
bundles, their input/word-identity audit, and twelve new v3 pairs have **not run**.

## Preceding SDK integration: two-scene canaries and joint queues

The [SDK 98f integration report](docs/validation/2026-09-09/next-integration/README.md)
records fresh 256-spp HIP canaries with native Holdout and explicit compatible
surface-Handler batching. Barbershop at 2048x858 renders in 36.6909 s with
50.0167 s session setup/JIT and a 416 B frame; Monk at 1440x1080 takes
12.5366 s, 31.8320 s and 200 B. Both have all 46 channels finite and all 15
pass comparisons complete. These are single observations against retained
Cycles references, not a fresh four-scene paired benchmark or queue-only A/B.
Barbershop remains 44.58% above the retained Cycles median; DiffInd relative
RMSE remains 7.1262% / 12.8816% for Barbershop / Monk.

The complete 64-spp diagnostic has exactly 112,459,776 generated paths in
both versions. Joint surface dispatches fall from 1,062 to 798, while surface
entries change only from 332,307,665 to 332,307,666. This is fewer scheduling
batches, not proof of less shading, smaller frames or eliminated indirect-light
differences. The [frozen frame audit](docs/validation/2026-09-09/barbershop-coroutine-audit/README.md)
retains exact old-SDK field-I/O evidence. Classroom's invalid separated passes,
Monk's localized repeat variation and complete cross-scene efficiency remain
open. That preceding capture does not include Monster/Classroom; the current
four-scene observations above do not add fresh timing pairs either.

## Preceding compiler and backend gates

The [film destination/state repair](docs/validation/2026-09-09/film-routing/README.md)
restores native first-shadow classification, captured flags/weights, and
direct/indirect address selection. It removes the six-RGB splitter and
unconditional writes to both destinations. Original GPU film checks improve
351/540 -> 540/540, shadow state 72/180 -> 180/180, and production AST
destination checks 0/5 -> 5/5. The zero-only BSDF ratio guard matches Cycles;
native tiny-value limitations are separately exposed above.

The 64-spp full-scene A/B/B/A has surface medians 5.536031 -> 5.432208 s
(-1.88%) and render medians 10.17955 -> 10.08000 s (-0.98%), with only two
observations per treatment and identical restored A binaries. This is not
isolated atomic-cost attribution. Main instruction sites decrease
158,005 -> 157,717. Both retain 47 calls, 256 VGPRs, 107 SGPRs, 2,464 private
bytes and a 416-byte frame. Surface visits are essentially unchanged;
no reduction in unnecessary paths or residual DiffInd repair is claimed.
Barbershop has emission-only fog and zero volume bounces, so corrected
mixed volume-scattering film routes cannot explain its DiffInd.

Those film-routing suites and timings used Luisa `da8fff856`. The subsequent upstream
Metal/CI integration to `8911828eb` leaves all six measured HIP binaries
byte-identical after an all-target build; complete host and focused HIP,
fallback and strict native Vulkan gates are rerun before publication.
The Metal reports and their published gitlink are preserved. The preceding
integration below uses Luisa `98f4667ca`: upstream shared-callable changes,
the generic prepare-to-update bypass repair and logical-priority/joint-Handler
batching. It includes native Holdout and the renderer's explicit batching
opt-in; the new reachability candidate is not part of these measured binaries.

| Gate | Result | Scope |
| --- | --- | --- |
| Full build | Passed | All 32 hardware threads |
| Psycles host | 183/183 | Complete host gate, 32 workers, 8.30 s |
| Psycles HIP | 192/192 | Complete suite, 540.44 s |
| Psycles fallback | 194/194 | Complete suite, 549.07 s |
| Strict native Vulkan | 12/12 | 726.28 s; 456 native compilations, no DXC/DXIL loader matches |
| Luisa scheduler native Vulkan | 17/17 | 163 cases / 21,349 assertions, 20.55 s; strict native route, no DXC/DXIL |
| Then-captured full-resolution scene finiteness | 2/2 | Barbershop and Monk only; current four-scene finite gate above is 3/4 |

Exact terminal logs, native guard/loader checks and hashes are in the
[integration report](docs/validation/2026-09-09/next-integration/README.md).
The frozen `6e58928d8` Holdout checkpoint remains **7/11 failed** on strict
Vulkan: `lamp_routing`, `zero_bsdf`, `holdout_render` and `ray_portal_render`
aborted after CFG output-verifier failures, CTest exit 8 / 84.28 s. These
historical failures are preserved, not relabeled as passing. The 98f gate
passes all four tests. Upstream `1a2c3ea922c2` already repaired the complete
original zero-BSDF module; the additional local CFG repair has a separate
prepare/update counterexample. A common old diagnostic alone was not its
root-cause proof. The historical six-canary finite gate remains 5/6 because
Classroom is invalid; the successful new backend gates do not waive it.

The unrelated native Vulkan area-sample gate remains 5/6 with 36 failing
numerical lanes reproduced in its pre-repair implementation. Additional
native f16/f64 `OpFRem` coverage retains 1,076 failures; float32 passes
1,744 assertions and HIP/fallback pass all 5,232. These are not waived by
the focused green canary or hidden with slower HIP math.

### What the structural/code-generation audit establishes

The [279-shader binding audit](docs/validation/2026-09-09/resource-identities/README.md)
finds identical node/payload layouts, stack addresses and jumps. All 600
image and 795 attribute references resolve to the same observed resource
identities. No words are normalized away. This proves layout/binding
identity, not decoded texel values or complete shader/path parity.

The [static function/texture audit](docs/validation/2026-09-09/surface-static-audit/README.md)
finds main SVM, 3D Noise and microfacet bodies already inlined; 169 byte
textures are not expanded to float32. The native observed function-address
closure is not a proven dynamic call graph. Static instructions/spills do
not establish executed costs. Full-scene experiments retaining extra
callable boundaries or changing surface groups to 1024 were slower and
are completely reverted. There is no manual inlining policy, register cap
or precision workaround in production.

| Completed investigation | Revision-pinned evidence |
| --- | --- |
| Generic CFG loop epochs and original-module restructuring | [Loop-scope repair](third_party/LuisaCompute/docs/validation/2026-09-08/loop-scope-restructure/README.md) |
| Shared-case representation and closure input/dispatch structure | [Shared labels](docs/validation/2026-09-09/shared-switch-cases/README.md), [closure inputs](docs/validation/2026-09-09/closure-input-guards/README.md) |
| Reverted launch/call-boundary interventions | [Launch geometry](docs/validation/2026-09-09/surface-launch-geometry/README.md), [microfacet boundaries](docs/validation/2026-09-08/microfacet-boundaries/README.md) |
| Native Noise/Voronoi structure | [Noise](docs/validation/2026-09-09/noise-codegen/README.md), [Voronoi](docs/validation/2026-09-09/voronoi-octave/README.md) |
| Scene-owned area/spot parameters | [Light parameters](docs/validation/2026-09-09/scene-light-parameters/README.md) |
| Native emission eligibility | [Emission](docs/validation/2026-09-09/surface-emission/README.md) |
| Group normalization and native resource bindings | [Group contexts](docs/validation/2026-09-08/svm-group-contexts/README.md), [bindings](docs/validation/2026-09-09/resource-identities/README.md) |
| HIP sampler descriptors and native SRDs | [Bound samplers](docs/validation/2026-09-08/bound-image-sampler/README.md), [SRDs](docs/validation/2026-09-08/hip-texture-descriptors/README.md) |

The ongoing review compares complete surface-path structure and ownership.
Per-ray lamp inverse reconstruction, eager geometry/shadow-terminator work
and pre-termination data-pass reductions remain leads, not an exhaustive
list or quantified dominant costs. First-bounce weight placement was corrected
by native continuation. Surface already omits unused node cases and diagnostic
status lanes and shares populated closures across NEE and continuation.
The roughly 1.49% shadow-intersection surplus and indirect/path differences
remain unresolved. Ordinary scalar/vector initialization still defaults to
zero; only explicitly lifetime-only Local storage omits initialization.

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
ctest --test-dir build --output-on-failure --parallel 1 -R '<focused-native-Vulkan-tests>'
```

Compiler corrections require formal cause, a minimal failing regression, a
generic fix and validation of the full original module. Expected shading
state comes from version-pinned Cycles source/GPU execution, never a second
host renderer. Do not use profile/prerender observations as local-array
bounds. Separate exact structural contracts from harmless native arithmetic
rounding, and keep fast math enabled.

The requested goal remains open: eight native semantic opcodes, private
legacy displacement removal, indirect/path structural differences and
cross-scene rendering efficiency are not complete. See
[compatibility status](docs/cycles-compatibility.md) for current scope and
[DEVELOP.md](DEVELOP.md) for mandatory implementation/publication rules.
