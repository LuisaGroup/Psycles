# Cycles compatibility status

Updated 2026-09-10. This page describes the current implementation, not the
performance or feature coverage of the legacy material executor.

## Default execution path

Scene compilation selects the native Cycles 5.2.1 SVM surface interpreter.
No environment opt-in is required. `PSYCLES_NATIVE_CYCLES_SVM_SURFACE=0`
does not restore the old route. This applies to megakernel and staged
wavefront rendering, including static triangles and admitted ribbon curves.

The implementation preserves Cycles' node word stream, typed payloads,
stack offsets, program counter, dispatch, closure state, feature masks,
surface/volume/displacement shader jumps, and bump-state transitions.
Blender/Cycles itself supplies the compiler and GPU-state oracles. There is
no independent CPU reference renderer.

Preserving this interpreter model is not end-to-end feature parity. The
[whole-surface audit](validation/2026-09-09/surface-semantic-audit/README.md)
reproduced three failures in its captured production HIP against original
Cycles HIP: pure-volume boundaries could exhaust the outer path budget;
Ray Portal had population but no selected-closure continuation; object
holdout was packed but not applied before emission. Seven tiny scene pairs
had four agreeing controls and three failures; both captured replay gates exit 2.
The [native lifetime correction](validation/2026-09-09/native-path-lifetime/README.md)
now removes the aggregate path budget and ABI member, leaving termination
to native event transitions and independent counters. Its permanent full-render
original-HIP regression passes both megakernel and staged schedulers. The
[native continuation correction](validation/2026-09-09/native-surface-continuation/README.md)
also restores selected Portal continuation and persistent portal depth, while
ordinary native continuation consumes original labels without a SurfaceEvent
roundtrip. The [Holdout correction](validation/2026-09-09/native-holdout/README.md)
restores object/node Holdout before emission, including the original allocation
and flag write sets. Transparent-glass settings and data-pass
predicates also retain integration gaps. These findings are not repaired by the default
switch or by 100 implemented opcode handlers.

The default switch is **not** completion of the legacy code removal:
the displacement prepass still has SurfaceProgram consumers. Its private
compilation input now contains only displacement plus an inert required root;
ordinary surface, volume, light and world admission no longer depends on it.
Native KernelShader metadata supplies all material bindings, while native
attribute requests supply geometry residency. Background, stacked volume,
shadow-volume, collision and majorant evaluation use native SVM. Classroom's
Object Index admission blocker is fixed. Remaining legacy dependencies are
removal work, not supported alternate SVM architectures. The
[default-path checkpoint](validation/2026-09-07/native-default/README.md)
records the actual deletions and remaining dependencies.

## Static specialization and local storage

After scene loading, compiler metadata bounds the JIT program. Neither
profiling, pre-rendering nor scene-name constants determine allocation sizes.

- Unused opcode cases are omitted at host recording time. The same Cycles
  node/scene feature masks also omit unreachable handler bodies and closure
  consumers. This includes BSSRDF exit setup on scenes without subsurface.
- Stack capacity comes from the native compiler's stack-address analysis.
  Opcode usage and stack high-water marks are tracked per ShaderJump entry;
  bump falls through into surface and shares its maximum allocation bound.
  Linking unions corresponding entries; unproven external images retain a
  conservative whole-image bound.
  The recorded array extent is passed to main surface, light, background,
  importance-bake, volume, density-bake and shadow SVM entries. The standalone diagnostic API
  retains a conservative default.
- Closure capacity follows the finalized Cycles graph count and scene cap,
  not the old SurfaceProgram estimator.
- Local scratch is distinct from persistent coroutine frame storage. Generic
  Luisa Local lifetime and coroutine extension/handler mechanisms express
  storage lifetime and scheduling. Ordinary scalar/vector initialization
  remains zero.
- Fallback's separate LLVM barrier frames use a CPU-worker-owned arena with
  reusable, pointer-stable overflow chunks. Its 4 MiB fast buffer is not a
  frame-size limit; capacity reuse avoids per-lane/per-dispatch malloc/free.
- Native fast math remains enabled. No forced noinline boundary or slow
  software floating-point/intersection path is used for bitwise alignment.

Proofs, counterexamples and regression boundaries:

- [Static opcode and feature pruning](validation/2026-09-07/native-static-pruning/README.md)
- [Native closure budget](validation/2026-09-07/native-closure-budget/README.md)
- [Scene-local stack extents](validation/2026-09-07/scene-local-extents/README.md)
- [Per-entry opcode and stack specialization](validation/2026-09-08/svm-entry-usage/README.md)
- [Published Luisa Local/coroutine integration](validation/2026-09-07/luisa-local-coro-publication/README.md)
- [Coroutine boundary audit and SSS queue correction](validation/2026-09-07/coroutine-boundaries/README.md)
- [Transitive read-only references and uniform frame state](validation/2026-09-08/coro-readonly-forwarding/README.md)

The latter reports are dated checkpoints; their isolated-snapshot or
then-unpublished qualifications describe those runs, not a second current
production route.

## Native SVM coverage and outstanding work

The AST regression checks 100 implemented handlers in Cycles' 110-tag
inventory. Eight semantic opcodes remain unimplemented: Radial
Tiling, Bevel, Ambient Occlusion, Raycast, AOV Start/Color/Value and Scene
Time. NONE and PAD1 are not executable shading handlers. A material using
an unsupported reachable operation must not silently fall back to the
legacy executor or substitute a socket default.

The current audit distinguishes missing handlers from admitted-but-incomplete
consumers, explicit unsupported scene domains, and work-placement differences.
The portal opcode fixture alone does not exercise a portal bounce; a separate
26-state original-GPU fixture and eleven complete-render scenes now cover that
consumer. The object-holdout failure uses only an ordinary emission node and
is now covered by the separate Holdout repair: twelve complete scenes, 36
original-GPU states and fifteen exact surface/volume word images. Neither
original failure depends on an unsupported reachable opcode. See the linked audit for source/field
ownership, all 110 opcode entries, the fresh four-scene program census, and
the original-HIP counterexamples. They supersede any inference of full surface
parity from node-test counts, not the scoped historical test results below.

Recent independently checked native families include:

| Area | Evidence |
| --- | --- |
| Surface allocation, closure setup/evaluation/sampling, BSSRDF exit and state flags | [Surface state](validation/2026-09-07/native-surface-state/README.md), [zero-BSDF state](validation/2026-09-07/zero-bsdf/README.md) |
| Selected Portal continuation, native labels and portal-depth shader consumers | [Native continuation and original-GPU fixtures](validation/2026-09-09/native-surface-continuation/README.md) |
| Object/node Holdout, exact allocator/flag writes and surface/volume projection | [Native Holdout and original word/state/film fixtures](validation/2026-09-09/native-holdout/README.md) |
| ShaderData geometry, packed object/primitive identity, curve segments and lamp emission | [Default-path checkpoint](validation/2026-09-07/native-default/README.md) |
| Background/NEE ShaderData, native world evaluation and camera-dependent importance baking | [Native background](validation/2026-09-08/native-background/README.md) |
| Volume Absorption/Scatter, Volume Coefficients and Principled Volume node streams and allocation state | [Native volume SVM](validation/2026-09-07/native-volume-svm/README.md) |
| Ordered volume stacks, main/shadow consumers, phase copy, runtime extrema and density baking | [Native volume consumers](validation/2026-09-08/native-volume-consumers/README.md) |
| Native scene admission, used-shader attribute residency, mesh constant emission and volume NEE emission | [Scene admission](validation/2026-09-08/native-scene-admission/README.md) |
| Assigned-but-failed image identity and native sampling before UV wrapping | [Failed-image state](validation/2026-09-08/native-missing-image/README.md) |
| Shared surface/volume closure weight accumulation, full word images and original GPU allocator state | [Shared closure weights](validation/2026-09-08/shared-closure-weights/README.md) |
| Map Range and analytic Sky node behavior | [Map Range](validation/2026-09-07/map-range/README.md), [analytic Sky](validation/2026-09-07/analytic-sky/README.md) |

The native volume consumer retains one closure allocator across the whole
ordered stack, Cycles' per-entry phase merging and its final eight-phase
active-prefix copy. Sigma_s includes only successfully allocated closures.
Majorant baking uses one/sixteen samples and session-owned resources built
after camera parameters are finalized. This does not claim imported sparse
volume grids or motion-object support.

Geometry setup uses the native KernelObject, packed triangle data,
KernelCurve and curve-key images. Curve acceleration segments map to
Cycles' containing curve plus packed segment index; they are not triangle
indices. The current scene upload admits static ribbon curves. This does not
claim motion geometry, other curve shapes or point-cloud rendering.

## Image parity and performance

The [current SDK 98f integration](validation/2026-09-09/next-integration/README.md)
passes complete host 183/183, HIP 192/192, fallback 194/194 and focused strict
native Vulkan 12/12. The Vulkan trace records 456 native SPIR-V compilations
with no DXC/DXIL load; a separate 17/17 SDK scheduler selection also passes
163 cases / 21,349 assertions on the required native route.

Fresh HIP 256-spp Barbershop (2048x858) and Monk (1440x1080) canaries take
36.6909 / 12.5366 s render-only, with 50.0167 / 31.8320 s session setup/JIT
and unchanged 416 / 200 B frames. All 46 channels are finite and all 15 pass
comparisons complete. These are single observations against retained Cycles
references, not a new four-scene paired benchmark. Barbershop remains 44.58%
above the retained native median; DiffInd relative RMSE is still 7.1262% /
12.8816%. Classroom's historical invalid separated passes and Monk's localized
run-to-run variation remain open, not waived as noise or one ULP.

The [frame/sorting audit](validation/2026-09-09/barbershop-coroutine-audit/README.md)
identified physical incoming queues splitting one logical surface target.
Generic logical-priority aggregation and explicit compatible Handler batching
now join those inputs before sorting. The complete 64-spp diagnostic changes
surface dispatches from 1,062 to 798 but surface entries from 332,307,665 to
332,307,666: reduced scheduling batches, not demonstrated less shading or a
smaller frame. No scene-name/profile-driven pruning or inline policy is used.

The following paired baseline and intermediate campaigns are revision-pinned
history; their measurements are not combined into a current performance score.

The [equal-pass HIP campaign](validation/2026-09-08/matched-pass-hip/README.md)
completed 12 fresh pairs at Psycles c5bf9247 / Luisa 9ea3b720f, using original
production Cycles 5.2.1 LTS build 9e2066aef7ef on the same RX 9070 XT.
Both engines produce exactly 15 linear passes / 46 channels at 256 fixed
samples. This supersedes earlier unequal-pass ratios and the withdrawn
Classroom speed-lead interpretation; those dated reports remain historical
evidence, not current performance claims.

Times are three-run medians in seconds. Cycles uses its main-loop wall
interval; Psycles uses render-only wall time. Main shader caching is disabled,
auxiliary/downstream/OS caches retain normal policy, native fast math is on,
and no concurrent build/render or GPU profiler overlaps these pairs.

| Scene / extent / authored seed | Cycles HIP | Psycles HIP | Relative time | Session init | Frame |
| --- | ---: | ---: | ---: | ---: | ---: |
| Monk / 1440x1080 / 0 | 13.2425 | 13.6826 | 1.0332 | 18.7327 | 220 B |
| Monster / 1080x1080 / 0 | 14.2865 | 15.1328 | 1.0592 | 22.1471 | 284 B |
| Classroom / 1920x1080 / 1 | 18.0319 | 18.7626 | 1.0405 | 18.1196 | 264 B |
| Barbershop / 2048x858 / 0 | 25.3775 | 39.2753 | 1.5476 | 25.6044 | 416 B |

At that paired checkpoint the efficiency goal is not complete: Barbershop is
54.8% slower, compared with 3.3%–5.9% for the other scenes. Session init is the CLI's
shader_jit_seconds, including JIT plus setup/baking, not compiler-only.
Cycles loads precompiled GPU kernels. Smaller IR, local arrays and frames
do not independently establish end-to-end speedup.

The [native descriptor sampler correction](validation/2026-09-08/bound-image-sampler/README.md)
and [HIP descriptor investigation](validation/2026-09-08/hip-texture-descriptors/README.md)
confirm image/sampler SRDs are already device-resident and sampled directly.
Neither more aggressive descriptor inlining nor explicit index grouping is
adopted after GPU controls. A smaller pointer chain or synthetic sampling
speed is not reported as an end-to-end renderer gain. Their dated follow-up
timings remain in those reports, not mixed with the current measurements above.

The earlier [light-endpoint correction](validation/2026-09-08/light-endpoints/README.md)
separates geometric lamp hits from spot/spread evaluation and indirect shader
visibility, matching original Cycles' empty-emission state transition. The
permanent original-GPU regression has 240 endpoint and 2048 visibility checks.
Fresh 64-spp Barbershop work counts now differ from Cycles by only 196 surface
and 70 volume visits out of 332.3M / 85.1M. This repairs a real structural
error, but surface visits were already within 0.04%; it does not explain away
the remaining surface cost per path. Subsequent lamp-routing work below fixes
post-lamp closest scheduling; shadow surplus and residual DiffInd remain.
Its original-GPU tests and dated canaries remain in that report; they are not
substituted for fresh paired Cycles measurements.

The [scheduler trace comparison regression](validation/2026-09-08/dispatch-trace-comparison/README.md)
also resolves the former fallback test failure: it passed 182/182 at that
checkpoint, with exact RNG/discrete state checks and unchanged film tolerances.
No renderer binaries or captured trace bits change with that test-only fix.
The [lamp-routing / CFG / surface investigation](validation/2026-09-08/lamp-routing-and-surface/README.md)
adds the original transparent-lamp traversal boundary, native miss-distance
normalization and loop-epoch CFG repair. The full original native module now
restructures. Hidden input default provenance and Vector Math host folding
have 81 original-Cycles word-image regressions. The
[graph-boundary repair](validation/2026-09-08/svm-graph-boundaries/README.md)
adds 24 exact images for bump edge ownership, retained BUMP displacement
entries and procedural Color/Factor sharing. The
[native socket/declaration repair](validation/2026-09-08/svm-socket-declarations/README.md)
adds 53 exact images for native texture POINT inputs, conversion type-pair
identity, closure declaration order and unavailable Voronoi defaults.
The [Blender forwarding repair](validation/2026-09-08/svm-group-forwarding/README.md)
adds 34 original images for linked/primitive group boundaries and separate
Blender luma/Gamma folds: 33 raw-exact and one with three-ULP typed literal
differences, without changing device arithmetic or expected words. The
[native Math expansion repair](validation/2026-09-08/svm-math-expansion/README.md)
adds 18 exact original images, with 15 failures before the repair. It restores
Clamp creation after native conversion links without changing SVM ranking.
The [Mapping declaration repair](validation/2026-09-08/svm-mapping-declarations/README.md)
at `61443f70` adds eighteen exact original images, including a five-node
counterexample where a redundant type conversion adds three SVM words.
The latest [group-context repair](validation/2026-09-08/svm-group-contexts/README.md)
at `239cade6` adds thirty exact original images and resolves twelve pre-fix
failures. Group inputs are evaluated lazily in the parent context; output
caches persist per instance, and sibling instances of one definition are
not mistaken for recursive nesting. All 279 Barbershop used-shader images
now have identical node/typed-payload layouts, stack addresses and jump/domain
structure: 115 raw-equal and 164 different only in declared resource-ID fields.
The six remaining schedule differences are resolved. A fresh
[same-session resource audit](validation/2026-09-09/resource-identities/README.md)
also resolves all 164 payload differences: 174 images, 21 named attributes,
600 image references and 795 attribute references agree in binding identity.
Numerically equal IDs are checked as well; no words are normalized. This
does not establish decoded texture/attribute values or complete shader parity.
At that host-only checkpoint the static stack bound is 33 floats and surface
machine code and register/private-memory requirements are unchanged.

Main SVM dispatch and most handlers are already inlined. Keeping more HIP
function boundaries in a controlled A/B/A experiment slows surface time by
35.5%; that intervention is reverted. No inlining policy or device arithmetic
is changed. A separate ordinary microfacet callable control is also reverted:
the baseline microfacet bodies are already inlined, and outlining expands
fixed private storage from 2,496 to 94,896 bytes with a large slowdown.
That checkpoint's 64-spp surface GPU total is 5.872749 s against the
retained original Cycles 2.962829 s. Surface visits remain effectively
unchanged at 332.3 million. Final machine code retains only three outlined
noise/math helpers, 256 VGPRs and 2496 fixed private bytes. The report records
actual code-object identities and callees rather than equating static spill
sites with dynamic memory traffic or mistaking pre-HIPRTC LLVM definitions
for final out-of-line functions. Per-invocation surface cost remains open.

Psycles `cc4d9974` / Luisa `d51a33d48` subsequently expose native OCML
floating remainder before IPO, preserving large-quotient range reduction.
The [original GPU and full-scene controls](validation/2026-09-09/hip-native-remainder/README.md)
show 0.44% fewer effective static instruction sites, with unchanged frame,
registers and private storage. A/B/B/A restores identical baseline `.text`,
but does not establish a substantial rendering speedup. No inline policy or
approximate quotient formula is added. Function instruction counts now use
ELF extents, excluding the padding counted by older textual audits.

Psycles `9f4c63b9` / Luisa `72bc85d96` now preserve shared BSDF case bodies
through the AST/XIR roundtrip and JIT closure-mask filtering. Seven native
groups replace 32 separately recorded bodies. The
[shared-switch audit](validation/2026-09-09/shared-switch-cases/README.md)
records 35 structural shapes and a full-scene A/control/B/B/A: main-function
instructions fall 184,759 to 159,801, while median surface/render time changes
only 0.9% / 0.4%. The actual machine code still has four functions and 49 call
sites; 256 VGPRs and the 416-byte frame are unchanged, with 2,480 fixed private
bytes. This is not a missing-inline fix or efficiency parity. Generic shared
switch-target CFG proxying and narrow signed SPIR-V literal encoding are also
repaired, each constrained by an independent failure and permanent regression.
The next checkpoint, Psycles `6a20f08d` / Luisa `da8fff856`, also restores
one native closure-setup switch with the original case groups and feature
guards. Its 72-shape production-AST regression fails 12 shapes before repair.
Full-scene A/B/B/A establishes no speedup: surface medians 5.8149 / 5.8621 s,
render medians 10.4698 / 10.5114 s. Main instructions become 159,927 while
register/frame/private sizes and 49 calls are unchanged.

Psycles `e79644af` restores original caustic/allocation input-evaluation
boundaries for standalone Glossy, Refraction, Glass and Metallic. Eight
production-AST configurations fail before repair and pass afterwards;
128 runtime states cover original GPU observations, rejected/no-storage
paths and full END/PC behavior. Full-scene A/B/B/A shows no speedup: median
surface 5.8641 to 5.8773 s and render 10.5145 to 10.5309 s. Main instructions
become 159,972, with unchanged frame/register/private sizes and 49 calls.
The typed scene census counts 237 Diffuse, 191 Glossy and one Principled
producer, not dynamic evaluation frequencies. An independent
[surface launch A/B/B/A](validation/2026-09-09/surface-launch-geometry/README.md)
finds 1024 threads slower than 512: VGPRs fall 256 to 192 but private bytes rise
2480 to 2816, with median surface/render time 5.72% / 3.06% higher. This
experiment is fully reverted, restoring identical A `.text`; all 46 channels
are finite and all 15 pass controls complete. No register cap or inline policy
changes. Remaining input/resource/code-generation differences are under audit,
not established causes of the performance gap.

The [Noise code-generation controls](validation/2026-09-09/noise-codegen/README.md)
do not reproduce a twofold primitive-cost gap or a missing 3D inline boundary.
Synthetic full-handler run medians are about 5.6% slower with strongly
overlapping dispatch ranges, despite roughly 32% more static code and fewer
registers. They are not an estimate of Noise's share of the full scene.
Psycles `ffb3f7f2` then restores the original shared F1 Voronoi octave fallback
and 3D homogeneous position W. Structural configurations change 3/8 to 8/8;
original HIP output states change 168/192 to 192/192 without editing the
fixture. Shared-F1 A/B/B/A improves median surface/render time by 1.17% / 0.66%.
The final combined version has 158,118 main instructions, 2,464 private bytes,
49 calls and the unchanged Barbershop 416-byte frame. Its final 64-spp profile
records 5.798762 surface seconds and 10.4441 render seconds; it is separate
from the isolated shared-F1 medians. No inline or launch policy changes.

Psycles `ff8385c1` restores seven scene-owned area/spot fields to the host light
packer and their original device predicates, removing per-use trigonometry
and the approximate full-spread flag. Seven AST consumers pass, and 37 native
tables / 296 original GPU states per denormal mode constrain the new boundary.
The [full-scene A/B/B/A](validation/2026-09-09/scene-light-parameters/README.md)
changes median surface/render time by only -0.73% / -0.22%, with two samples
per treatment and 0.76% variation between the A surface observations. It does
not establish a large gain, and the 256-spp follow-up is essentially unchanged.
The [static audit](validation/2026-09-09/surface-static-audit/README.md) does not
support missing major inline boundaries or float expansion of byte textures
as the main cause. Per-ray inverse-transform reconstruction remains a
confirmed phase-ownership difference whose cost is not yet quantified.

Psycles `9e3ba165` restores native captured shadow flags/weights and
direct/indirect film destinations, following the original emission/exit
eligibility guard in `4f487ba5`. The six-RGB splitter and writes to both
direct/indirect outputs are removed. Original-GPU film/state comparisons
improve 351/540 -> 540/540 and 72/180 -> 180/180; five production AST address
checks change 0/5 -> 5/5. The shared BSDF ratio now uses the native zero-only
guard, without a normal-value epsilon cutoff.

Its [full-scene A/B/B/A](validation/2026-09-09/film-routing/README.md) improves
surface/render medians by 1.88% / 0.98%, with only two observations per
treatment. Main instruction sites change 158,005 -> 157,717; 47 calls,
256 VGPRs, 2,464 private bytes and the 416-byte Barbershop frame remain.
Inlining, launch policy, RNG, SVM word structure and fast math do not change.
The corrected mixed volume-scattering routes cannot explain Barbershop's
DiffInd: that scene has emission-only fog and zero volume bounces.

The film-routing checkpoint passed host 176/176, HIP 187/187 and fallback 189/189.
Those complete runs used Luisa `da8fff856`. Upstream Metal/CI changes were
integrated to `8911828eb` before publication; the all-target Linux build
leaves all six measured HIP binaries byte-identical. Host 176/176 and
focused HIP/fallback/native Vulkan 6/6 each are rerun. The prior child
`unit*` selection is 133/133 in the HIP configuration, not a new all-platform
run. The strict native Vulkan film/emission/shadow/NEE gate has 62 native
compilations before integration and 47 in the rerun, with no DXC/DXIL load.
The frozen native-Holdout correction on published Luisa `6e58928d8`
passes complete host **183/183**, HIP **192/192** and fallback **194/194**.
Fallback takes 617.61 s after a separately retained interrupted attempt.
Strict native Vulkan **fails at 7/11**: `lamp_routing`, `zero_bsdf`,
`holdout_render` and `ray_portal_render` abort with `Instruction operand does
not dominate its use` during `restructure_cfg` output verification. CTest
exits 8 after 84.28 s. All native guards are set, and the loader audit finds
67 successful native SPIR-V compilations with no DXC/DXIL load; this is not
an all-passed gate. The compiler failure blocked that frozen checkpoint and
its completed results archive. The fresh `98f4667ca` integration linked above
passes the expanded 12-test gate; the old failed evidence is not rewritten.
No task-owned SDK implementation changes are part of this snapshot. The
preceding S2 report retains its complete fallback 192/192 and strict native
Vulkan 9/9 results, including 288 native compilations without DXC/DXIL load;
those did not substitute for the frozen repair's failed gate. All builds use 32
threads; backend tests run sequentially.

The earlier native lighting gate remains 5/6 with 36 rectangle-area sample
numerical lanes also reproduced in the pre-repair implementation. The
additional f16/f64 `OpFRem` test retains 1,076 failures; isolated float32
passes 1,744 assertions and HIP/fallback pass all 5,232. These gaps are not
waived by the focused green canary or hidden with slower HIP arithmetic.
The fallback lost-wakeup repair, source-size cleanup and shared-switch
regressions remain revision-pinned in their dated reports.

The frozen SDK-6e native-Holdout six full-resolution 256-spp renders complete against retained Cycles
references, with exact prior geometry/images and refreshed socket metadata.
Monk/Monster/Classroom one-run render times are 12.4505 / 13.7470 / 17.2434 s.
Barbershop's three are 37.8379 / 37.8313 / 37.8524 s: median 37.8379 s,
49.1% above retained native. Frames are 200 / 272 / 252 / 416 B. These are
temporal follow-ups, not fresh timing pairs or isolated causal estimates.
The Barbershop median is 0.91% above S2; no Holdout-driven speed effect is
established. Session init is 43.9887 / 52.6500 / 38.3884 s for the first three
scenes and 58.1651 / 19.1775 / 20.5389 s for Barbershop. Its main HIPRTC link
takes 27.001 s initially and 0.092 / 0.109 s on repeats, producing an
862,656-byte code object. Main shader caching is
disabled, while downstream/OS caches retain ordinary policy; init
includes setup/baking and is neither compiler-only nor controlled cold JIT.

The complete-scene all-finite gate is **5/6, not green**. Classroom repeats
18 invalid values at eight unique pixels with unchanged NaN/+Inf/-Inf masks:
seven each in DiffDir/GlossDir,
two each in DiffInd/GlossInd. Combined is finite. An original-GPU/production
DSL diagnostic reproduces the same subnormal-BSDF NaN/Inf weight classes
under native fastmath, with matching zero/normal controls. This is evidence
of a shared arithmetic boundary, not a per-path proof or a finite-pixel
waiver. No epsilon cutoff or slow division path is restored. The runner
records all comparisons on explicit finite domains and exits 2; no render
observations are discarded from that frozen campaign.
See the [historical Holdout report](validation/2026-09-09/native-holdout/README.md) for
all 15 pass metrics, source/implementation hashes and limitations. DiffInd
remains essentially unchanged. The earlier
[S1 report](validation/2026-09-09/native-path-lifetime/README.md) isolates its
SDK change and finds local Monk variation on a same-binary repeat. The
frozen S2/S3 captures also retain local variation: Monk's Combined RMSE is
5.49e-4 and Normal RMSE 1.47e-4, with maximum errors 0.283 and 0.0535. The
cause is unresolved; these changes are not attributed to Holdout, path-bound
exhaustion, SDK integration or floating-point noise without further evidence.

At the older revision-pinned paired baseline, all 46 actual channels were
finite; its first-pair relative RMSE was:

| Scene | Combined | DiffCol | DiffInd |
| --- | ---: | ---: | ---: |
| Monk | 0.01241440 | 0.000545920 | 0.12881733 |
| Monster | 0.00547872 | 0.000110133 | 0.02552496 |
| Classroom | 0.00353306 | 0.000102311 | 0.17820333 |
| Barbershop | 0.01079848 | 0.001597578 | 0.07454033 |

The report retains all 15 passes for all repeats, exact commands, observed
ranges, source/export/output/implementation hashes and seed/frame semantics.
First-pair Combined triptychs were inspected at resized viewing resolution.
Original Cycles Classroom has 25 non-finite DiffDir and 27 non-finite
GlossDir pixels per run; metrics explicitly exclude the invalid union.
Nearly empty passes retain absolute errors and reference signal scale.
These residuals are open correctness work, not waived as noise or one ULP.

Barbershop's unavailable-image admission and missing shared transparent
closure are fixed by original-Cycles word/GPU-state regressions. At the
diagnosed pixel, the first four surface events and all 45 sampled random
fields match, but a later NEE event still selects an adjacent emitter triangle.
Its generic read-only-reference frame correction reduces 896 B to 416 B
without changing six-stage control flow. The
[volume work correction](validation/2026-09-08/volume-work/README.md) has no
measurable volume-kernel speedup; surface remains the larger performance
difference. None of these observations proves global path parity.

The [same-sample Monk diagnosis](validation/2026-09-07/lone-monk-residual/README.md)
identifies a visibility divergence at coincident leaf geometry. Original
duplicate primitives are retained. It does not justify deduplication, an
unproven global RNG-mismatch claim or a slow bit-matching intersection path.
The [sampler contract](validation/2026-09-07/sampler-contract/README.md) pins
Cycles' actual automatic-scrambling property.

Remaining structural DiffInd/visibility alignment and cross-scene efficiency
are open; average-energy agreement is not same-path correctness. The old
SurfaceProgram instruction/topology histogram and CLI/API no longer describe
the native path. Closure-count histograms and indexed path traces remain
available to diagnose work without changing sampling dimensions.

## Differential policy

Use original Cycles 5.2.1 source and original Cycles GPU execution for oracle
results. Keep fixture inputs and expected outputs separate; do not generate
expected streams with the Psycles compiler.

Compare discrete state, node words, stack addresses, PC transitions,
visibility, object/primitive identity and RNG dimensions structurally.
Continuous values use native-fast-math tolerances. One-ULP differences do
not justify slower algorithms or strict software emulation.

Scene reports must state source bundle, revisions, device/backend, sampler,
seed, dimensions, sample range, scheduler and feature settings. Separate
render-only timing from cold JIT and scene compilation. Verify finite linear
passes, contribution-weighted residuals and same-sample paths; do not use
DiffInd relative RMSE alone as a count of unnecessary paths or shading.

Builds use all 32 available threads. Validate HIP first, then fallback.
Vulkan canaries require native XIR-to-SPIR-V with DXC disabled; a successful
Vulkan render through a different compiler route is not the required gate.
