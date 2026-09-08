# Empty light hits are still path events

## Technical summary

The actual Barbershop surface workload is almost the same as original Cycles,
not 1.4x larger as a dispatch-count comparison might suggest. A separate real
control-flow error was found: Psycles rejected geometrically hit spot/area
lights when their emission evaluated to zero, and rejected indirect shader
visibility exclusions during intersection. Cycles handles those predicates in
the forward-light shading stage; the hit still advances the ray, increments
the transparent bounce count and splits any volume segment.

The fix restores these endpoint semantics without changing SVM words, random
numbers, floating-point tolerances, fast math, inlining, descriptor layout or
Luisa code. It substantially aligns active surface, volume and lamp counts,
but does not establish a surface-kernel speedup or complete the parity goal.
Implementation `7419301e` is published on `origin/main`; Luisa remains
unchanged at `9ea3b720f`.

## Actual path visits explain the routing error, not the surface cost

These are exact queue cardinalities for one original Barbershop render at
2048x858, 64 spp, fifteen linear passes and 112,459,776 primary paths. Original
Cycles is production Blender 5.2.1 LTS build 9e2066aef7ef on HIP. The before and
after Psycles runs use the same exported scene and sampling settings.

| Stage | Cycles visits | Psycles before | Psycles after |
| --- | ---: | ---: | ---: |
| Surface | 332,307,907 | 332,438,334 | 332,307,711 |
| Volume | 85,077,391 | 82,957,336 | 85,077,321 |
| Forward lamp | 12,869,560 | 7,636,826 | 12,868,738 |
| Background | 377,081 | 378,002 | 376,879 |
| NEE light | 138,961,986 | 139,002,446 | 138,958,516 |
| Shadow intersection | 45,312,251 | 46,001,217 | 45,986,709 |
| Shadow shading | 20,711,462 | 20,806,079 | 20,799,634 |
| Closest intersection continuation | 444,271,208 | 431,516,523 | 431,394,723 |

Before the fix, surface visits were only 0.03925% above Cycles. The retained
pre-fix GPU profile's 5.6336 versus 2.9628 surface seconds therefore cannot be
explained by a comparable excess of surface-stage visits. This does not count
SVM instructions, closure operations, texture samples or divergent lane work
inside a visit; those remain possible per-path cost differences.

After the fix, surface differs by -196 visits, volume by -70 and lamp shading
by -822. Near-equal aggregate counts are not proof of identical paths or
random decisions. Shadow intersection still runs about 1.49% more visits.

The closest row also has a known structural qualification: current Psycles
caches the mesh intersection while visiting intervening transparent lamps,
whereas Cycles returns from SHADE_LIGHT_FORWARD to INTERSECT_CLOSEST. Adding
Psycles' lamp visits to its closest visits gives 444,263,461, just 7,747 below
Cycles, but that is a **counterfactual count**, not work that Psycles actually
executes. This remaining stage boundary must not be hidden by relabeling the
raw counters.

## Why a zero evaluation cannot delete the intersection

Original `lights_intersect_impl<true>` uses point/spot/area geometry and camera
exclusion or indirect MIS participation to select the nearest lamp. It does
not require a positive spot attenuation, area spread evaluation or indirect
shader visibility. Original `integrate_light_forward` advances `ray.tmin`
before its empty-evaluation and visibility returns; its caller increments the
transparent bounce count even for SHADER_EVAL_EMPTY.

Psycles had conflated geometric intersection with the emission measure:
`intersect_spot` required a positive evaluation factor, `intersect_area`
required `evaluation.valid`, and the closest stage applied every visibility
mask before selecting the hit. The error changes future state even when the
immediate light contribution is zero. In a volume, removing a boundary also
changes the segment lengths and the sequence of distance-tracking decisions.

The correction keeps `valid` geometric, moves indirect shader visibility to
the forward-light stage, and performs shader/MIS work only after the original
eligibility checks. The exact Cycles visibility predicate is retained:
diffuse transmission observes EXCLUDE_DIFFUSE, while EXCLUDE_GLOSSY applies
only to reflection. Ordinary mask overlap is not equivalent for compound
path visibility.

## Original GPU regressions and image evidence

The new [oracle](../../../../tools/cycles_light_endpoint_oracle.hip) directly
executes original Cycles `lights_intersect_impl<true>`,
`light_eval_from_intersection` and `is_light_shader_visible_to_path` on HIP.
The shared fixture contains authored inputs, not expected intersection or
transport formulas. Twelve geometric cases and five visibility/MIS modes
cover point spheres/disks, lit/dark spots, lit/dark rectangles/ellipses,
back-facing areas, zero radius, one-sided spots and excluded ray intervals.

The permanent [runtime regression](../../../../tests/test_luisa_light_endpoints.cpp)
calls the actual Psycles closest-event stage and checks 240 output components
against the saved GPU oracle. The original implementation failed 46 checks;
the corrected implementation passes. A further 2,048 original GPU visibility
predicates cover all five visibility/exclusion bits and both reflection flag
states. Discrete results are exact; continuous values use a 1e-4 scaled
tolerance and fast math remains enabled. No CPU reference renderer is added.

The full 64-spp before/after comparison uses a fresh production Cycles image,
matching build metadata and every one of the fifteen passes. All 46 actual
channels, including alpha, are finite. Relative RMSE changes as follows:

| Pass | Before | After |
| --- | ---: | ---: |
| Combined | 2.04374% | 1.97879% |
| Diffuse color | 0.190017% | 0.190007% |
| Diffuse indirect | 11.19308% | 10.59732% |

Both volume light passes are identically zero in this absorption-only scene.
Their zero RMSE is not evidence that volume boundary placement is irrelevant.
The residual DiffInd difference remains unresolved and is not dismissed as
sampling noise or addressed with slower bit-matching arithmetic.

## Full-resolution 256-spp follow-up

Six sequential canaries use frozen implementation binaries, the same fifteen
passes and the retained original Cycles 256-spp references. They are not new
paired Cycles/Psycles timing measurements. Main shader caching is disabled;
downstream cache policy is unchanged. No own build, test or profiler overlaps
these renders. Session initialization includes JIT, setup and baking, not
only compilation.

| Scene / repeat | Resolution | Scene compile, s | Session init, s | Render, s | Frame |
| --- | --- | ---: | ---: | ---: | ---: |
| Barbershop / 1 | 2048x858 | 15.0547 | 25.2726 | 39.2464 | 416 B |
| Lone Monk / 1 | 1440x1080 | 4.96221 | 18.6766 | 13.4148 | 220 B |
| Monster / 1 | 1080x1080 | 1.46416 | 62.6490 | 14.8794 | 284 B |
| Classroom / 1 | 1920x1080 | 2.01063 | 21.8161 | 18.0993 | 264 B |
| Barbershop / 2 | 2048x858 | 14.7303 | 22.7758 | 39.2392 | 416 B |
| Barbershop / 3 | 2048x858 | 14.9778 | 24.8939 | 39.2907 | 416 B |

Barbershop's median is **39.2464 s**, 0.92% longer than the preceding sampler
checkpoint's 38.8902 s, and 1.5465x the retained Cycles median of 25.3775 s.
This correctness correction is not a speedup. These desktop follow-ups do
not establish statistical significance for the small before/after change.

All 46 Psycles channels are finite in every run. First-repeat relative RMSE
against the retained 256-spp Cycles images is:

| Scene | Combined | Diffuse color | Diffuse indirect |
| --- | ---: | ---: | ---: |
| Barbershop | 1.05209% | 0.159748% | 7.12616% |
| Lone Monk | 1.24024% | 0.054593% | 12.88156% |
| Monster | 0.547872% | 0.010992% | 2.55250% |
| Classroom | 0.353320% | 0.010232% | 17.82033% |

Barbershop DiffInd improves from the preceding checkpoint's 7.45404% to
7.12616%. The higher 64-spp errors above use a different sample count and
fresh reference; they are not an additional before/after speed or quality
comparison. The retained Classroom reference has 25 invalid DiffDir and 27
invalid GlossDir pixels; those pass comparisons exclude their invalid-pixel
union. Psycles does not introduce non-finite output. The checked-in evidence
retains all fifteen passes and their valid/invalid denominators.

Monster's first initialization includes links taking 4.1445 s and 34.8881 s
for 440,256-byte and 1,032,256-byte code objects. A separate unchanged-command
repeat, keeping main shader caching disabled, takes 0.03715 s and 0.15263 s
for those links; initialization is 22.159 s and rendering 14.8515 s. All 46
channels remain finite and all fifteen passes are compared again. This
supports a transient/downstream-link-cache explanation, not an isolated
causal cache experiment. The first result is retained rather than replaced
by this extra warm control.

## Validation status

The 32-thread all-target build succeeds. Registered HIP tests pass 181/181;
fallback passes 183/183, including the formerly failing dispatch-film test.
Host tests pass 156/157: only the existing four source-size violations remain,
with no limit relaxed. The strict native XIR -> SPIR-V endpoint canary passes
all 2288 checks across two modules; loader tracing confirms no DXC/DXIL load.
The final original-Cycles GPU oracle reproduces its checked-in data exactly.

## Measurement boundaries and next work

Cycles TRACE `GPU queue launch ..., work_size ...` supplies active queue
lengths for the shading/intersection stages. CYCLES_DEBUG_PER_KERNEL_PERFORMANCE
is unset, so the optional per-kernel synchronization path is not enabled.
Cycles camera initialization is the exception: its reported work includes
tile padding. The camera volume-stack count, not that padded launch size,
matches the primary-path total in this scene.

Luisa's existing `LUISA_CORO_WAVEFRONT_STATS=1` records the exact frame count
submitted to each continuation, without modifying kernels, frames or queue
semantics. The analyzer checks entry/generator and resume conservation for
each dispatch. Pre-resume sort extensions are recorded separately and never
added a second time to surface visits. Auxiliary NEE/shadow queues are distinct
from main-path continuations. Both Psycles frames remain 93 fields / 416 bytes.

These logging runs are work-count diagnostics, not timing benchmarks. The
retained kernel profile predates this correction. The remaining useful work
is to restore the post-lamp closest-stage boundary, separate lamp intersection
from evaluation temporaries, investigate the shadow-work surplus, and compare
surface cost per active path. Aggregate parity does not prove whole-path
parity, and the large surface performance gap remains open.

Exact commands, sources, checksums and reproduction are in the
[source notes](source-notes.md). The [audited evidence](evidence-summary.json)
retains work counts, high-resolution canaries, the separate Monster repeat,
implementation hashes and gates. The [original red log](data/endpoint-red-hip.log)
preserves the missing-endpoint failures.
