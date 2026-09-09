# Native surface continuation

This repair addresses **S2** of the [whole-surface audit](../surface-semantic-audit/README.md):
the SVM already populated Ray Portal closures, but the production integrator
sent a selected portal to the ordinary BSDF sampler. The same consumer boundary
also discarded native label bits and imposed additional validity predicates.
The native production path now consumes its retained ShaderData and closure
pool directly. It does not evaluate the material a second time.

The final all-target build uses 32 threads. Host tests pass **178/178**, HIP
**190/190**, fallback **192/192**, and strict native Vulkan **9/9**. The latter
records 288 native SPIR-V compilations and no DXC/DXIL loader matches. All six
high-resolution renders complete, but the all-channel finite gate is **5/6**:
Classroom retains its separated-pass nonfinite values. This is not completion
of the whole-surface compatibility or performance goal. Object holdout, the
legacy displacement bridge, and the other open audit boundaries remain open.

## Native state and control contract

The source oracle is unmodified Cycles 5.2.1 at
`cb168525138fecc792cc393f94afc39582b0103c`. Complete reference films come from
the installed production HIP build `9e2066aef7ef`, not a local CPU renderer.
The tiny GPU observer includes and calls the original integrator functions;
its host code only prepares immutable inputs and copies observed results.

| Boundary | Original operation retained by the production consumer |
| --- | --- |
| Eligibility and randoms | Reject absent SD_BSDF/SD_BSSRDF before requesting PRNG_SURFACE_BSDF; request the same counter-based tuple after NEE; pick once from the initialized closure prefix. |
| Dispatch | Selected BSSRDF, then selected Ray Portal, then ordinary BSDF. Host scene-feature checks omit unreachable portal/SSS branches before recording DSL. |
| Portal selection mass | Ordered sum over BSDF/BSSRDF closures; original `sum <= 0` rejection; multiply signed throughput by the selected weight divided by its selection PDF. |
| Portal position | Compare squared P displacement with the original `1e-9` threshold. A relocated portal clears only the source object identity. Otherwise use the original triangle self-intersection/offset operation on the closure's P, not an invented replacement position. |
| Portal ray write set | P, D, tmin=0, tmax=FLT_MAX, compact dP. No renormalization of stored portal D; no dD, MIS, pass-weight or primitive-identity update. |
| Native labels | Carry the original label to `path_state_next` and volume-boundary handling. No production `ClosureLabel -> SurfaceEvent -> ClosureLabel` roundtrip. |
| Ordinary rejection | Exactly zero PDF or zero BsdfEval, without a selected-weight, positive-PDF, finite-value or nonnegative-throughput guard. |
| Ordinary transparent edge | Update only ray tmin. Direction normalization, triangle offset and differential widening belong to the nontransparent edge. |
| First-bounce pass weights | Compute ordinary component ratios inside bounce==0, including transparent closures; keep the separate successful BSSRDF-entry update. |
| Persistent counters | Carry portal depth through surface, world, analytic/NEE light and transparent-shadow shader contexts. Update it from SD_RAY_PORTAL on the native transparent/portal transition, including a transparent closure selected from a mixed portal shader. |

The static/motion inverse passed to the offset helper is the effective native
`object_get_inverse_transform`, not unconditional use of the originally
uninitialized static `sd.ob_itfm_motion`. This does not enable motion-object
scene admission. Portal plus volume NEE is also **not certified** by these
surface fixtures: the original volume-direct-light producer does not copy a
portal counter like its surface counterpart, and that separate boundary still
needs an original execution-domain witness.

The public diagnostic SurfaceSample adapter remains available for its existing
tests. Production native continuation bypasses it; this does not pretend that
all legacy adapters or the displacement prepass have already been deleted.
Transparent-glass settings/service integration remains open even though its
native label bit is no longer lost by the ordinary continuation consumer.

## Permanent original-GPU regressions

The state fixture covers 26 cases: relocation and no relocation, triangle
interior/outside and both sides of the displacement threshold, applied and
unapplied transforms, signed weight, mixed closure selection, excluded holdout
mass, zero/negative total mass, transparent/portal limits, native transition
labels, and both triangles at an observed shared-edge point. The test compares
28 float lanes with `2e-6 + 2e-6 * abs(original)` tolerance and 16 integer lanes
exactly. Diagnostic transition controls are explicitly distinguished from
states known to be produced by admitted scenes. Fields outside the helper's
write set are a source-level write-set check, not an independently observed
full-render preservation test.

The complete-render fixture imports exact original scene JSON and geometry
bytes and uses the normal production compiler/session. Eleven 16x16, one-sample
scenes run with megakernel and staged scheduling (capacity 64, exercising refill):

- Ordinary transparent and baseline direction-only portal controls.
- Portal depth read by world and subsequent surface emission.
- A two-portal chain and signed portal weight.
- Transparent/portal mixture and transparent limit zero.
- Portal depth in NEE light emission and transparent shadow shading.
- The deliberately unlinked Position contract, without a shared-edge pixel diagonal.

All 256 RGB pixels are compared against the original HIP film. There is no
host-calculated expected image, per-pixel exclusion, altered tolerance, or
software arithmetic path. Both original-input hashes and the Position-link
contract are checked by the permanent host fixture test.

The original red production witness remains in
`/var/tmp/psycles-ray-portal-pzGSbl/render-red-hip.log`: both schedulers disagree
on all 768 portal RGB lanes, while both transparent controls agree. CTest's
repeated printing of failed output is not counted as additional observations.

## A fixture defect found by the full fallback gate

The first extended fixture set passed HIP but failed fallback: eight extended
scenes disagreed, with a 16-pixel diagonal in the basic depth scenes. This was
not waived or presented as a green full suite. The failed suite, diagnostic
pixel output, original inputs and original/actual path captures are retained
under `/var/tmp/psycles-ray-portal-pzGSbl`.

The generator had assigned a numeric Position default without linking it.
Original `RayPortalBsdfNode::compile` uses `input_link("Position")` because this
socket is LINK_POSITION. Therefore those scenes did **not** test the intended
relocation: they kept sd.P and changed direction only. At pixel (15,15), the
original HIP and actual HIP select triangle 0; fallback selects triangle 1 at
the same shared edge. All reconstruct the same surface point
`(0.468780517578125, 0.468780517578125, 3)`.

Fallback then alternates between neighboring triangles after the offset,
consuming transparent bounces. The unchanged original GPU portal function,
fed either exact triangle input, produces z=`3.00006104`; HIP and fallback
match both new boundary-state rows. Thus there is no demonstrated difference
in the ported offset predicate for those inputs. Different shared-edge
traversal choices must not be silently relabeled as a coroutine or SVM bug.
General cross-backend behavior at this degenerate boundary is not certified.

The corrected generator connects an explicit position vector, so the intended
relocation is actually executed, and adds an independently captured unlinked
Position control away from the tie. All affected films were recaptured by the
same production Cycles HIP binary. The original tied-edge captures remain
evidence; the extra GPU state cases preserve the exact offset contract. No
renderer or Luisa arithmetic/intersection implementation was changed to make
this fixture pass.

## Storage and provenance boundaries

Non-portal scene specialization omits persistent portal-counter writes. Main
coroutine storage is measured separately from SVM scratch and the shadow task.
The latter uses Cycles' adjacent 16-bit transparent/portal counters: its AoS
size remains 224 bytes, but Luisa's scalar SoA slots are word-sized, so this
adds **four SoA bytes per path**. Unchanged AoS size must not be reported as
zero extra storage. The permanent task-layout test checks width and roundtrips
the new field with two capacities.

The implementation binary snapshot incorporates root upstream `63b36376` and
Luisa `6e58928d8`. The latter is the separately published generic packed-Boolean
definedness correction. Its formal analysis and dedicated split/materialize
and HIP packed-word regressions were reviewed/run here. No renderer-specific
policy or unrelated child changes were added to Luisa for this continuation
repair. Any before/after comparison against the earlier S1 snapshot changes
both renderer and SDK and cannot isolate an S2 speedup.

## Six-render HIP follow-up

The RX 9070 XT (gfx1201, ROCm 7.2) campaign uses 256 spp, the retained equal-pass
Cycles references, exact prior geometry/images and refreshed socket metadata.
These are temporal observations, **not fresh paired speedup measurements**.

| Scene | Resolution | Render seconds | Session init seconds | Main coroutine |
| --- | --- | ---: | ---: | --- |
| Lone Monk | 1440 x 1080 | 12.6946 | 42.8280 | 4 stages, 50 fields, 200 B |
| Monster | 1080 x 1080 | 13.7475 | 53.6947 | 6 stages, 68 fields, 272 B |
| Classroom | 1920 x 1080 | 17.2365 | 38.7979 | 5 stages, 63 fields, 252 B |
| Barbershop, median of three | 2048 x 858 | 37.4951 | 59.1673 / 17.3720 / 18.3936 | 6 stages, 92 fields, 416 B |

Barbershop renders are 37.3847 / 37.5162 / 37.4951 s. Its median remains
47.8% above retained Cycles (25.3775 s); the performance goal is not met.
The main code object is 860,096 bytes. Its HIPRTC links take 25,015.45 ms
initially and 87.48 / 89.76 ms on repeats. Session initialization includes
setup and baking, not only compilation. The previous S1 frames were
216 / 276 / 256 / 416 B; renderer and SDK both changed, so neither the
frame reduction nor timing change is exclusively attributed to this repair.

First-run DiffInd relative RMSE is 12.882% / 2.553% / 17.820% / 7.126%
in table order, essentially unchanged. Combined is 1.240% / 0.548% /
0.353% / 1.052%. The report retains all 15 passes, not just these summaries.
Classroom has 18 invalid channel values at eight pixels, with exactly the
same NaN/+Inf/-Inf masks as S1: DiffDir/GlossDir each have seven invalid
pixels, DiffInd/GlossInd two each; Combined remains finite. Original Cycles
has 25/27 invalid DiffDir/GlossDir pixels at different coordinates. Metrics
explicitly report and exclude the invalid union; the campaign exits 2.

The captured S1-to-current image comparison also retains local Monk changes
(Normal RMSE 0.00007935, Combined RMSE 0.00029449). An earlier S1 same-binary
repeat already showed local variation. Its cause remains unresolved; neither
that repeat nor the small full-image RMSE proves current path parity or
licenses attributing these changes to ordinary floating-point noise.
Classroom and Monster Normal captures are unchanged. These observations
and all input/output/binary identities are archived in [results.json](results.json).

## Remaining whole-surface work

The interpreter's word stream, typed payloads, PC, feature masks, host-side
case pruning and static stack/closure sizing are unchanged. The node-presence
inventory remains 99 handlers, nine missing semantic opcodes and two sentinels,
not a semantic-coverage percentage.

The repair removes the source-level W3/W4 and surface-BSDF part of W5 placement
differences. Their actual cost must be measured, not inferred from source size.
Barbershop has no portal closure, so portal correctness itself cannot explain
its performance or DiffInd gap. High-priority remaining work includes:

- W1: ShaderData/SurfacePoint reconstruction and eagerly retained triangle,
  normal and material data; final load use-def and lifetime evidence required.
- W2/S5: native data-pass eligibility, termination ordering and atomics.
- W6-W8: forward MIS rejection, NEE finalization and packed lamp-transform ownership.
- S3 and the other feature boundaries: object/node holdout, missing opcodes,
  transparent glass, shadow catcher, native displacement and incomplete services.

No new inline/noinline policy, CPU reference renderer, profile-derived local
allocation, scene-name specialization or slow bit-matching path is introduced.

## Gate replay and evidence

The fixture input and original film identities are in
`tests/data/cycles_ray_portal_render/manifest.json`; the standalone original
GPU state provenance is in `tests/data/cycles_ray_portal_state.json`.
`capture_results.py` archives existing observations and `validate_results.py`
checks their identities and stated limits. Neither computes a transport oracle.

```sh
cmake --build build --parallel 32
ctest --test-dir build --parallel 32 --output-on-failure -E '_(hip|fallback|vk|cuda|metal|dx)$'
ctest --test-dir build --parallel 1 --output-on-failure -V -R '_hip$'
ctest --test-dir build --parallel 1 --output-on-failure -V -R '_fallback$'
env LUISA_VULKAN_USE_XIR=1 LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 LUISA_VULKAN_DISABLE_DXC=1 LD_DEBUG=libs \
  ctest --test-dir build --parallel 1 --output-on-failure -V \
  -R '^psycles\.luisa_cycles_(path_lifetime|volume_boundary|volume_emission_film|lamp_routing|zero_bsdf|film_routing|svm_subsurface_exit|ray_portal_state|ray_portal_render)_vk$'
```

Device tests and performance renders run sequentially; no compiler build or
other device job overlaps a timed render. Main shader caching is disabled for
the render campaign; ordinary downstream/OS cache policy is unchanged. Init
includes setup and baking and must not be called a compiler-only or controlled
cold-JIT measurement. The retained original Cycles images/timings are not new
same-session timing pairs. The previous S1 images and same-binary repeat are
retained controls, not an isolated SDK-constant experiment for this repair.
