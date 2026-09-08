# Native closure-setup dispatch: structure, not a speedup

Psycles `6a20f08d` / Luisa `da8fff856`, published to their requested branches,
replace the SVM closure-family predicate chain with Cycles' native switch.
The complete Barbershop A/B/B/A **does not show acceleration**: B's median
surface time is 0.8% higher and render time 0.4% higher. The correction is
retained for the required original control-flow structure, not as a claimed
performance fix. Barbershop's large remaining efficiency gap is unresolved.

## Structural reference and permanent red

The sole shader reference is Cycles
`cb168525138fecc792cc393f94afc39582b0103c`,
`intern/cycles/kernel/svm/closure.h::svm_node_closure_bsdf`.
It has one closure-type switch after the common domain and mix-weight gates,
with original shared groups and separate velvet/sheen cases. Psycles instead
recorded Principled and family-classification if/else chains followed by a
small diffuse/translucent/transparent switch. The replacement restores the
original partition and hair/principled-hair/subsurface feature guards.

The production-AST test in `tests/test_cycles_svm_closure_dispatch.cpp`
checks six feature combinations, four node masks and three shader domains:
72 shapes. The unchanged implementation fails all 12 active surface/BSDF
shapes and passes the 60 controls. The new implementation passes all 72.
Shared labels own one body, different original groups retain distinct bodies,
and non-surface / emission-only paths are constrained independently. This
records device DSL; it is not a CPU shader evaluator.

The typed payload reads, PC increments, physical closure allocation rules,
mix-zero skip, diagnostic rejection and emission-only path are unchanged.
Separate velvet/sheen cases pass their own constant type to the same helper.
No SVM word image, stack/closure extent analysis, inline/noinline policy,
register limit, fast-math setting or floating-point approximation changes.
This intervention is separate from the preceding
[shared BSDF case-body repair](../shared-switch-cases/README.md).

## Full-scene A/B/B/A

Each run uses the same full Barbershop bundle, 2048x858 / 64 spp / seed 0,
15 passes and 46 channels. The sequence is old dispatch, new dispatch twice,
then restored old source; we launch no overlapping heavy task. These are
profiled intervention controls, not fresh Cycles timing pairs.

| Run | Surface GPU seconds | Render-only seconds | Session initialization |
| --- | ---: | ---: | ---: |
| A before | 5.792392 | 10.4451 | 18.5021 |
| B native switch | 5.866546 | 10.5081 | 56.8037 |
| B repeat | 5.857586 | 10.5146 | 19.1209 |
| A restored | 5.837389 | 10.4944 | 18.9547 |

A/B medians are 5.814890 / 5.862066 seconds for surface and
10.46975 / 10.51135 seconds for render. Two observations per treatment do
not establish a significant speed difference. Main shader caching is disabled,
but downstream caches are not cleared; session initialization includes
setup/baking. The first B and repeat B are not matched cold-JIT trials.

Both A `.text` images are byte-identical, SHA-256
`0d51dac45f914ceaddcbb3e7760d698d29b4034e0a491d241542f1301f905935`.
Both B images are also identical, SHA-256
`f0cfe90d5f8c00599641929432477f0f386b78e841e91033b7b79400eeb5830d`.
The final production source and binary are B, not the restored control.

| Final surface metadata | A | B |
| --- | ---: | ---: |
| Main function instructions | 159,801 | 159,927 |
| Main function bytes | 851,788 | 852,744 |
| Static scratch load / store sites | 1,214 / 607 | 1,272 / 622 |
| VGPR / SGPR | 256 / 107 | 256 / 107 |
| Fixed private bytes | 2,480 | 2,480 |
| VGPR / SGPR spill metadata | 491 / 65 | 504 / 65 |
| Coroutine stages / fields / bytes | 6 / 93 / 416 | 6 / 93 / 416 |
| Static device call sites | 49 | 49 |

Counts use ELF `STT_FUNC` extents, not disassembly padding; static sites and
spill metadata are not executed instruction counts or dynamic spill traffic.
Actual code still has only the main kernel, 1-D noise selection, 4-D signed
noise and OCML tangent. The SVM/microfacet bodies are already inlined; the
pre-HIPRTC `hip_kernel_final_*.ll` is not the final device call graph.

All four images have 46 finite channels and all 15 pass comparisons. B versus
A Combined relative RMSE is 1.38e-5 / 1.52e-5, versus restored A's 8.13e-6;
DiffInd is 5.83e-5 / 3.83e-5, versus restored A's 3.80e-5. Maximum B DiffInd
absolute error is 0.013215. These are intervention controls, not original
Cycles parity evidence or a reason to waive the unresolved larger DiffInd gap.

## Four-scene 256-spp follow-up

Six sequential, unprofiled renders use frozen binary hashes and the original
full extents/seeds. The Cycles images/timings are retained equal-pass references,
not fresh timing pairs. All 15 pass comparisons complete and all 46 actual
channels are finite. All four first-run Combined triptychs were inspected.

| Scene | Render seconds | Session initialization | Frame | Combined / DiffInd relative RMSE |
| --- | ---: | ---: | ---: | ---: |
| Lone Monk, 1440x1080 / seed 0 | 13.6233 | 43.4675 | 220 B | 0.012398 / 0.128817 |
| Monster, 1080x1080 / seed 0 | 14.9411 | 50.8580 | 280 B | 0.005479 / 0.025525 |
| Classroom, 1920x1080 / seed 1 | 18.2859 | 33.4346 | 260 B | 0.003533 / 0.178203 |
| Barbershop, 2048x858 / seed 0 | 40.1981 median | 17.4830 / 17.7756 / 17.9603 | 416 B | 0.010521 / 0.071262 |

Barbershop's individual times are 40.2174 / 40.1981 / 40.1825 seconds. The
median is 58.4% slower than the retained Cycles median of 25.3775 seconds.
Compared with the preceding Psycles median 39.9860 seconds it is 0.53% higher;
this temporal comparison does not isolate the change. Initialization is
JIT plus setup/baking, not compiler-only; first-run downstream cache misses
are retained, not compared as matched cold JIT. The original Classroom has
25 invalid DiffDir and 27 invalid GlossDir pixels, excluded explicitly by
the comparator's invalid union. Residual correctness/performance gaps remain.

## Validation and reproducibility

The full build uses all 32 threads. Host 170/170, HIP 182/182 and fallback
184/184 pass, as do 20 focused HIP closure/hair/BSDF tests. Strict native
Vulkan lamp-routing, bump-state and BSDF-dispatch canaries pass 3/3 with
33 native SPIR-V compilations and no DXC/DXIL library load. All three guards
are required: `LUISA_VULKAN_USE_XIR=1`,
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, `LUISA_VULKAN_DISABLE_DXC=1`.
The separate existing native Vulkan f16/f64 remainder diagnostic is still
open; this three-test gate does not claim it passed.

Raw evidence is `/var/tmp/psycles-closure-dispatch-v2Ct9F`.
[archive_profiles.py](archive_profiles.py) takes `profile-before-1EjqpM`,
`profile-flat-oUpFQn`, `profile-flat-repeat-1uvQrE`,
`profile-restored-1eJpmE` and an output JSON, in that order.
[archive_results.py](archive_results.py) joins the resulting `profiles.json`,
the permanent red/green, full backend logs and frozen four-scene campaign.
[results.json](results.json) retains all exact commands, hashes and pass metrics.

The next independent audit concerns parameter evaluation placement relative
to original caustic and allocation guards. A source argument is evaluated
before a helper's guard unless optimization sinks it. Initial final-code
reductions confirm premature normal/tangent selection for rejected caustics,
including divergent visibility, but do not measure its whole-scene cost.
The 164 used-shader resource binding identities, residual shadow/DiffInd
differences, unimplemented nodes, legacy displacement removal and remaining
general CFG proof obligations also stay open.
