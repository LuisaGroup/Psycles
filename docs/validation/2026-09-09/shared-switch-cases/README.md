# Barbershop shared-case structure and final-code audit

Psycles `9f4c63b9` / Luisa `72bc85d96`, both published, restore Cycles'
shared BSDF case bodies without changing inlining policy. The Barbershop
surface main function has **13.5% fewer machine instructions**, but the
controlled surface-time change is only **0.9%** and render-wall change **0.4%**.
This is a retained structural correction, not an explanation of the remaining
large Cycles performance gap or a claim of efficiency parity.

## What differed, and what is repaired

Original Cycles `cb168525138fecc792cc393f94afc39582b0103c`,
`intern/cycles/kernel/closure/bsdf.h`, groups GGX reflection/refraction/glass
around one sample/evaluation body, and similarly groups Beckmann. Roughness,
label and blur switches have their own explicit shared groups. Psycles had
recorded the same helper separately for each label. Seven native groups now
record seven bodies instead of 32. Host closure-mask filtering remains: an
unused label is not emitted, and an empty group records no body. The change
does not merge arbitrary equal-looking computations or change SVM words.

XIR already supports several labels targeting one block, but its AST
roundtrip duplicated the body. Luisa now retains a group of integral labels
around one owned body through DSL, duplication, serialization, XIR and AST
backends. The old single-label tag, hashes and binary form are preserved.
The production-AST regression fails 18 of 35 shapes before the application
change and passes all 35 afterwards, including filtered subsets, absent
groups and different native groups retaining distinct bodies.

Native Vulkan exposed two independent generic defects during validation:

- `split_switch_cases` introduced separate proxies for a shared target,
  destroying its case-entry dominance and causing exit repair to invent an
  artificial loop. A five-block direct-XIR reduction gives 40 failed checks
  before repair; all 48 shapes / 552 assertions pass afterwards. Removing
  that obsolete normalization leaves genuine cross-construct entry checks
  and `fixup_construct_exits` validation intact.
- Signed narrow literals were correctly canonicalized in XIR but not sign
  extended in physical 32-bit SPIR-V words. The old ungrouped spelling also
  fails. Native emission now sign-extends signed 8-/16-bit values; unsigned
  and wide values are unchanged, with no added device arithmetic.

The child [formal audit and permanent regressions](../../../../third_party/LuisaCompute/docs/validation/2026-09-09/shared-switch-cases/README.md)
record the reductions and remaining general CFG proof obligations.
[OpSwitch](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpSwitch)
requires unique literal values, not unique targets, and specifies narrow
signed literal extension. No shader-specific compiler exception is used.

## Full-scene A/control/B/B/A

All runs use the full Barbershop scene, 2048x858 / 64 spp / seed 0, the same
exported socket/geometry/image data and sequential rocprofv3 capture. No
other heavy task we launched overlaps these timed runs. A/control/B/B/A
means old source, translator-only control, grouped source twice, then old
application spelling restored with the generic compiler fixes still present.
These are intervention controls, **not fresh Cycles timing pairs**.

| Run | Surface GPU seconds | Render-only seconds | Session initialization |
| --- | ---: | ---: | ---: |
| A before | 5.862077 | 10.5105 | 25.7230 |
| Translator-only control | 5.867189 | 10.5401 | 25.6491 |
| B grouped | 5.818743 | 10.4794 | 44.7635 |
| B repeat | 5.799153 | 10.4565 | 18.8310 |
| A restored | 5.858331 | 10.5076 | 23.5979 |

A versus B medians are 5.860204 / 5.808948 seconds for surface and
10.50905 / 10.46795 seconds for render wall. Only two observations per
intervention are available; these small differences are not a significant
speedup claim. Main shader caching is disabled, but downstream HIPRTC/OS
caches are not cleared. Session initialization includes setup/baking, not
only compilation; first B and repeated B are not comparable cold-JIT trials.

Translator-only and restored A `.text` are byte-identical to the original A:
SHA-256 `fa5f457a07fc55b687c642c18b72b92995df48590eefb523a4b4a50dc8a3dde4`.
Both B `.text` images are also identical to one another. Thus the shared-case
facility alone does not perturb this ungrouped program's final machine code.

| Final surface metadata | A | B |
| --- | ---: | ---: |
| Main function instructions | 184,759 | 159,801 |
| Main function bytes | 992,192 | 851,788 |
| All four functions' instructions | 188,137 | 163,179 |
| VGPR / SGPR | 256 / 107 | 256 / 107 |
| Fixed private bytes | 2,496 | 2,480 |
| VGPR / SGPR spill metadata | 494 / 65 | 491 / 65 |
| Coroutine stages / fields / bytes | 6 / 93 / 416 | 6 / 93 / 416 |
| Static device call sites | 49 | 49 |

Instruction counts are bounded by ELF `STT_FUNC` extents, excluding decoded
alignment padding. They are not dynamic instruction counts or spill traffic.
Actual final code contains the main kernel, 1-D noise selection, 4-D signed
noise and OCML tangent. **The interpreter and microfacet bodies are already
inlined.** The pre-HIPRTC file called `hip_kernel_final_*.ll` still contains
other definitions which HIPRTC later inlines; its name is not proof that a
device call survives. No forceinline/noinline or register-cap intervention is
used here. Earlier unsuccessful callable-boundary experiments remain reverted.

Every image has 15 passes / 46 finite channels. Against A, B Combined relative
RMSE is 1.39e-5 / 1.76e-5, versus restored A's 1.52e-5. DiffInd is
5.95e-5 / 4.80e-5, versus restored A's 3.82e-5. The maximum B DiffInd absolute
difference is 0.013215. Surface work is 332,307,892 / 332,307,895 /
332,307,896 visits for A / B / B repeat; small run-to-run differences are
retained, not forced into bit equality. These image controls
do not replace original-Cycles comparisons or justify ignoring residual DiffInd.

## Four-scene 256-spp follow-up

Six unprofiled renders use frozen binary hashes, the original full-resolution
scenes and authored seeds, fresh socket metadata and byte-identical prior
geometry/images. The Cycles references are retained equal-pass runs, not new
timing pairs. Every render has 15 comparisons / 46 finite actual channels;
all four first-run Combined triptychs were visually inspected.

| Scene | Render seconds | Session initialization | Frame | Combined / DiffInd relative RMSE |
| --- | ---: | ---: | ---: | ---: |
| Lone Monk, 1440x1080 / seed 0 | 13.5134 | 29.1013 | 220 B | 0.012409 / 0.128815 |
| Monster, 1080x1080 / seed 0 | 14.8374 | 39.3481 | 280 B | 0.005479 / 0.025525 |
| Classroom, 1920x1080 / seed 1 | 18.3748 | 28.6316 | 260 B | 0.003533 / 0.178203 |
| Barbershop, 2048x858 / seed 0 | 39.9860 median | 18.2914 / 17.8437 / 18.3194 | 416 B | 0.010521 / 0.071262 |

Barbershop's three times are 39.9699 / 40.0083 / 39.9860 seconds. Its median
is 57.6% slower than the retained Cycles median of 25.3775 seconds and 0.57%
below the preceding Psycles checkpoint; neither temporal comparison isolates
this source change. Initialization is not render time or a matched cold-JIT
comparison. Original Classroom still has 25 invalid DiffDir and 27 invalid
GlossDir pixels; comparisons explicitly exclude the union of invalid pixels.
Residual image differences and the performance objective remain unresolved.

## Validation and reproduction

Both complete builds use 32 threads. Current gates pass host 169/169, HIP
182/182, fallback 184/184 and child HIP-configuration unit tests 133/133.
Strict native Vulkan lamp-routing, bump-state and BSDF-dispatch canaries pass
3/3, with 33 native SPIR-V compilations and no DXC/DXIL load. The new generic
shared-case runtime has 211 passing assertions on each of HIP, fallback and
strict native Vulkan, including narrow/wide signed/unsigned selectors,
default, break/continue/return and both coroutine schedulers.

The separate existing native Vulkan f16/f64 remainder diagnostic remains
open (1,076 failing assertions at its retained checkpoint). Passing these
switch/lamp/bump canaries does not turn that diagnostic green.

Raw evidence: `/var/tmp/psycles-shared-case-Btjjlq`. Profile directories are
`profile-before-U9uSUl`, `profile-translator-aCqk6M`, `profile-grouped-rx13cv`,
`profile-grouped-repeat-6slThm` and `profile-restored-j95Rae` in that order.
[archive_profiles.py](archive_profiles.py) takes those five directories and
an output JSON. [archive_results.py](archive_results.py) joins the profiles,
regressions, backend gates and frozen four-scene campaign. [results.json](results.json)
retains exact commands, file hashes, image channels, stage counts and timings.

Remaining surface work includes closure-setup switch structure and the
placement of parameter evaluation relative to native caustic/allocation
guards; neither is yet established as the dominant runtime cost. The 164
used-shader resource binding identities, residual shadow/DiffInd differences,
unimplemented nodes and remaining legacy displacement removal also stay open.
