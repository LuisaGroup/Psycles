# Closure input guards: corrected work placement, no measured speedup

Psycles `e79644af` / Luisa `da8fff856` restore the original typed-input
evaluation boundaries for standalone Glossy, Refraction, Glass and Metallic.
The original-GPU state regression and full backend gates pass. The complete
Barbershop A/B/B/A does **not** show acceleration: median surface time changes
from 5.8641 to 5.8773 seconds and render time from 10.5145 to 10.5309 seconds.
This is a structural correction, not the solution to the remaining large
surface-kernel efficiency gap. No inline policy or register cap changes.

## Cause, reduction and permanent regression

The sole shader reference is Cycles
`cb168525138fecc792cc393f94afc39582b0103c`, `kernel/svm/closure.h`.
Its `svm_node_get<T>()` advances the payload PC without evaluating its inputs.
The four native handlers reject disabled caustics before normal/parameter
stack reads. Psycles instead evaluated those arguments before entering a
setup helper which owned the rejection. An input expression can contain
conditional dynamic stack loads; it is not just a host-side argument.

The isolated Glossy reduction uses production device routines, dynamic
visibility/words/stack values, fast math and ordinary compiler decisions.
Final HIP ISA confirms premature normal and tangent register selection in
both uniform and divergent visibility controls. In `eager-divergent/isa.txt`,
`v_movrels` at 0x1648..0x1650 and 0x1730..0x1738 precede the visibility EXEC
branch at 0x1760. The guarded control puts that branch at 0x15a4 before the
payload selections. This proves extra work in the reduction, not its share
of full-scene execution time.

`tests/test_cycles_svm_closure_guards.cpp` records the production AST and
follows scalar data dependencies from visibility, not source names. Eight
configurations cover four caustic settings with/without physical storage,
checking four native shared families each. All eight fail before the repair
and all pass afterwards. The test constrains the caustic scope and no-storage
case, not every finer allocation/anisotropy branch by itself. The independent
72-shape closure-dispatch regression remains green.

`NodeDataView<Payload>` captures the word-buffer address and advances the PC
once by the real typed size. `offsetof`-checked accessors load fields at their
use site; there is no new device object, stream layout or dispatch mechanism.

| Native handler | Retained input/allocation boundary |
| --- | --- |
| Glossy | Caustic gate, N, allocation, roughness/anisotropy; tangent/rotation only under anisotropy; color only for MultiGGX |
| Refraction | Caustic gate, N, allocation, IOR/roughness |
| Glass | Caustic gates, N/thin film, ordinary plus extra allocation, remaining inputs |
| Metallic | Caustic gate, allocation, N/roughness/anisotropy; distribution flags before extra allocation; conductor/F82 inputs after success |

Rejected/no-storage cases still advance the complete payload. Failed extra
allocation retains native rollback and already-set distribution flags.
Principled and emission-only behavior are untouched. The four eager internal
setup interfaces had no other clients and are removed, not kept as aliases.

`tools/cycles_svm_closure_guards_oracle.hip` runs the **original HIP SVM** with
four previously frozen external word images and original `scene/shader.tables`.
It captures 96 states: four caustic settings, four shaders, camera/diffuse
visibility and capacities 0/1/4. The runtime regression adds 32 genuine
no-storage controls and checks all 128 through full `eval_nodes`, including
exact flags/count/left/type and END/PC. Live floats use finite abs-relative
tolerance 2e-5; no shading algorithm changes for numerical agreement.

Two harness issues are retained in the evidence: missing original lookup
tables initially caused a GPU fault, and an initial observer read Beckmann's
unused/uninitialized `energy_scale`. Both observers now capture that field
only for GGX. No production initialization or expected shading values were
patched to hide either issue. The corrected original output is byte-identical
to the frozen 96-row fixture. There is no CPU shader evaluator.

## Full Barbershop control

Four sequential profiled runs use the same 2048x858 / 64 spp / seed 0 bundle,
15 passes / 46 channels and no overlapping heavy task launched by the agent.
These are intervention controls, not fresh Cycles timing pairs.

| Run | Surface GPU seconds | Render seconds | Session initialization |
| --- | ---: | ---: | ---: |
| A before | 5.856467 | 10.5051 | 18.7192 |
| B guarded | 5.892689 | 10.5413 | 52.4028 |
| B repeat | 5.861863 | 10.5204 | 20.0859 |
| A restored | 5.871651 | 10.5239 | 19.1482 |

The B medians are 0.23% higher for surface and 0.16% higher for render; two
observations per treatment do not establish a significant difference.
Session initialization includes setup/baking. Main shader caching is disabled,
but downstream caches are not cleared; first/repeat B are not matched cold
JIT measurements. Render timing excludes initialization.

Both A `.text` images are identical, SHA-256
`f0cfe90d5f8c00599641929432477f0f386b78e841e91033b7b79400eeb5830d`.
Both B images are identical,
`431b702c763b641fcc499690078472dc428fe62f7cc33297a4fab44c59e1e9fe`.
The production source and binary were restored to B before full validation.

Main-function bytes/instructions change from 852,744 / 159,927 to
852,836 / 159,972. Static scratch load/store sites change from 1,272 / 622 to
1,274 / 622. All controls retain 256 VGPRs, 107 SGPRs, 2,480 private bytes,
504/65 VGPR/SGPR spill metadata, and six stages / 93 fields / 416 frame bytes.
Static sites and spill metadata are not executed counts or dynamic traffic.
All 49 call sites remain; actual functions are the main kernel, 1-D noise,
4-D signed noise and OCML tangent. SVM/microfacet bodies are already inlined.

All 46 channels are finite. B-versus-A Combined relative RMSE is
7.83e-6 / 1.38e-5, versus restored A's 1.72e-5; DiffInd is
3.66e-5 / 5.85e-5, versus restored A's 4.66e-5. All 15 passes are archived.
These small intervention differences do not explain or waive the larger
original-Cycles residual differences.

## Static scene mix and next independent checks

The typed census checks original/actual node boundaries and closure-type
words for all 279 used Barbershop shaders. It finds 462 surface closure
producers: 237 Diffuse, 191 Glossy (162 GGX, 10 Beckmann, 19 MultiGGX),
26 Transparent, five Glass, one Refraction, one Translucent and one Principled.
There are 346 image, 188 box-image, 138 Noise and 40 Voronoi nodes.
These are **static counts, not dynamic frequencies**. In particular,
Principled is not the dominant authored node family in this scene.

The original profile launches surface at 1024 threads/group; current Psycles
uses 512. This is an identified launch-policy difference, not a measured
cause of the slowdown. Subsequent controls must isolate it from SVM changes,
keep normal compiler inline decisions and preserve shader sorting semantics.
The 164 unresolved resource binding identities, residual shadow/DiffInd
differences and remaining input/closure/code-generation audits remain open.

## Validation, follow-up and reproduction

The full 32-thread build, host 171/171, HIP 183/183, fallback 185/185 and
focused HIP 20/20 pass. Four strict native Vulkan canaries pass, with 49
native SPIR-V compilations and no DXC/DXIL load. They cover lamp routing,
bump state, BSDF dispatch and the new closure guards; all three environment
guards are mandatory. Existing Vulkan f16/f64 remainder failures remain
separate and unresolved. Luisa source/gitlink did not change in this repair.

Six further full-resolution 256-spp renders use frozen binaries and retained
equal-pass Cycles references, not newly paired original-engine timings.
All 46 channels are finite and all 15 pass comparisons complete. The first
Combined triptych for each scene was visually inspected; visible residuals
remain, not a claim of image parity.

| Scene / run | Render seconds | Session initialization | Frame bytes | Combined relative RMSE | DiffInd relative RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Barbershop 1 | 40.2134 | 17.7810 | 416 | 0.01052115 | 0.07126167 |
| Lone Monk | 13.6060 | 43.3679 | 220 | 0.01240716 | 0.12881606 |
| Monster | 14.8859 | 49.8781 | 280 | 0.00547855 | 0.02552525 |
| Classroom | 18.2708 | 36.9795 | 260 | 0.00353320 | 0.17820333 |
| Barbershop 2 | 40.1447 | 17.7165 | 416 | 0.01052116 | 0.07126171 |
| Barbershop 3 | 40.2137 | 18.3975 | 416 | 0.01052116 | 0.07126167 |

Barbershop's median is 40.2134 s, 58.5% above retained Cycles 25.3775 s
and 0.04% above the preceding 40.1981 s. These temporal comparisons do not
isolate an implementation effect. Original Classroom's 25 invalid DiffDir
and 27 invalid GlossDir pixels are excluded by the explicit invalid-union
rule; actual output has no invalid pixels. Main shader caching is disabled,
but downstream cache policy is unchanged and the profile already warmed
Barbershop. Initialization is JIT plus setup/baking, not compiler-only or a
matched cold/warm comparison, and is excluded from render time.

Raw evidence is `/var/tmp/psycles-closure-input-guards-h5ZCFo`.
[archive_profiles.py](archive_profiles.py) takes `profile-before-Y47ymI`,
`profile-guarded-qrdgbM`, `profile-guarded-repeat-eAeU40`,
`profile-restored-iAV3kZ` and an output JSON. The static census is reproduced
by [archive_word_mix.py](archive_word_mix.py); [archive_results.py](archive_results.py)
joins the original observer, profiles, gates and scene campaign in
[results.json](results.json).

Original observer compilation uses `hipcc -parallel-jobs=32
--offload-arch=gfx1201 -DHIPCC -std=c++20 -O3 -ffast-math`, the original
`intern/cycles` include root and the committed `.hip` source. Dump-producing
Luisa commands must run **inside a fresh isolated evidence directory**:
`LUISA_DUMP_LLVM_IR=1` writes to cwd even when `LUISA_DUMP_HIP_ISA` is explicit.
During restored A this rule was missed: 45 pre-existing untracked root `.ll`
files were overwritten. New dump outputs were archived, but the old root
contents have not been recovered. The user was notified; these files are not
staged. No source was overwritten by that dump. This execution incident is
recorded rather than claiming the original artifacts were preserved.
