# Lamp routing, CFG restructuring and native surface cost

This investigation fixes real control-flow and imported-graph differences.
It does **not** establish Cycles performance parity. In particular, the
experiment that retained more HIP function boundaries was slower and has
been completely reverted. No inlining-policy change is shipped here.

This is the `cf59ab5b` / `85e5300f1` checkpoint. The subsequent
[graph-boundary repair](../svm-graph-boundaries/README.md) resolves the extra
Geometry.Normal and repeated procedural-texture evaluations diagnosed below,
restores BUMP displacement tails, and carries the newer measurements.

## Published control-flow corrections

Psycles `773f1aca` / Luisa `4284e8cb9` are published to `origin/main` /
`origin/next`. They include the lamp-routing regression and the generic
loop-epoch restructuring repair, including the published child gitlink.

- Original Cycles traces a closest intersection again after each transparent
  lamp endpoint, without advancing the outer path step or RNG dimension.
  The coroutine now yields at that same traversal boundary. Saved BSSRDF
  exits retain their separate route.
- A native Vulkan miss exposed a second application-level mistake:
  `committed_ray_t` is not defined for a miss. The unified traversal boundary
  now assigns the requested ray maximum explicitly on misses, including
  mixed and shadow queries. This is not an assumption added to Luisa's ABI.
- Original Cycles GPU code supplies eight visibility/depth configurations
  and 43 complete paths. Permanent tests cover monolithic rendering and
  staged SoA/AoS frames, repeated with a 19-path capacity.

The [child CFG audit](../../../../third_party/LuisaCompute/docs/validation/2026-09-08/loop-scope-restructure/README.md)
contains the formal phase preconditions, 11-block counterexample, permanent
crossing-epoch rejection test, transactional guarantees, and original-module
replay. The exact original native input has SHA-256
`c8f1adaad0b72690a0d00d0da34ef11f8252e7367d63eed1ee2262b85c9b93ae`.
Its largest function has 1,685 blocks and 57,534 instructions. This module,
not a smaller sibling, now restructures successfully.

The root cause was selection-merge inference inside an unrecovered loop
epoch. Exit repair repeatedly converted a real loop edge into a nonlocal
exit, grew new selector funnels, and exposed another apparent loop.
Recovering natural loops first fixes the valid input. A finite discharged
`(header, parent header)` relation bounds `fixup_construct_exits` and rejects
repeated obligations; it is not an increased iteration/size/time budget.
Both callers propagate failure. A failed transaction leaves original blocks,
instructions and constants unchanged. The complete pass is **not claimed
machine-proven**; remaining ownership/dominance/alias proof obligations are
listed in the child report.

At that published checkpoint: full 32-thread root/child builds; Psycles HIP
182/182, fallback 184/184; child unit 153/153; 100 repeated loop regressions;
strict native Vulkan lamp/traversal canaries. Host tests were 156/157, with
four existing source-size violations, not waived.

## Is surface slow because functions were not inlined?

The surface stage was identified from the actual shader map, not a guessed
module ordinal. The 64-spp full Barbershop capture is
`kernel_96eb3744a6e940ba`, final LLVM module 12. Module 6 is a shadow stage.
The main interpreter and most node handlers are **already inlined**.

| Actual surface code object | Psycles | Original Cycles ordinary surface |
| --- | ---: | ---: |
| VGPRs | 256 | 192 |
| Private segment, bytes | 2,496 | 6,976 |
| VGPR spill metadata | 500 | 2 |
| SGPR spill metadata | 65 | 0 |
| Selected main function bytes | 996,532 | 441,580 |

These are static code-generation facts, **not dynamic spill traffic**.
The original textual instruction counts included alignment padding and have
been removed from this table; the
[current auditor](../../2026-09-09/hip-native-remainder/README.md) bounds counts
by ELF function extents. Archived JSON retains its historical counting scope.
Cycles' row selects `integrate_surface<1979>` and excludes outlined callees;
it is not the total transitive shader size. Cycles outlines noise and BSDF
setup, whereas Psycles' object contains only the main kernel, a 1-D noise
helper, 4-D signed noise and OCML tangent. RX 9070 XT does not advertise
rocprof PC-sampling support, so no dynamic instruction-stall attribution is
claimed. The original compiler input, actual ELF notes, function symbols,
ISA, stage maps and reconciled trace/stat totals are recorded in
[codegen-inspection.json](codegen-inspection.json).

After normal IPO and ABI projection, Luisa's existing
`inline_unique_oversized_generated_callables` mechanically inlines private
single-use generated functions whose ABI exceeds 32 argument/return VGPR
locations. It invokes LLVM's inlining legality machinery, not its cost
model. In an isolated replay it inlines two functions / 387 argument VGPR
locations; kernel IR grows from 19 blocks / 1,293 instructions to 3,948 /
135,705. This replay is explanatory, not an end-to-end speed measurement.

The full-scene A/B/A intervention skipped **only** this late helper in B.
ABI projection/demotion, native fast math, normal LLVM/HIPRTC optimization
and all renderer code remained enabled. No `noinline`, register cap, or
shader-specific compiler option was added.

| 2048x858, 64 spp, seed 0 | A: existing | B: retain boundaries | A: restored |
| --- | ---: | ---: | ---: |
| Render wall, seconds | 10.5397 | 12.6654 | 10.5676 |
| Surface GPU sum, seconds | 5.865505 | 7.961985 | 5.883198 |
| Closest GPU sum, seconds | 1.668429 | 1.674923 | 1.666124 |
| Surface private bytes | 2,496 | 5,040 | 2,496 |
| Surface visits | 332,307,917 | 332,307,834 | 332,307,922 |

B is about 35.5% slower in surface GPU time and 20% slower in render wall
time. Fewer static spill sites did not make it faster. A's code-object hash
matches the restored A exactly. All 46 channels are finite; all 15 pass
control comparisons are retained in [inline-experiment.json](inline-experiment.json).
Minor stochastic/traversal image differences between repeated controls are
recorded, not described as bit-exact. These are profiler runs, not fresh
Cycles timing pairs. The intervention is rejected and the child checkout
is restored to its published contents.

## Imported SVM differences and repairs

These implementation/regression changes and the new published child gitlink
are in Psycles `cf59ab5b`, pushed to `origin/main`.

`dump_scene_svm.cpp` links the production used-material collector and native
compiler. It is a diagnostic compiler driver, **not an oracle or CPU
renderer**. The comparison uses the observer dump from original Cycles
5.2.1, not words generated by Psycles. Only global ShaderJump addresses are
relocated. Image/attribute IDs, all literals and all other words remain
unmodified; different resource-manager numbering is not yet a semantic bug.

Barbershop has 280 compile units / 279 uniquely named used shaders, in a
490-entry dense table. The original global dump contains 541 shaders,
including unused ones. The world and renderer-authored default surface have
explicit documented name/index mappings in `audit_scene_words.py`.

| Raw used-shader comparison | Before | After hidden-input / Vector Math repair |
| --- | ---: | ---: |
| Entire local image equal | 80 | 100 |
| Same length, differing words | 86 | 115 |
| Different length | 113 | 64 |
| Psycles global words | 816,907 | 815,999 |
| Static stack / closure capacity | 36 floats / 12 | 36 floats / 12 |

Two concrete causes were established with new original-Cycles probes:

1. Blender's `SOCK_HIDE_VALUE` is semantic input provenance.
   `ShaderNodesInliner::set_input_socket_value` leaves an input unlinked
   when both it and the originating unlinked `InputSocketValue` hide their
   values. The exporter omitted this flag; group lowering materialized a
   zero vector instead. Export it and preserve the origin through groups,
   reroutes and Blender-vector type aliases. Materialized operations and
   actual type conversions do not retain that origin. Explicitly linked
   zero vectors remain linked values. `umbrella_metal` now matches Cycles'
   complete 24-word image; `blinds_plastic` matches its 83-word image.
2. `VectorMathNode` omitted Cycles' host constant fold, dynamic-operand
   identity rules and linear-operation classification. Port the complete
   30-operation family and original rule set. Blender all-primitive folding
   and later Cycles graph folding remain separate phases: Blender's
   near-zero normalization guard is not Cycles' exact-zero guard. Both are
   host graph compilation; no slow arithmetic is added to device shading.

Independent compiler regressions retain 15 hidden/default/linked/nested/
conversion/Bump graphs and 66 Vector Math graphs, including six near-zero
boundary cases, nonzero cross-product cancellation, and a projection whose
nonzero direction has an underflowed squared length. The last two prevent
incorrectly deleting a closure by folding its weight to zero. Blender's
existing high-precision cross product is used only for immutable host
constants; the device algorithm is unchanged. The former compares every
word exactly. The latter permits
`2e-6 * max(1, abs(reference))` only in the three typed emission-weight
float literals of constant probes and treats differing NaN payload bits as
the same invalid category; finite/infinite/NaN classification, all structure
and dynamic-input records are exact. This does not relax device-state or
runtime image tolerances.

At this checkpoint, differing lengths were not waved away. Missing
separate displacement tails must be distinguished from surface work;
some large total-image deficits do not occur on the executed surface entry.
Splitting at the original ShaderJump domain targets shows that 223 used
surface/bump spans have equal lengths, 48 have three extra words, `bricks`
has six extra, `pages_mat` twelve, and `wall_tiles` / `wall_tiles.001` 84
extra each. Four damaged-label shaders have nine fewer words. These are
serialized span sizes, not measured executed-node counts. The raw comparison
is not a claim that all materials, resource identities or control flow are
fully aligned; both whole-image and per-domain counts remain in the JSON.
The subsequent graph-boundary report supersedes these outstanding-count
claims; the frozen data here remains the before-image for that repair.

This does not claim Blender's complete primitive-propagation phase is
ported. Upstream Value/Color/conversion propagation and other folding
families still require their own original-source phase audit; applying every
Cycles algebraic identity early would not reproduce Blender's inliner.

## Final compiler/backend verification

Both full builds use all 32 hardware threads. The final native implementation
passes HIP 182/182, fallback 184/184 and the native Vulkan lamp-routing plus
bump-state tests (2/2), with all three native-XIR guards and no DXC/DXIL load.
Host tests pass 158/159, including the 81 new original-Cycles material images.
The remaining failure is the unwaived source-size gate for four existing
files: `cycles_svm_nodes.cpp` (2026 lines), `test_cycles_svm_compiler.cpp`
(2088), `test_luisa_compact_surface_preparation.cpp` (2114), and
`test_luisa_cycles_svm.cpp` (2038). This is not an entirely green host suite.

A repeated parallel fallback run exposed a separate generic lost-wakeup
bug. Its 183/184 completed tests were not reported as a successful suite.
The [child queue audit](../../../../third_party/LuisaCompute/docs/validation/2026-09-08/fallback-queue-wakeup/README.md)
contains formal condition-variable reasoning and independent one/two-worker
counterexamples using the production queue, without any shader. Both wait
predicate publications now acquire their associated mutex. Child
`85e5300f1` is published to `origin/next`: 155/155 tests, both new races
repeated 100 times each. The parent full fallback rerun passes 184/184 in
89.67 seconds, and the original camera sampling test passes 100 repetitions.
No scheduler policy, frame layout or device arithmetic is changed by this fix.

## Four-scene 256-spp follow-up

The implementation is Psycles `cf59ab5b` / Luisa `85e5300f1`. All six
Psycles runs completed without concurrent builds, tests, other renders or
profiling, using the controlled inputs described below. Commands, six frozen
binary hashes, source/export/geometry hashes, all 15 pass comparisons,
all 46 channel checks and the preceding six post-routing runs are retained
in [validation-results.json](validation-results.json).

| Scene / run | Extent | Render seconds | Session init seconds | Scene compile seconds | Frame bytes / fields / stages |
| --- | --- | ---: | ---: | ---: | --- |
| Barbershop 1 | 2048x858 | 40.2248 | 79.0785 | 15.1464 | 416 / 93 / 6 |
| Lone Monk 1 | 1440x1080 | 13.4514 | 18.6520 | 5.05374 | 220 / 55 / 4 |
| Monster 1 | 1080x1080 | 14.8482 | 22.0809 | 1.42613 | 280 / 70 / 6 |
| Classroom 1 | 1920x1080 | 18.3248 | 17.9154 | 2.05248 | 260 / 65 / 5 |
| Barbershop 2 | 2048x858 | 40.2962 | 23.3234 | 14.9962 | 416 / 93 / 6 |
| Barbershop 3 | 2048x858 | 40.3063 | 24.6326 | 15.0221 | 416 / 93 / 6 |

Barbershop's new median is 40.2962 s versus 40.3670 s after the preceding
lamp-routing fix: only 0.18% lower, **not evidence of a meaningful speedup**.
It is still 1.5879x the retained Cycles median, 25.3775 s. The three other
scenes have one new run each, not new three-run medians or paired Cycles
timings. Their retained Cycles medians are 13.2425 / 14.2865 / 18.0319 s.
The latest input/compiler repairs do not close the performance gap.

Session initialization includes JIT plus setup/baking. Barbershop's first
79.0785 s observation is retained alongside its 23.3234 / 24.6326 s repeats;
downstream cache warming is not presented as a compiler optimization.
Frame sizes do not change from the post-routing checkpoint. The older
matched-pair Monster/Classroom 284/264 B frames are not current frame sizes.

| First follow-up error | Combined relative RMSE | DiffCol | DiffInd |
| --- | ---: | ---: | ---: |
| Barbershop | 0.01052120 | 0.001597753 | 0.07126170 |
| Lone Monk | 0.01240527 | 0.000545932 | 0.12881543 |
| Monster | 0.00547872 | 0.000109916 | 0.02552497 |
| Classroom | 0.00353320 | 0.000102324 | 0.17820333 |

All actual channels are finite. Four first-run Combined triptychs were
visually inspected at reduced viewing resolution; no new gross scene/layout
regression is visible, but the existing image differences remain. The
original Classroom reference's 25 DiffDir / 27 GlossDir non-finite pixels
are retained with the comparator's explicit invalid-union exclusion, not
replaced or silently treated as valid output.

### Post-repair kernel check

After all unprofiled runs completed, a separate full-scene 64-spp profile
rechecked the actual stage map and code object. The current surface kernel
is `kernel_fe65567e53db5725` (LLVM module 12, verified against ELF symbols),
not the earlier control's `kernel_96eb3744a6e940ba`.

| Measured GPU sum, seconds | Pre-repair restored control | Current |
| --- | ---: | ---: |
| Surface | 5.883198 | 5.867683 |
| Closest intersection | 1.666124 | 1.672009 |
| Volume | 0.419831 | 0.419152 |
| Render wall, profiled | 10.5676 | 10.5422 |

Surface launches remain 1062; visits are 332,307,922 versus 332,307,894.
Current surface resources remain 256 VGPRs / 107 SGPRs / 2496 private bytes.
VGPR spill metadata falls from 500 to 494, but measured surface time changes
by only -0.26%. Its main function is 995,980 bytes, with the same three
outlined noise/math helpers. This does not
support missed SVM inlining as the remaining explanation. The retained
original Cycles ordinary-surface GPU sum is 2.962829 s; the gap is still
about 1.98x at this stage, not attributable to a large excess of stage visits.
It does not yet distinguish dynamic stack/closure handling, instruction
selection and within-stage work. Hardware PC sampling is unavailable here.

[profile-followup.json](profile-followup.json) retains both profiles' exact
counts, reconciled trace/stat totals, ELF/IR/ISA hashes and all 15 finite
image-control comparisons. The old Cycles stage profile remains explicitly
retained evidence in the [descriptor audit](../hip-texture-descriptors/README.md),
not represented as a freshly rerun Cycles profile.

## Reproduction and measurement boundaries

Source: original Cycles tree `blender-cycles-trace-5.2` at
`cb168525138fecc792cc393f94afc39582b0103c`. The read-only SVM observer build is
Blender 5.2.1 `cb168525138f`; production HIP is 5.2.1 `9e2066aef7ef`.
The two original trees/checkouts outside this worktree are not development
targets and were not edited.

Create scenes with `tools/create_cycles_hidden_socket_probe.py` and
`tools/create_cycles_vector_fold_probe.py`. Run the observer Blender with
`PSYCLES_CYCLES_SVM_DUMP=/new/path/image.svm52`,
`tools/render_cycles_golden.py`, `--cycles-device HIP`, and 32 threads.
Export with `tools/export_psycles_scene.py`. The fixture extraction script
copies raw material/group graphs and original words, dropping only unused
geometry. It never generates expected words from the implementation.

Retained raw evidence:

- `/var/tmp/psycles-lamp-routing-sCLxKs`: CFG/lamp validation, LLVM/ELF/ISA,
  A/B/A controls and six post-routing 256-spp renders.
- `/var/tmp/psycles-hidden-socket-U0KI5L`: probe sources, original HIP dumps,
  red/green tests, fresh four-scene exports, controlled bundles and follow-up
  verification. Final report artifacts identify exact files and hashes.

Fresh exports have identical JSON except socket flags/exporter identity,
but Blender's evaluated geometry bytes can differ between exports. The
performance controls therefore use `prepare_socket_control.py`: fresh
verified JSON plus byte-identical earlier geometry/images. No scene default
or shader node is fabricated. This controls the input change without
introducing slow intersection arithmetic to force bitwise geometry parity.

Render-only wall time, initialization (JIT plus setup/baking), scene
compilation and profiled GPU sums are separate measures. Four-scene
follow-ups use retained equal-pass Cycles images; they are not new Cycles
timing pairs. Native fast math remains on and no build/other render overlaps
a timed run.
