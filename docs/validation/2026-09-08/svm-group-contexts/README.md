# Lazy group inputs and persistent instance contexts

Psycles `239cade6` (pushed to `origin/main`) removes the final six known node-layout differences
in the complete original Barbershop SVM table. All 279 used named shader
images now have identical node layouts, stack addresses, typed non-resource
payloads and domain/jump structure. Of these, 115 are raw-exact and 164 differ
only in declared image/attribute ID fields. **Resource binding equivalence is
not established by that classification.** No IDs or expected words are
normalized away. This is a structural result, not a measured speedup or full
shader parity.

## Cause and invariants

Original Blender 5.2.1 `shader_nodes_inline.cc:640-674` follows a requested
group output into its cached instance compute context. A Group Input then
follows the corresponding parent instance input in the parent context.
Memoization is per `(context, socket)`. It does not eagerly evaluate every
input on the instance, and two instances of one definition are not recursive
nesting merely because one depends on the other.

Psycles previously evaluated every instance input before entering the group,
guarded recursion with a set of definition names across those calls, and
discarded the child output caches after each requested output. Two concrete
failures follow:

1. Unused input producers are created early. Later native cleanup can merge
   them with live producers while retaining their earlier creation IDs.
   Those IDs break Sethi-Ullman ties, changing native scheduling and stack
   assignment. Removing the dead nodes at the end is too late.
2. A dependency on an upstream sibling instance of the same definition is
   misclassified as recursion and replaced by a default, losing live work.

The repair keeps one context per `(parent context, group instance)` and
resolves only requested group inputs in the parent. Output and shared-output
caches persist for that instance. A lexical guard restores the current
context on every return and exception. Recursive definitions are checked
against the current context's ancestors, not suspended sibling calls; true
socket cycles are guarded by active `(context, socket, source type)` keys.
Linked versus primitive conversion and surface/volume closure-domain keys
are unchanged. No SVM ranking, node-ID patch, device arithmetic, compiler
inline policy or Luisa change is introduced.

## Original counterexamples

`tools/create_cycles_group_liveness_probe.py` creates sixteen original graphs:
direct/group/nested/chain, surface/bump, and connected/disconnected ignored
input controls. Eight fail before the repair. Four have the same 89/229-word
length but wrong order; four chains collapse to 22 words.

`tools/create_cycles_group_context_probe.py` adds fourteen original controls
for minimal chains, nested instance contexts, multiple outputs and separate
instances with different defaults. The smallest failing material has five
authored nodes: Light Path Ray Depth -> two instances of one passthrough
group -> Diffuse Roughness -> Output. Original Cycles emits 25 words; old
Psycles emits 22. Four additional controls fail on a replay of the unchanged
pre-fix normalizer. All thirty images pass raw-exact after restoring the
repair. Neither fixture changes after its recorded red run.

The original full `heater_mat` was first reduced with original Cycles HIP
after each graph change. Image-to-Checker reduction must preserve shared
source-image identity; replacing each image with a unique Checker hides the
witness. Top-level and nested single-link deletion reaches a 232-word
witness with a connected but internally unused group input. This is
one-link-minimal in that reduction scope, not a global minimality proof.
The preceding default-coordinate hypothesis had 32 raw-exact controls and
did not justify a source fix. See the
[reduction history](../svm-default-coordinates/README.md).

| Complete Barbershop shader | Differing words before | After |
| --- | ---: | ---: |
| heater_mat | 954 | 10 |
| extinguisher_rough_copper | 107 | 3 |
| extinguisher_copper | 1,633 | 9 |
| wood_furniture_dresser | 3,782 | 20 |
| wood_cupboard | 3,944 | 15 |
| swirl_vintage | 188 | 11 |

Every remaining difference in this table is in a declared resource-ID field.
The production table retains 895,648 linked words, 33 statically required
float stack lanes and twelve closures. It includes the complete production
used-material domain, not only visible or hand-selected materials.

## Complete module, profile and high-resolution validation

The full 32-thread rebuild passes. Host tests pass 168/168 (1.26 s), HIP
182/182 (145.30 s), fallback 184/184 (91.40 s), then strict native Vulkan
2/2 (23.69 s), with 31 SPIR-V modules and no DXC/DXIL load. All three native
guards are recorded. Luisa `85e5300f1` is unchanged; its prior 155/155 child
suite is retained, not represented as rerun here.

| Complete Barbershop 2048x858 / 64 spp / seed 0 | Before | After |
| --- | ---: | ---: |
| Render wall seconds | 10.5088 | 10.4753 |
| Surface GPU seconds | 5.904181622 | 5.872749375 |
| Closest GPU seconds | 1.689932574 | 1.682760276 |
| Volume GPU seconds | 0.423224443 | 0.419618622 |
| Surface invocations | 332,307,894 | 332,307,892 |
| Coroutine stages / fields / bytes | 6 / 93 / 416 | 6 / 93 / 416 |
| VGPR / SGPR / private bytes | 256 / 107 / 2,496 | 256 / 107 / 2,496 |

The surface ELF `.text` is byte-identical to before, SHA-256
`2c0374c4975eeabb289841e2f6f9dc3aabbc02546db84cf230ab6b780da44fbb`.
No speedup is established by these single sequential profiler controls.
Surface GPU time remains 1.982 times the retained original Cycles 2.962829 s,
with essentially unchanged work. Main SVM and microfacet bodies remain
inlined; only the previously identified noise/math helpers are outlined.
The remaining per-invocation cost is not explained by this host graph fix.
All 46 channels are finite; all 15 pass controls complete. Combined and
DiffInd relative RMSE against before are 0.0000185473 and 0.0000574817,
respectively, not measurements of original-Cycles parity.

Stage attribution exposed a diagnostic assumption: LLVM dump 13 and code
object 12 both contain `kernel_9bbb1beb16d04044` in this run. The counters are
independent. The shared profile inspector now selects before/final LLVM and
ELF by the actual stage entry symbol instead of hard-coding dump 12, and
checks the notes/disassembly entry too. Four permanent parser tests cover
different counters, missing entries and ambiguous LLVM/ELF matches. The
profile archive has been regenerated with the correct sources; no measured
kernel times or resources depend on the old filename assumption.

Six new, frozen-binary 256-spp runs complete overnight September 8-9 without
any overlapping build, test, profiler or render. They use the same scene
geometry/textures and source socket metadata as the previous campaign.
These are follow-ups against retained equal-pass Cycles references, **not
fresh timing pairs**. Main shader caching is disabled; downstream caches
retain normal policy and have been warmed by the earlier profile.

| Scene / repeat | Extent / seed | Render seconds | Session init seconds | Frame |
| --- | --- | ---: | ---: | ---: |
| Barbershop 1 | 2048x858 / 0 | 40.1352 | 25.9988 | 416 B |
| Lone Monk | 1440x1080 / 0 | 13.3748 | 19.2910 | 220 B |
| Monster | 1080x1080 / 0 | 14.7691 | 22.4903 | 280 B |
| Classroom | 1920x1080 / 1 | 18.2831 | 19.4370 | 260 B |
| Barbershop 2 | 2048x858 / 0 | 40.0687 | 26.5395 | 416 B |
| Barbershop 3 | 2048x858 / 0 | 40.9465 | 25.9027 | 416 B |

Barbershop's median is 40.1352 s, 58.2% slower than the retained original
25.3775 s, and just 0.29% below the previous 40.2529 s: no isolated causal
gain is claimed. Monk/Monster/Classroom are 1.0100/1.0338/1.0139 times the
retained Cycles medians. Session init includes JIT plus setup/baking; it is
neither compiler-only nor a cold-cache measurement, and is separate from
the render interval.

All six outputs have 46 finite channels and all 15 pass comparisons. First
Combined relative RMSE for Monk/Monster/Classroom/Barbershop is
0.012411261 / 0.005478723 / 0.003533202 / 0.010521149; DiffInd is
0.128816658 / 0.025524965 / 0.178203327 / 0.071261658. The four first-run
Combined triptychs were viewed. The original Classroom reference retains 25
invalid DiffDir and 27 invalid GlossDir pixels; comparisons explicitly
exclude the union of invalid pixels. Residual DiffInd is not excused as
sampling noise. Resource binding identity, surface execution cost and
residual path differences remain open.

## Reproduction and evidence

Evidence root: `/var/tmp/psycles-group-lazy-inputs-oYf8iF`. It retains the
pre-fix `formal-cause.md`, full intervention patch, v1/v2 original probes,
minimal/context probe, HIP EXRs, exports, raw dumps, and red/green/build logs.
The v1 probe's identical MixRGB endpoint defaults accidentally fold one
branch away; its weaker result is retained, not presented as the v2 witness.

Run each generator with the original observer Blender and a new `.blend`
path. Render with `PSYCLES_CYCLES_SVM_DUMP=OUTPUT.svm52` and
`tools/render_cycles_golden.py -- OUTPUT.exr 16 16 1 0 --cycles-device HIP`,
then export that same blend with `tools/export_psycles_scene.py`. Always use
Blender `--python-exit-code 1`. Extract the unchanged original graphs/words
with `lamp-routing-and-surface/extract_hidden_socket_fixture.py`, using
stems/counts `cycles_group_liveness`/16 and `cycles_group_context`/14.
`psycles.cycles_svm_group_liveness` checks both permanent fixtures.

The published Math report supplies the fail-closed typed word decoder; this
report uses it without changing its resource-field classification. Original
Cycles remains the only oracle. No CPU shader evaluator is introduced.

Run `archive_results.py EVIDENCE results.json --implementation 239cade6` to
rebuild the [validation archive](results.json), raw and typed shader tables.
The archive validates all gates and the six-run pass/hash contract before
writing results. The full profile is `profile-fNYE2I`; run the sibling
`audit_profile_followup.py BEFORE AFTER` first. Parser attribution tests are
`lamp-routing-and-surface/test_profile_artifacts.py`.
