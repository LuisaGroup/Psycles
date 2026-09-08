# Native Math expansion and typed Barbershop SVM audit

This revision-pinned checkpoint is superseded by the
[group-context repair](../svm-group-contexts/README.md), which resolves the
remaining six layouts and supplies the latest full-resolution campaign.

## Technical result

Psycles `cbb73185` restores original Cycles' Math clamp expansion phase.
Fifteen of eighteen independent original-Cycles word images fail before the
repair; all eighteen pass exactly afterwards, without changing expected
words or applying a floating-point tolerance. Six full Barbershop materials'
opcode scheduling discrepancies disappear. This is a structural repair, not
a measured performance improvement or complete SVM parity.

Luisa remains at published `85e5300f1`. No device arithmetic, inlining policy,
register cap or shader-specific compiler option changes. The original full
surface machine-code `.text` is byte-identical after the repair. The separate
[microfacet callable A/B/A experiment](../microfacet-boundaries/README.md)
is fully reverted; the baseline microfacet bodies are already inlined.

## Formal cause and counterexamples

Original Cycles `scene/shader_nodes.cpp`, `MathNode::expand`, appends one
ClampNode after all authored nodes and native conversion links exist. It
relinks **every** Math output consumer, then connects the raw result to the
new Clamp's Value input. `ShaderGraph::simplify` runs expansion before default
inputs and cleanup. Native SVM scheduling uses creation IDs to break equal
Sethi-Ullman priority ties.

Psycles instead expanded `use_clamp` while recursively importing the Math
node, before native socket conversion reconstruction. This inverted a
conversion-versus-Clamp creation-ID tie. The scheduling ranking itself
matches Cycles and is not changed. The repair retains the Math property
through import and performs the original relink in `MathNode::expand`.
The remaining private legacy displacement prepass retains clamped semantics;
this is not an added surface execution path or completion of legacy removal.

The smallest pair has seven authored nodes including Output and Diffuse:
UV -> Multiply -> color Mix, versus Light Path Ray Depth -> Add -> Mix factor.
Only Add's `use_clamp` changes. The unclamped 60-word control was exact; the
clamped 66-word image first differed at local word 7. The larger matrix adds
Checker, groups, shared Color consumers and surface / explicit Normal /
BUMP / BOTH domains. Original Cycles, not Psycles, generates all expected
streams. Bump cloning propagates the ordering error but is not its cause.

## Complete original Barbershop word image

The production used-shader dump has 280 units, 490 dense entries and 895,648
words; 279 named shader images map to original Cycles. Every corresponding
image has equal length. Static capacity remains 33 float stack lanes and
12 closures. The new decoder uses C++ `sizeof`/`offsetof` of all 104 original
typed declarations, after verifying their mirrored bodies are identical.
It consumes closure payloads and variable tables, and checks jump targets at
decoded instruction boundaries. This is a parser, not a CPU shader oracle.

| Full-image classification | Materials | Qualification |
| --- | ---: | --- |
| Raw identical | 115 | Only ShaderJump global-to-local relocation |
| Identical node layout, differences only in declared resource-ID fields | 158 | Resource binding equivalence is still unresolved |
| Different node layout | 6 | Same node counts and word lengths do not imply same schedule |
| Other same-layout typed payload differences | 0 | Not a global floating-point/runtime parity result |

No resource ID, constant, stack address or opcode is normalized away. The
six repaired schedules still have genuine raw resource-ID differences:

| Material | Differing words before | After |
| --- | ---: | ---: |
| curtain_fabric | 249 | 10 |
| door_inside | 2,774 | 5 |
| door_inside.001 | 2,774 | 5 |
| Towel | 209 | 5 |
| globe_main | 4,144 | 5 |
| clock_backface | 217 | 5 |

The remaining layout discrepancies start as follows. Typed field annotations
at unmatched positions describe only the original, not aligned semantics.

| Material | Local word | Original node | Psycles node |
| --- | ---: | --- | --- |
| heater_mat | 1,126 | MAPPING | TEX_IMAGE |
| extinguisher_rough_copper | 1,221 | TEX_VORONOI | TEX_IMAGE |
| extinguisher_copper | 1,062 | MAPPING | ATTR_DERIVATIVE |
| wood_furniture_dresser | 16,284 | GEOMETRY | ATTR |
| wood_cupboard | 13,104 | GEOMETRY | ATTR |
| swirl_vintage | 8 | TEXTURE_MAPPING | TEX_IMAGE_BOX |

These require new original-input reductions. Neither case-specific node
renumbering nor a blanket claim of resource-only parity is justified.

## Full-module runtime and backend controls

The before run is the restored A of the preceding microfacet experiment;
the after run uses the same exported scene with the native Math repair.
Full-resolution 2048x858 / 64 spp / seed 0 profiler runs do not overlap
builds, tests or another render. They are not fresh paired Cycles wall times.

| Metric | Before | After |
| --- | ---: | ---: |
| Render wall seconds | 10.4971 | 10.4395 |
| Surface GPU seconds | 5.895572170 | 5.915631274 |
| Closest GPU seconds | 1.681304056 | 1.681565371 |
| Volume GPU seconds | 0.420247384 | 0.424453466 |
| Surface invocations | 332,307,895 | 332,307,894 |
| Coroutine stages / fields / bytes | 6 / 93 / 416 | 6 / 93 / 416 |
| VGPR / SGPR / fixed private bytes | 256 / 107 / 2,496 | 256 / 107 / 2,496 |

Stage-map, final LLVM and ELF symbols independently identify
`kernel_9bbb1beb16d04044`. Both surface `.text` sections have SHA-256
`2c0374c4975eeabb289841e2f6f9dc3aabbc02546db84cf230ab6b780da44fbb`.
Entire ELF hashes differ; byte identity is claimed only for `.text`.
The main body has 185,614 static instructions plus three outlined noise/math
helpers, unchanged. Static spill metadata is not dynamic spill traffic.
The retained original Cycles surface total is 2.962829 seconds; the roughly
twofold per-invocation surface gap remains unresolved.

All 46 actual channels are finite and all 15 before/after pass comparisons
complete. Combined relative RMSE is 0.000011725 and DiffInd 0.000047077.
These tiny intervention deltas do not establish original-Cycles parity or
waive the larger residual DiffInd differences in the full-scene controls.

| Gate | Result | Scope |
| --- | --- | --- |
| Full build | Passed | All 32 hardware threads, 312 targets |
| Host | 166/166 | 1.27 s; eighteen new exact original images |
| HIP | 182/182 | 136.42 s, complete registered suite |
| Fallback | 184/184 | 200.10 s, complete registered suite |
| Strict native Vulkan | 2/2 | 19.91 s; 31 native SPIR-V modules, no DXC/DXIL load |
| Luisa child | 155/155 retained | Unchanged child checkpoint, not rerun here |

## Four-scene 256-spp follow-up

All six full-resolution runs complete with frozen hashes of the executable
and five runtime libraries, main shader caching disabled and normal
auxiliary/downstream/OS cache policy. Geometry and texture bytes are exactly
the earlier baseline's, with the reviewed fresh socket metadata. No build,
other render or profiler overlaps these measurements. References are the
retained equal-pass original Cycles run-1 images, not fresh timing pairs.

| Scene / extent / seed | Render seconds | Session init seconds | Frame bytes |
| --- | ---: | ---: | ---: |
| Barbershop 1 / 2048x858 / 0 | 40.2016 | 26.5348 | 416 |
| Monk / 1440x1080 / 0 | 13.3807 | 19.8795 | 220 |
| Monster / 1080x1080 / 0 | 14.8904 | 23.6108 | 280 |
| Classroom / 1920x1080 / 1 | 18.3319 | 18.7251 | 260 |
| Barbershop 2 | 40.6183 | 25.5005 | 416 |
| Barbershop 3 | 40.2529 | 26.9425 | 416 |

Barbershop's median is 40.2529 seconds, 58.6% slower than retained original
Cycles' 25.3775 seconds. The 2.82% decrease from the preceding 41.4192-second
checkpoint is temporal follow-up, not an isolated causal improvement.
Monk/Monster/Classroom have one new run each. Session init is the CLI's
`shader_jit_seconds`, including setup/baking; it is not compiler-only or
cold JIT, and the earlier profile has warmed downstream caches.

All 46 actual channels are finite and all 15 pass comparisons complete in
every run. All four first-run Combined triptychs were inspected at resized
viewing resolution; this is not pixel parity. First-run residuals are:

| Scene | Combined relative RMSE | DiffInd relative RMSE |
| --- | ---: | ---: |
| Monk | 0.012400860 | 0.128817800 |
| Monster | 0.005478723 | 0.025524965 |
| Classroom | 0.003533202 | 0.178203327 |
| Barbershop | 0.010521163 | 0.071261712 |

Original Classroom contains 25 invalid DiffDir and 27 invalid GlossDir
pixels; comparisons exclude the invalid union explicitly. The complete
[results.json](results.json) retains all pass metrics, masks, output hashes,
exact commands and implementation identities. The remaining discrepancies
are not dismissed as noise or one ULP.

## Reproduction and evidence

Evidence root: `/var/tmp/psycles-svm-schedule-LaccgH`. `formal-analysis.md`
predates the repair. `red.log` retains all fifteen failures, and `green.log`
the final exact result. Original observer Blender is
`/home/mike/Projects/blender-install-psycles-trace-5.2/blender`, build
`cb168525138f`; source is `/home/mike/Projects/blender-cycles-trace-5.2`.

Generate new input graphs with `tools/create_cycles_scheduling_probe.py`,
render them through original HIP using `tools/render_cycles_golden.py` with
`PSYCLES_CYCLES_SVM_DUMP`, and export using `tools/export_psycles_scene.py`.
The committed `cycles_math_expand` fixtures retain that export and original
words, with only global shader-jump relocation. The fixture regression is
`psycles.cycles_svm_math_expand`.

Run `make_word_layouts.py ORIGINAL_SOURCE OUTPUT`, configure/build its generated
C++ program with 32 threads, and capture `layouts.json`. `audit_typed_words.py`
takes that metadata, the existing raw word audit and both unmodified binary
SVM dumps. `test_typed_words.py` takes metadata and the complete original dump:
it checks original decoding plus malformed headers, jumps, opcodes, closures,
tables and layouts. Unimplemented variable Raycast decoding fails closed.
It also checks that opcode-like table values remain data: 541 original
shader images decode and all ten additional validity/corruption checks pass.

The complete profile is `profile-1BZ5qE`. Existing
`lamp-routing-and-surface/audit_profile_followup.py` compares it with
`/var/tmp/psycles-microfacet-boundary-bWpsKZ/restored-lYIxeE`; this report's
`archive_results.py` replaces that helper's historical scope text and freezes
raw/typed word audits, all gates, source hashes and complete canary records.
Remaining resource bindings, six graph schedules, legacy displacement removal
and cross-scene correctness/efficiency remain open.
