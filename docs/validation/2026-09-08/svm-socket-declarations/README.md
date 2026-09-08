# Native socket declarations and closure dependency order

Frozen `33c7b335` checkpoint. The [subsequent Blender forwarding repair](../svm-group-forwarding/README.md)
removes the remaining `bricks` length mismatch and records newer complete
gates and four-scene measurements. The results below remain revision-pinned.

## Technical summary

Psycles `33c7b335995e3e39abc21f4010c50b5f01a06eb1`, with unchanged published
Luisa `85e5300f1`, restores native texture POINT inputs, type-pair conversion
identity and Cycles closure declaration order. It also fixes the native
Voronoi prototype's unavailable Smoothness default. There are 53 new exact
original-Cycles word-image cases; the complete Barbershop used-shader audit
now has only one unequal-length image, `bricks` (+6 words).

The preceding `322a3855` commit separates existing color tests, an AST visitor
and surface fixture constructors into cohesive test modules. All original
function bodies/assertions are retained. ConvertNode now has its own ordinary
translation unit. The complete host suite, including the unwaived source-size
gate, passes. Neither commit changes Luisa inlining policy, device arithmetic,
fast math, runtime node dispatch, feature masks, RNG or coroutine boundaries.

## Why matching float3 storage was not enough

Original `scene/shader_nodes.cpp::ConvertNode::get_node_types()` registers a
different node type for each `(from, to)` pair. COLOR, VECTOR, POINT and NORMAL
have component-preserving float3 conversions, but remain distinct graph types.
In particular, sharing FLOAT-to-COLOR with a FLOAT-to-POINT consumer changes
the graph that Cycles schedules and clones for bump evaluation.

The contract expresses scalar-to-vector as FLOAT -> COLOR -> VECTOR. The
projector formerly retained its replication into COLOR, even when the native
consumer required POINT. A real Color consumer then shared that replication
instead of retaining the original separate FLOAT-to-POINT node. A single
evaluation was short by three words; authored bump cloned the discrepancy
three times and was short by nine. This explains the four damaged-label
graphs in Barbershop, without any shader-name-specific rule.

For the projection-created terminal conversion, the normal form is
`FLOAT -> COLOR -> VECTOR -> T` to `FLOAT -> T`, where `T` is a float3 type.
Only scalar replication and component identities are composed; neither a
lossy color-to-scalar nor an integer conversion is included. Existing Color
consumers keep their edges. The native ConvertNode's equality also checks the
destination type, preserving Cycles' original type-pair identity despite the
shared internal factory tag.

Precise POINT declarations are restored for the whole supported texture-input
family, including Noise, White Noise, Gradient, Gabor, Voronoi, Image and
Environment, alongside the previously restored IES/Wave/Magic/Checker/Brick
inputs. This is not a claim that every remaining non-texture socket's exact
Cycles type has been audited.

## Projection must respect the original phase boundary

Blender primitive propagation runs before Cycles constructs its native graph.
A newly inserted POINT converter must not prevent that earlier constant
propagation. Likewise, a contract helper encoding a texture's internal
TextureMapping property is not a separate source MappingNode. It must be
recovered into the texture before native conversion nodes are introduced.

The initial typed repair exposed both problems in the complete host suite
and prevented the original Barbershop module from compiling. Restoring the
types after Blender function folding and TextureMapping property recovery,
but before native defaults/cleanup/bump refinement, resolves those failures.
Existing Gradient/White Noise literal and all TextureMapping oracle words
remain unchanged and pass. Here “legacy TextureMapping” denotes Blender's
per-texture mapping property, not an alternate surface executor.

The Image/Environment test builder also used an unlinked vector value where
its original word fixture clearly requires a linked constant coordinate.
Cycles replaces a genuinely unlinked hidden coordinate with UV/Position.
The builder now supplies its constant through Combine XYZ, as a primitive
producer; every expected word remains unchanged. The separate default-UV
test is retained.

## Closure declaration order controls stack lifetimes

Both original and Psycles `SVMCompiler::generate_closure_node` traverse the
node's inputs one at a time, generating each input's unscheduled dependency
graph before moving to the next. Matching the scheduling algorithm is therefore
insufficient when the input declarations have a different order.

For example, original Diffuse declares Color, Normal, SurfaceMixWeight, then
Roughness; the contract used Color, Roughness, Normal, then the added mix
weight. This reordered Geometry.Normal and the texture dependency, changing
word and stack addresses. The projector now follows the original declaration
for each supported surface closure, including the larger Principled and
Metallic lists. It validates the complete input-name set before ordering it;
unknown/missing inputs reject projection instead of silently being ignored.
The reviewed volume declarations already have the required relative order.

The unrelated equal-length Voronoi control localized another prototype
difference: native `Smoothness` defaults to 5, while the Blender UI value is
1. When that socket is unavailable, original `blender/shader.cpp` skips
`set_default_value`, retaining 5. The importer already skips unavailable
inputs; the contract prototype now contains the correct native value. An
available authored value still overrides it normally.

## Original counterexamples and full-scene word audit

Original references come from observer-only Cycles 5.2.1 LTS `cb168525138f`.
The probe builders create Blender graphs only. The original HIP backend
renders and dumps them; the extractor retains their raw exported inputs and
relocates only the three ShaderJump addresses. No CPU renderer or expected
word generator is involved.

| Family | Cases | Before repair | Final |
| --- | ---: | --- | --- |
| Nine texture kinds, scalar-single/shared/bump | 27 | 18 length failures plus one Voronoi payload failure | 27 exact |
| Thirteen closure kinds, default/linked inputs | 26 | Nine linked-input images fail | 26 exact |
| Conversion type-pair identity | One host invariant | POINT and COLOR incorrectly compare equal | Distinct; same-type pair still equal |

No new floating-point tolerance is used. The earlier separate Vector Math
literal tolerance is unchanged. The 53 images join the preceding 105 images
across the hidden-input, host-fold, bump-alias and texture-output families.

| Barbershop used-shader comparison | `5096a41f` | `33c7b335` |
| --- | ---: | ---: |
| Entire raw image equal | 112 | 115 |
| Same length, different raw words | 162 | 163 |
| Different length | 5 | 1 |
| Static stack / closure bound | 33 floats / 12 | 33 floats / 12 |
| Global word storage | 895,618 | 895,654 |

There are 280 compile units, 279 uniquely named used shaders and 490 dense
indices. The extra 36 words restore the four label graphs' missing conversions.
Resource IDs and all literal payloads remain unmodified in the comparator;
equal lengths are not semantic parity. `bricks` still has an eager group-boundary
Color-to-Float-to-Color conversion, requiring a separate original regression
and linked-versus-primitive provenance analysis.

## Surface cost is still not explained by missing inlining

The new full Barbershop profile retains the identical input bundle and seed,
2048x858 at 64 spp and all 15 passes / 46 finite channels. Stage names are
resolved from the runtime shader map, not inferred from code-object file
numbers. `shade_surface` is `kernel_9bbb1beb16d04044`; its selected code object
is `hip_isa_12.co`. These are profiler follow-ups, not new paired Cycles wall
timings. The preceding profile uses `5096a41f`.

| Measure | `5096a41f` | `33c7b335` |
| --- | ---: | ---: |
| Surface GPU seconds | 6.033289805 | 5.871503748 |
| Closest-intersection GPU seconds | 1.699354368 | 1.669304023 |
| Volume GPU seconds | 0.430163245 | 0.418396027 |
| Profiled render wall seconds | 10.7462 | 10.5310 |
| Surface invocations | 332,307,894 | 332,307,890 |
| Surface launches | 1,062 | 1,062 |
| Surface VGPR / SGPR | 256 / 107 | 256 / 107 |
| Surface fixed private bytes | 2,496 | 2,496 |
| Coroutine stages / fields / bytes | 6 / 93 / 416 | 6 / 93 / 416 |

The complete surface ELF `.text` sections are byte-identical, with SHA-256
`2c0374c4975eeabb289841e2f6f9dc3aabbc02546db84cf230ab6b780da44fbb`.
The main function still has 995,980 bytes and 185,614 static instructions,
1,469 scratch-load sites, 610 scratch-store sites and 49 call sites. The
three surviving outlined functions are 1D noise, 4D signed noise and OCML
tangent. SVM dispatch and closure handlers are already inlined. Pre-HIPRTC
LLVM definitions alone are not evidence that a function remains outlined
in the final machine code.

These facts do not establish a causal speedup from the shorter temporal
profile. They show that the repair changes interpreted program data, not
the machine code or resource allocation of the broad interpreter. Surface
invocations differ by four in 332 million; excessive surface invocations are
not the explanation for this checkpoint's remaining per-invocation cost.
The retained original Cycles ordinary surface kernel totals 2.962828646 GPU
seconds, versus the new Psycles 5.871503748. That selected Cycles main function
excludes its outlined BSDF/noise functions; comparing main-function sizes
alone would not compare transitive shader code size. Static scratch/spill
sites are not measured dynamic traffic or stalls. PC sampling is unavailable
on this GPU/tool setup.

The preceding full-scene A/B/A experiment retained more HIP callable
boundaries and made surface GPU time 35.5% worse; it remains reverted.
This checkpoint adds no forced inline/noinline policy, register limit or
slow floating-point path. The much larger surface cost remains open for
structural and code-generation investigation, not attributed to the few
remaining noise/math calls without a counterexample and measured control.

Against the preceding Psycles profile image, Combined relative RMSE is
0.000013465 and DiffInd is 0.000057525; all volume passes are unchanged.
Those are intervention controls, not original-Cycles parity measurements.

## Six full-resolution 256-spp follow-ups

The six runs use frozen `33c7b335` binaries, native fast math and disabled
main shader caching. Six implementation binaries are hashed before the
campaign and after every run. The controlled bundles retain the prior exact
geometry/image bytes with the new source socket metadata. No profiler,
build, heavy test or concurrent renderer overlaps these runs. References are
retained original-Cycles equal-pass images, not newly paired timings.

| Scene / run | Extent / seed | Render seconds | Session init seconds | Frame bytes |
| --- | --- | ---: | ---: | ---: |
| Barbershop 1 | 2048x858 / 0 | 40.2231 | 24.9958 | 416 |
| Lone Monk 1 | 1440x1080 / 0 | 13.4347 | 18.8892 | 220 |
| Monster 1 | 1080x1080 / 0 | 14.7923 | 22.0922 | 280 |
| Classroom 1 | 1920x1080 / 1 | 18.2949 | 17.9811 | 260 |
| Barbershop 2 | 2048x858 / 0 | 40.2419 | 24.9958 | 416 |
| Barbershop 3 | 2048x858 / 0 | 40.1987 | 24.7800 | 416 |

Barbershop's median is 40.2231 s, with range 40.1987–40.2419 s. This is
1.65% below the previous checkpoint's 40.8982 s, but not an isolated causal
speedup estimate. It remains 1.5850 times the retained Cycles median of
25.3775 s. The other scenes each have one new measurement, not repeated
performance estimates. The large Barbershop efficiency gap remains.

Session initialization is the CLI's `shader_jit_seconds`: JIT plus setup and
baking, not compiler-only. The earlier profile initialized in 26.0744 s with
IR/profiler output and ran before these canaries. Auxiliary/downstream/OS
caches retain normal policy; these initialization times are not cold JIT.
Scene compilation is separately retained for every run in the archive and
is not included in either render time or session initialization.

| First-run image comparison | Combined relative RMSE | DiffInd relative RMSE |
| --- | ---: | ---: |
| Lone Monk | 0.01240456 | 0.12881620 |
| Monster | 0.00547872 | 0.02552497 |
| Classroom | 0.00353320 | 0.17820333 |
| Barbershop | 0.01052115 | 0.07126171 |

All six actual EXRs have exactly 46 finite channels; all 15 pass comparisons
complete. The first four Combined triptychs were inspected at resized viewing
resolution, with no obvious new layout/material regression. This is not
pixel parity. The structural DiffInd/path differences remain open. Original
Classroom has 25 invalid DiffDir and 27 invalid GlossDir pixels; the affected
metrics explicitly exclude the invalid union. Nearly empty passes retain
their absolute errors and reference signal scale in [results.json](results.json).

## Verification

| Gate | Result | Scope |
| --- | --- | --- |
| Full build | Passed | All 32 hardware threads |
| Host | 164/164 | Including the unwaived source-size gate and split color tests |
| HIP | 182/182 | Complete registered suite, 135.13 s |
| Fallback | 184/184 | Complete parallel suite, 86.99 s |
| Strict native Vulkan | 2/2 | 31 native SPIR-V compilations, 20.87 s, no DXC/DXIL load |
| Luisa child | 155/155 retained | Unchanged `85e5300f1`, not rerun at this checkpoint |

The native gate sets `LUISA_VULKAN_USE_XIR=1`,
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1` and `LUISA_VULKAN_DISABLE_DXC=1`.
The source-size gate now has no exceptions or remaining violations: the
oversized compiler-node and test files were separated by semantic ownership,
not by deleting assertions or relaxing the threshold.

Evidence is retained under `/var/tmp/psycles-svm-socket-types-d3rAJv`.
The two immutable original dump hashes are:

- Texture inputs: `681ca0deb86666f8bc45fc63a6fb78748541da9542647c6525b9330d16ba175d`.
- Closure inputs: `6c3936b3999e2b5a17cdb75256fdfcea541f1ffd54e6751d72490a9fad256b1d`.

Use `tools/create_cycles_socket_types_probe.py` and
`tools/create_cycles_closure_inputs_probe.py`, render/dump with the original
observer's HIP backend, export the same scenes, then use the preceding
report's `extract_hidden_socket_fixture.py` with counts 27 and 26. The complete
original Barbershop used-material population is recompiled after the repairs.

The [per-shader word audit](scene-words.json) keeps unmodified resource IDs,
constants, stack addresses and domain counts. `archive_results.py` verifies
the original red/green probe outputs, full gate totals, native compiler/loader
route, complete six-run order, binary and source hashes, and finite/pass
contracts before generating [results.json](results.json). Intermediate failed
builds/tests are retained as phase-history evidence, not overwritten by green
logs. Regenerate with:

```bash
python docs/validation/2026-09-08/svm-socket-declarations/archive_results.py \
  /var/tmp/psycles-svm-socket-types-d3rAJv \
  docs/validation/2026-09-08/svm-socket-declarations/results.json
```

The requested goal is not complete: group-boundary linked/primitive semantics,
remaining equal-length word differences, indirect/path alignment, unsupported
native opcodes, private legacy displacement removal and the large surface
efficiency gap still require work. Neither the new exact probes nor all-green
registered suites certify those unimplemented boundaries.
