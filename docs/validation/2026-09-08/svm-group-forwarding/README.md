# Blender socket forwarding and Barbershop surface follow-up

## Technical summary

Psycles `6f743862`, with unchanged published Luisa `85e5300f1`, preserves
Blender's linked-versus-primitive socket semantics through group boundaries,
reroutes and muted internal links. It also separates Blender primitive luma
and Gamma folding from Cycles' later folding. The repair has 34 new original
Cycles 5.2.1 probes and passes the complete host/HIP/fallback gates and strict
native Vulkan canary. It changes host graph construction, not device math,
fast math, inlining policy, interpreter dispatch, RNG or coroutine boundaries.

All 279 named used Barbershop shader images now have the same lengths as
Cycles; 115 are raw-equal. This is not complete structural or semantic parity.
Some equal-length images still have different opcode ordering. The complete
surface machine-code text remains byte-identical to the preceding checkpoint,
and the large per-invocation surface cost remains unresolved.

## Formal source-category boundary

Original `source/blender/nodes/intern/shader_nodes_inline.cc` distinguishes
`LinkedSocketValue`, `PrimitiveSocketValue` and `InputSocketValue`.
`handle_implicit_conversion` forwards a linked producer without applying the
group interface's conversion. `set_input_socket_value` eventually links that
producer to the real consumer; Cycles introduces the conversion for those two
native socket types. Primitive values, including convertible input defaults,
are converted at the intermediate typed boundaries.

The old normalizer eagerly inserted Color -> Float -> Color at a Float group
interface even for a linked Color producer and Color consumer. It lost color
and emitted six extra words. Deleting inverse conversions in the late Cycles
optimizer would be incorrect: Color -> Float is lossy. Ignoring all group
conversions would also be incorrect for primitive values.

An authored RGB node is numerically constant but is **not** a Blender primitive
at this phase: original `node_shader_rgb.cc` registers no multi-function.
All-constant White Noise does have a multi-function and produces a primitive.
The original v1 reduction exposed this distinction; its old `primitive` mode
name and 21 failures are retained as evidence, while v2 calls it `rgb-node`.

The contract now records import origin, includes it in structural hashing and
validates it during native projection. Generated input values enter the
primitive phase; authored RGB nodes remain linked. A temporary forwarded-source
edge survives primitive propagation and TextureMapping property recovery.
Native socket projection resolves it and introduces the ordinary Cycles
ConvertNode for the final producer/consumer. The temporary metadata is then
cleared before cleanup, deduplication, bump cloning and compilation. It never
becomes an SVM opcode or device-side flag.

### Two different host luminance and Gamma semantics

Blender primitive Color-to-Float conversion uses
`IMB_colormanagement_get_luminance`, with OCIO `getDefaultLumaCoefs`.
Cycles' later `ShaderManager::rgb_to_y` instead derives its coefficients from
the RGB/XYZ transform. The exporter now retains the former independently as
`shader_transforms.blender_luma`; the importer defaults to the original
0.2126/0.7152/0.0722 fallback when this new field is absent. The controlled
benchmark bundles use that fallback, matching the verified original config.
Independent exporter/importer controls check a changed luma configuration
without changing the XYZ rows.

All-primitive Gamma uses original Blender `safe_pow`: negative input, or zero
with a non-positive exponent, returns the input. It does not use Cycles' later
Gamma==0 all-ones fold. Six original exponent -1/0/2 controls separate authored
RGB from all-primitive Combine XYZ; Blender clamps the authored RGB probe's
negative input at creation, whereas the primitive negative input is retained.
No device Gamma implementation or slow bit-matching path changes.

## Original regressions and full module

The original observer-only Blender `cb168525138f` renders and dumps the probes
on HIP. `create_cycles_group_forward_probe.py` creates graphs, not expected
words. The existing fixture extractor copies raw exported graphs and original
words, relocating only the three ShaderJump targets.

| Probe family | Cases | Final result |
| --- | ---: | --- |
| Linked Checker, authored RGB, computed White Noise; eight routes each | 24 | Original staged semantics |
| Unlinked group defaults; four routes | 4 | Original staged conversions |
| Authored / primitive Gamma, three exponents each | 6 | Original phase-dependent folds |
| Total | 34 | 33 raw-exact; one typed-literal roundoff difference |

The one difference is three ULPs in the two RGB payloads of
`rgb-node-actual-float`. The expected stream is unchanged. Its comparator allows
four ULPs only in six finite float fields of the recognized constant-Diffuse
typed grammar. Opcodes, jumps, stack inputs/addresses, flags and padding remain
exact. Mutation checks reject a five-ULP change and a change to every structural
word; non-finite/tagged stack payloads are not accepted as literal tolerance.
Existing fixture comparators remain unchanged unless this explicit callback is
passed. This is a test policy, not a performance-sacrificing renderer change.

| Full Barbershop used-shader audit | `33c7b335` | `6f743862` |
| --- | ---: | ---: |
| Entire raw image equal | 115 | 115 |
| Equal length, different raw words | 163 | 164 |
| Unequal length | 1 | 0 |
| Global word storage | 895,654 | 895,648 |
| Static stack / closure bound | 33 floats / 12 | 33 floats / 12 |

The production used-material compiler handles 280 units / 490 dense indices;
279 images have unique used names. The six extra `bricks` words disappear.
Neither resource IDs nor literal payloads are normalized by the comparator.
In particular, six equal-length images already differ at the guaranteed first
node boundary: original ATTR versus Psycles LIGHT_PATH. These include
`curtain_fabric`, both `door_inside` variants, `Towel`, `globe_main` and
`clock_backface`. Their scheduling/projection differences need further
reduction; equal word counts cannot exempt them from the structural audit.

## Final machine code and measured kernel work

The full Barbershop profile uses 2048x858, 64 spp, seed 0 and the identical
controlled exported bundle. The stage map identifies surface
`kernel_9bbb1beb16d04044`, independently verified against LLVM and ELF symbols.
The comparison is a temporal profiler follow-up, not a fresh paired Cycles run.

| Measure | `33c7b335` | `6f743862` |
| --- | ---: | ---: |
| Surface GPU seconds | 5.871503748 | 6.093110071 |
| Closest-intersection GPU seconds | 1.669304023 | 1.707738357 |
| Volume GPU seconds | 0.418396027 | 0.437522628 |
| Profiled render wall seconds | 10.5310 | 10.8669 |
| Surface invocations | 332,307,890 | 332,307,893 |
| Surface launches | 1,062 | 1,062 |
| Stages / frame fields / frame bytes | 6 / 93 / 416 | 6 / 93 / 416 |
| Surface VGPR / SGPR / fixed private bytes | 256 / 107 / 2,496 | 256 / 107 / 2,496 |

The complete `.text` sections are byte-identical, SHA-256
`2c0374c4975eeabb289841e2f6f9dc3aabbc02546db84cf230ab6b780da44fbb`.
The main function has 995,980 bytes / 185,614 static instructions, 1,469
scratch-load sites, 610 scratch-store sites and 49 call sites. The only final
outlined functions are 1D noise, 4D signed noise and OCML tangent. SVM dispatch
and closure handlers are already inlined; pre-HIPRTC LLVM function definitions
alone do not establish a final out-of-line call.

There is no measured performance gain here. The newer profile is slower,
including the unchanged closest/volume kernels; without interleaved controls
this is not a causal regression estimate either. The original retained Cycles
ordinary-surface GPU sum is 2.962828646 s. Its main function excludes outlined
BSDF/noise bodies, so main-function size is not a transitive code-size metric.
Static spill sites and metadata are not dynamic scratch traffic or stalls.
The earlier more-outlined A/B/A experiment remains reverted; this repair adds
no inline/noinline override or register limit.

All 46 profile channels are finite. Against the preceding Psycles image,
Combined relative RMSE is 0.000011551 and DiffInd is 0.000046970; both volume
passes are unchanged. These are intervention controls, not Cycles parity.

## Verification and reproduction

### Six full-resolution 256-spp follow-ups

All six runs use frozen `6f743862` binaries, native fast math and disabled
main shader caching. Six implementation files are hashed before the campaign
and after every run. Geometry/images, authored seed and all 15 passes remain
fixed. No build, profiler, heavy test or other renderer overlaps these runs.
The reference images/timings are retained original-Cycles equal-pass results,
not fresh timing pairs. All 46 actual channels are finite; all 15 comparisons
complete, and all four first-run Combined triptychs were visually inspected.

| Scene / run | Extent / seed | Render seconds | Session init seconds | Frame bytes |
| --- | --- | ---: | ---: | ---: |
| Barbershop 1 | 2048x858 / 0 | 41.1692 | 31.8932 | 416 |
| Lone Monk 1 | 1440x1080 / 0 | 13.3832 | 23.4175 | 220 |
| Monster 1 | 1080x1080 / 0 | 14.8542 | 24.0953 | 280 |
| Classroom 1 | 1920x1080 / 1 | 18.7549 | 21.3365 | 260 |
| Barbershop 2 | 2048x858 / 0 | 41.4192 | 28.7845 | 416 |
| Barbershop 3 | 2048x858 / 0 | 41.4301 | 28.4156 | 416 |

Barbershop's median is 41.4192 s, range 41.1692–41.4301 s: 1.6321 times the
retained Cycles median 25.3775 s, and 2.97% above the preceding Psycles median
40.2231 s. This is not an isolated causal regression estimate; it certainly
does not show a speedup. The other scenes each have one new run, respectively
1.0106 / 1.0397 / 1.0401 times their retained Cycles medians.

Session initialization is CLI `shader_jit_seconds`: JIT plus setup/baking,
not compiler-only. It is separate from render time and scene compilation.
It also increases in this temporal follow-up. The preceding IR/profile run
already used the same binaries, and downstream/auxiliary/OS caches retain
normal policy; these are not cold-JIT measurements. A fresh interleaved control
is needed before attributing either timing movement to this host repair.

First-run Combined relative RMSE for Monk / Monster / Classroom / Barbershop
is 1.2405% / 0.5479% / 0.3533% / 1.0521%; DiffInd is 12.8817% / 2.5525% /
17.8203% / 7.1262%. These unresolved differences are not dismissed as sampling
noise or evidence by themselves of extra paths. Original Classroom retains
25 invalid DiffDir / 27 invalid GlossDir pixels; comparisons exclude the union
of invalid pixels explicitly. No actual pixels are invalid.

### Backend gates

| Gate | Result | Elapsed seconds |
| --- | ---: | ---: |
| Complete host, including source-size gate | 165/165 | 1.40 |
| HIP | 182/182 | 143.91 |
| fallback | 184/184 | 88.41 |
| Strict native Vulkan lamp-routing + bump-state | 2/2, 31 native modules | 19.93 |
| Unchanged child `85e5300f1` | Retained 155/155, not rerun here | — |

All builds use 32 threads. HIP runs before fallback, then Vulkan with all
three native-XIR guards and dynamic-loader inspection excluding DXC/DXIL.
Host tests use `-E '_(hip|fallback|vk)$'`; backend tests use the respective
`_hip$` / `_fallback$` suffixes. Earlier partial host logs and the intermediate
failed build are retained, not relabeled as final complete gates.

Evidence root: `/var/tmp/psycles-svm-group-forward-BBpoLx`.
The immutable v2 original dump has SHA-256
`682a2e914ceb813a0c033c27b45e26b06c3ef1325f43f16ef0d8400cef6cc788`.
The current [results archive](results.json) retains original inputs/images,
red/green logs, final gates, code objects, profile, all pass comparisons and
their source hashes. [scene-words.json](scene-words.json) retains the whole
used-shader comparison. Regenerate the archive using:

```sh
python docs/validation/2026-09-08/svm-group-forwarding/archive_results.py \
  /var/tmp/psycles-svm-group-forward-BBpoLx \
  docs/validation/2026-09-08/svm-group-forwarding/results.json
```

Remaining questions include equal-length scheduling/payload differences,
incomplete Blender primitive folding and muted-unlinked-default coverage,
remaining native opcodes and legacy displacement cleanup, residual DiffInd
and shadow-work differences, the surface code-generation cost and the child
CFG audit's open formal obligations. This checkpoint does not close those
goals or claim complete Cycles compatibility.
