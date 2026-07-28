# Psycles validation record — 2026-07-29

This record covers the renderer boundary on Psycles `main` and LuisaCompute
`next@0e6f4376e`. It records the commands,
numeric results, visual inspection, and known limitations for the AMD GPU
bring-up, Cycles flat-light distribution, XIR control-flow repair, and
multilayer OpenEXR output.

## Verdict

- Psycles configures and builds the Luisa fallback, HIP, and Vulkan backends
  together. All three requested backend targets are strict CMake
  postconditions rather than optional best-effort features.
- The complete Psycles gate passes 12/12 after a 32-job build.
- Luisa's focused `restructure_cfg` gate passes 51/51 tests and 1013
  assertions. The complete Luisa unit gate passes 87/88; the sole failure is
  the pre-existing EASTL `fixed_vector` allocation contract described below.
- Vulkan and HIP both render the focused flat-light scene on the Radeon RX
  9070 XT. Vulkan also passes the two transparent-closure probes.
- Psycles now writes one full-float multilayer OpenEXR with
  `ViewLayer.<pass>.<component>` channel names. The end-to-end comparison reads
  Cycles EXR and Psycles EXR directly.
- The Lone Monk `column marble` blocker is repaired without changing its raw
  closure graph: whole-scene attribute metadata is now device data rather than
  host-recorded shader control flow.
- The focused Cycles/Psycles images are visually coincident at normal display
  scale. The committed triptychs include the independently amplified absolute
  difference.

This is not a claim of complete Cycles compatibility. The focused probes are
64×64 at 256 spp. A current-Cycles Lone Monk run at 480p or 1080p and a
same-device Cycles/Psycles performance comparison remain the next full-scene
gate.

## Reference policy

Cycles is the only rendering and sampling oracle. The validation path is:

1. create or open the Blender scene;
2. render the requested linear passes with Cycles;
3. export geometry, settings, and the original node/closure graph;
4. compile and execute that graph through the Luisa DSL/JIT;
5. compare the Cycles and Psycles linear EXR channels numerically and
   visually.

There is no CPU reference renderer, CPU sampler, or host-side BSDF/light/MIS
oracle in this workflow. Blender/Cycles is not used to pre-bake materials.
Host code only normalizes immutable scene data and preserves the original
closure topology and socket values for Luisa execution.

## Revisions and machine

| Item | Validated value |
|---|---|
| Psycles renderer implementation | `a10d686` on `main`; later `main` commits pin and document the Luisa repair |
| Psycles input boundary | `32d4217dc543b1778729f23a18f3f3143e001a24` |
| LuisaCompute | `0e6f4376e` on published `next` |
| Cycles source inspected | Blender `main@9353fed6d7cdc25b2aa03c30a155b044b313c8ec`, 2026-07-28 |
| Cycles render executable | Blender 5.2.0 LTS, build hash `fbe6228777e7`, 2026-07-13 |
| OS/kernel | Arch Linux, Linux `7.1.4-zen1-1-zen` |
| CPU/build concurrency | Ryzen 9 9950X3D, 16 cores / 32 threads; build and CTest use 32 jobs |
| GPU | AMD Radeon RX 9070 XT, Navi 48 / `gfx1201` |
| Vulkan | instance 1.4.350; RADV device API 1.4.354; Mesa 26.1.5 |
| HIP | ROCm HIP runtime 7.2.4; HIP reports 7.2.53211 |
| Compiler/build tools | GCC 16.1.1, CMake 4.4.0, Ninja 1.13.2 |
| Fallback dependencies | LLVM 22.1.8, Embree 4.4.1 |
| Image dependencies | OpenImageIO 3.1.12.1, OpenEXR 3.4.13, NumPy 2.5.1, Pillow 12.3.0 |

The checked-out Cycles source is newer than the packaged Blender render
binary. Source inspection therefore follows current `main`, while the
committed pixels are explicitly a Blender 5.2.0 LTS reference. Building and
rendering the current source checkout is required before calling a future
full-scene result an exact current-`main` comparison.

## Configure, build, and unit checks

The exact three-backend release configuration was:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPSYCLES_ENABLE_LUISA_FALLBACK=ON \
  -DPSYCLES_ENABLE_LUISA_HIP=ON \
  -DPSYCLES_ENABLE_LUISA_VULKAN=ON \
  -DPSYCLES_ENABLE_OPENIMAGEIO=ON \
  -DLLVM_DIR=/usr/lib/cmake/llvm \
  -Dembree_DIR=/usr/lib/cmake/embree-4.4.1
cmake --build build --parallel 32
ctest --test-dir build --output-on-failure -j32
```

The generated cache contains all four requested feature flags and all three
Luisa backend flags as `ON`. CMake reported the HIP 7.2.53211, Vulkan, and
LLVM 22.1.8 / Embree 4.4.1 fallback backends. The final Psycles result was
12/12 tests passing in 0.42 seconds.

The new `psycles.openexr` regression writes Combined RGBA and Normal XYZ,
reopens the file through OpenImageIO, checks the seven Cycles-compatible
channel names, and compares every float value exactly. The Blender 4.5/5.2
multilayer-EXR API compatibility regression is also part of the 12-test gate.

## Luisa XIR repair

The repair is expressed in control-flow and ownership invariants rather than
scene-specific pattern patches:

- a block belongs to a structured loop region only when it is reachable
  within the region and dominated by the loop entry;
- nested Loop, SimpleLoop, and Switch constructs are atomic break scopes
  during parent-loop boundary traversal;
- for a non-trivial old loop-update region `R`, with old update `U`, prepare
  `P`, merge `M`, and canonical update trampoline `U'`, the semantics are:

  ```text
  Continue(source outside R -> U) => Branch(source -> U)
  Continue(source inside  R -> U) => Continue(source -> U')
  U'                              => Branch(U' -> P)
  ```

  This preserves execution of state updates and early Break paths in `R`;
  redirecting all continues to `U'` would incorrectly bypass them.
- a selection arm re-entered after its merge is node-split at the declared
  merge frontier, making the new entry explicit instead of repeatedly wrapping
  an invalid multi-entry construct;
- opaque ray-query state is affine. When node splitting duplicates a mutually
  exclusive initializer, its alloca is cloned and all cloned uses are remapped
  to the new storage;
- disconnected predecessors are excluded consistently from executable-CFG
  construct legality.
- an if-restructuring batch processes its entire stable candidate snapshot.
  Every successful rewrite strictly reduces the raw conditional count and
  creates no raw conditional, so progress is monotonic rather than one
  graph-wide walk per branch;
- for every emitted selection `(H, M)`, no edge may leave the region dominated
  by `M` and re-enter the `H`-dominated interior before `M`. Merge inference
  ranks ordinary common joins first and an enclosing loop-boundary convergence
  proxy second.

The relevant published Luisa commits are:

| Commit | Result |
|---|---|
| `6ead8e714` | backend resource paths work when Luisa is built as a subdirectory |
| `83a04feb8` | preserve CFG and SSA invariants during restructuring |
| `f83725d27` | preserve executable semantics for update regions, re-entry, affine state, and disconnected edges |
| `0e6f4376e` | make conditional batching monotonic, verify selection merge frontiers, and add opt-in pass tracing |

The focused and complete commands were:

```bash
./build/luisa-tests/bin/test_xir_pass_restructure_cfg
ctest --test-dir build/luisa-tests --output-on-failure -j32
```

The focused binary passes 51 tests / 1013 assertions. The complete suite
passes 87/88. `test_eastl_allocation` fails eight assertions concerning
EASTL `fixed_vector` max-size and move/overflow buffer ownership. It reproduces
when run alone and is outside the XIR files changed by these commits; it is
recorded as an existing dependency/toolchain baseline failure, not hidden or
counted as passing.

## Vulkan reset diagnosis and formal specialization

Before specialization, the opaque flat-light scene generated about 130,021
SPIR-V words, 16 loops, and 6340 Phi nodes; XIR legalization took about 3.2
seconds and RADV reported a guilty device reset. Diagnostic removal of
material evaluation in the transparent-shadow callback reduced the shader,
showing that the failure scaled with closure dispatch rather than scene
geometry or the light distribution.

The final fix keeps the original closure graphs and specializes only on the
static capability `may_be_transparent`: the
`transparent_extinction` dispatch table contains exactly the surface programs
that may contribute transparent extinction. Provably opaque programs are not
recorded into that callable. A regression uses host recording counters to
prove that an opaque surface is recorded zero times and a transparent surface
exactly once.

The final opaque shader is 61,528 SPIR-V words. It no longer resets RADV and
still uses the same material programs for all non-transparent operations.
This is JIT dead-dispatch elimination from a semantic capability, not closure
pre-baking or a value approximation.

## Render process

The focused probe command was:

```bash
python3 tools/run_cycles_shader_probes.py \
  --blender /usr/bin/blender \
  --psycles-render build/bin/psycles_render_blender_scene \
  --output-dir /tmp/psycles-exr-validation-20260729 \
  --backend vk \
  --width 64 --height 64 --samples 256 \
  flat_light_distribution \
  transparent_mix \
  transparent_data_pass
```

For each probe the runner created the `.blend`, rendered a Cycles multilayer
EXR, exported the raw scene and node graphs, rendered with Luisa/Vulkan,
wrote a Psycles multilayer EXR, compared all requested named channels, and
generated triptychs. Blender is launched with `--python-exit-code 1`, so a
Python-side failure cannot be mistaken for a passing process.

The Cycles reference script also accepts an explicit compute backend and
device-name filter. Before beginning the full-scene gate, both selection paths
were exercised at 16×16/1 spp:

```bash
/usr/bin/blender flat_light_distribution.blend --background \
  --python-exit-code 1 --python tools/render_cycles_golden.py -- \
  cpu.exr 16 16 1 0 --cycles-device CPU

/usr/bin/blender flat_light_distribution.blend --background \
  --python-exit-code 1 --python tools/render_cycles_golden.py -- \
  hip.exr 16 16 1 0 --cycles-device HIP \
  --device-name "Radeon RX 9070 XT"
```

The CPU metadata contains only
`CPU: AMD Ryzen 9 9950X3D 16-Core Processor`. The HIP metadata contains only
`HIP: AMD Radeon RX 9070 XT`, device id
`HIP_AMD Radeon RX 9070 XT_0000:03:00`, while `scene.cycles.device` is `GPU`.
This explicit device inventory is recorded beside every future golden EXR so
that a silent CPU fallback cannot be reported as a same-device comparison.

The flat distribution follows the current Cycles flat-light construction:
world-space emissive-triangle area weighting, uniform analytic/background
lamp probability, and a 50/50 class split when triangles and lamps both
exist. Selection and PDF lookup use one uploaded CDF and one Luisa upper-bound
callable. Material emission-sampling metadata and world-sampling metadata are
imported with the original material graphs.

## Lone Monk full-scene bring-up

The first complex-scene input is
`lone-monk_cycles_and_exposure-node_demo.blend`, SHA-256
`4250d4205d8d01cefd98c15e81021d6dead540b2923797378bf7b32e96e8b8f7`.
Blender 5.2 reads scene `daylight`, frame 4, camera `cam.001`. Its original
configuration is 1440×1080 at 4096 samples with adaptive sampling and
denoising enabled. Differential runs will disable the latter two features and
use one identical fixed sample count.

The raw export command was:

```bash
/usr/bin/blender \
  /home/mike/Downloads/lone-monk_cycles_and_exposure-node_demo.blend \
  --background --python-exit-code 1 \
  --python tools/export_psycles_scene.py -- \
  /tmp/lone-monk-20260729/export
```

It completed in 24.85 seconds and produced 350 geometries, 7,543 evaluated
instances, 35 original material graphs, and 47 images. `geometry.bin` is
450,966,096 bytes and `scene.json` is 10,834,008 bytes. The largest material
has 38 nodes. The set includes 32 Principled nodes and real combinations of
Image Texture, Noise, Mapping, Mix, Color Ramp, Bump, Normal Map, Light Path,
Transparent, Glossy, and Translucent nodes. Geometry/modifier/particle
evaluation is permitted; material closure evaluation remains in Luisa and is
not pre-baked by Blender or Cycles.

The first 64×48/1 spp Vulkan smoke used the default optimization and persistent
shader-cache settings. The auxiliary shader compiled, Vulkan selected
`AMD Radeon RX 9070 XT (RADV GFX1201)`, and the scene allocated about 5.0 GiB
VRAM. The main render kernel did not produce a cache entry or first pixel
within 1,245.51 seconds and was deliberately interrupted at the documented
20-minute bound. It consumed about 198% CPU and was sampled at 9,749,995,520
bytes RSS.

A diagnostic Vulkan run set:

```bash
LUISA_XIR_DISABLE_OPTIMIZATION=1
LUISA_SPIRV_OPT_LEVEL=0
```

The auxiliary SPIR-V changed from 11,221 optimized words to 14,927
unoptimized words, proving the switches took effect. The main kernel retained
the same approximately two-thread, multi-gigabyte code-generation profile and
did not reach a cache artifact or first pixel in 311.18 seconds. This narrows
the dominant cost to the giant render-kernel code-generation boundary rather
than an optional XIR or SPIR-V optimization pass alone.

The equivalent HIP smoke selected the RX 9070 XT through HIP 7.2.53211.
HIPRT then remained in acceleration-structure construction for 622.45 seconds
at 100% observed GPU utilization and about 5.5 GiB VRAM, without reaching the
main render-kernel JIT. It was interrupted at the documented ten-minute bound.
HIP acceleration-structure construction and Vulkan monolithic-kernel JIT are
therefore recorded as separate engineering issues.

Opt-in XIR tracing then reduced the problem without changing geometry. A
six-original-material controlled input retained all 350 geometries and 7,543
instances. The old one-rewrite-per-batch implementation spent approximately
113.29 seconds and still emitted an invalid selection: a merge-dominated path
branched back into the selection interior. The invariant-based batch and merge
repair passes its two red/green regressions and renders the same controlled
input. Vulkan scene compilation took 0.648155 seconds, main-kernel JIT
22.5549 seconds, and 64×48/1 spp rendering 0.00485903 seconds; the 497,049-word
SPIR-V passed validation and Psycles wrote PPM, PFM passes, and multilayer EXR.

The decoded image was inspected at original resolution. It contains finite
image data without a full-frame NaN, Inf, or exposure failure, but its sparse
one-sample appearance and reduced material set make it a compiler/first-pixel
smoke only. The committed preview and inspection record are in the
[Lone Monk investigation](docs/validation/2026-07-29/lone-monk/README.md).

An eleven-material controlled input reached 50,020 blocks and 2,205,924
instructions before restructuring and was interrupted at 180 seconds. Five
single-material trials isolated the nonlinear lowering to `column marble`.
That material alone reaches 45,900 blocks, 1,543,437 instructions, and 15,266
raw conditional branches; its graph includes a 257-sample Color Ramp, Bump,
and two RGB Curves. The other four isolated materials compile and render in
3.35–18.73 seconds.

Reducing the Color Ramp and both RGB Curve tables from 257 samples to two did
not change the `column marble` XIR counts, disproving sampled-table cardinality
as the cause. Disconnecting Base Color reduced the graph to 900 blocks and
137,567 instructions, while retaining only the Base Color dependency kept
45,420 blocks and 1,500,987 instructions. That path reaches Vertex Color.
Lone Monk has 367 named UV layers plus 12 color attributes. The old attribute
service recorded one `$if` for each of these 379 whole-scene bindings every
time a material callable used an attribute.

The formal code-size invariant is that, for a fixed shader and fixed number of
attribute lookup operations, recorded AST/XIR control-flow size is independent
of scene attribute-table cardinality. The repair uploads a flat binding table
and one compact range per geometry. Luisa reads only the current geometry's
range through bindless buffers and performs one device loop whose predicate
contains a `found` state. No material node, link, lookup sample, or closure was
removed or pre-baked.

The regression records the real shader service and translates it to XIR. A
512-binding old implementation failed the constant selection-count bound; the
device-table implementation stays at or below eight structured selections.
The original `column marble` graph now enters restructuring with 1,360 blocks,
203,797 instructions, 406 raw conditionals, and 10 indexed branches. XIR
restructuring completes in 10.47494 seconds, Vulkan JIT in 12.3495 seconds,
and the 432,610-word shader renders 64×48/1 spp in 0.00485309 seconds.
Psycles' complete 12-test gate passes with 32-way scheduling.

The complete commands, stage timings, per-material results, test counts, and
machine-readable measurements are in the
[Lone Monk bring-up report](docs/validation/2026-07-29/lone-monk/bringup.json).
The 40-channel diagnostic EXR contains no NaN or Inf values, and its
[decoded preview](docs/validation/2026-07-29/lone-monk/column-marble-vulkan-smoke.png)
was inspected at original resolution. No Lone Monk triptych is fabricated from
this single-material smoke. The required Cycles / Psycles /
amplified-linear-difference triptychs remain pending until the unmodified
material set produces comparable EXRs. The already committed focused
triptychs continue to test the comparison pipeline.

## Numeric comparison

All tables compare linear float channels. Relative RMSE is RMSE divided by the
Cycles RMS for that pass.

### Flat light distribution

| Pass | RMSE | Relative RMSE | Luminance ratio | Maximum error | Invalid pixels |
|---|---:|---:|---:|---:|---:|
| Combined | `0.000256542553` | `1.294632%` | `1.000089120` | `0.001330458` | 0 |
| DiffDir | `0.000557016116` | `0.904113%` | `1.000055316` | `0.002803124` | 0 |
| DiffCol | `0` | `0` | `1` | `0` | 0 |
| Normal | `0` | `0` | `1` | `0` | 0 |

Machine-readable result:
[flat-light report](docs/validation/2026-07-29/flat-light-vk/report.json).

Triptychs:
[Combined](docs/validation/2026-07-29/flat-light-vk/triptychs/combined.png),
[Diffuse Direct](docs/validation/2026-07-29/flat-light-vk/triptychs/diffdir.png),
[Diffuse Color](docs/validation/2026-07-29/flat-light-vk/triptychs/diffcol.png),
and [Normal](docs/validation/2026-07-29/flat-light-vk/triptychs/normal.png).

### Transparent closure mix

| Pass | RMSE | Relative RMSE | Luminance ratio | Maximum error | Invalid pixels |
|---|---:|---:|---:|---:|---:|
| Combined | `0.000000793035` | `0.000183525%` | `0.999997026` | `0.000001326203` | 0 |
| Emit | `0` | `0` | `1` | `0` | 0 |
| Env | `0` | `0` | `1` | `0` | 0 |
| Normal | `0` | `0` | `1` | `0` | 0 |

Machine-readable result:
[transparent-mix report](docs/validation/2026-07-29/transparent-mix-vk/report.json).

Triptychs:
[Combined](docs/validation/2026-07-29/transparent-mix-vk/triptychs/combined.png),
[Emission](docs/validation/2026-07-29/transparent-mix-vk/triptychs/emit.png),
[Environment](docs/validation/2026-07-29/transparent-mix-vk/triptychs/env.png),
and [Normal](docs/validation/2026-07-29/transparent-mix-vk/triptychs/normal.png).

### Transparent data passes

| Pass | RMSE | Relative RMSE | Luminance ratio | Maximum error | Invalid pixels |
|---|---:|---:|---:|---:|---:|
| Combined | `0.000179912153` | `0.321085%` | `0.999992053` | `0.001019992` | 0 |
| DiffDir | `0.001175414538` | `0.409927%` | `0.999992597` | `0.006618649` | 0 |
| DiffCol | `0` | `0` | `1` | `0` | 0 |
| Normal | `0` | `0` | `1` | `0` | 0 |

Machine-readable result:
[transparent-data report](docs/validation/2026-07-29/transparent-data-pass-vk/report.json).

Triptychs:
[Combined](docs/validation/2026-07-29/transparent-data-pass-vk/triptychs/combined.png),
[Diffuse Direct](docs/validation/2026-07-29/transparent-data-pass-vk/triptychs/diffdir.png),
[Diffuse Color](docs/validation/2026-07-29/transparent-data-pass-vk/triptychs/diffcol.png),
and [Normal](docs/validation/2026-07-29/transparent-data-pass-vk/triptychs/normal.png).

## Visual inspection

The Cycles and Psycles panels in each triptych share one diagnostic linear to
sRGB mapping and one exposure scale. Normal uses the fixed
`normal * 0.5 + 0.5` mapping. The third panel is absolute linear difference
scaled by the reciprocal of its 99.5th percentile; the multiplier is printed
in the title. Thus the left/right appearance is directly comparable while the
third panel deliberately makes very small errors visible.

The inspection result is:

- flat-light Combined and DiffDir are visually coincident. The amplified
  difference is per-pixel sampling noise whose amplitude follows scene
  radiance; it has no geometry-edge displacement, missing light region, or
  coherent color bias;
- transparent-mix Combined is visually identical. At approximately
  `6.79e5×` amplification, the remaining difference is a spatially uniform
  green-channel rounding offset;
- transparent-data Combined is visually identical. At approximately
  `1.32e3×` amplification, the difference is unstructured per-pixel sampling
  noise with no silhouette or systematic shading pattern;
- the committed exact DiffCol and Normal triptychs have black difference
  panels.

No orientation search is used to select a result. EXR is compared in its
top-left scanline order; legacy PFM support applies its fixed format-defined
vertical conversion. The reports retain identity and flipped-orientation
diagnostics as a guard against accidental format changes.

## Focused GPU timings

These are small 64×64/256 spp probes and are useful for shader-size and
backend health only.

| Backend/probe | Scene compile | Shader JIT | Render |
|---|---:|---:|---:|
| Vulkan flat, cold cache | `0.025667 s` | `0.559935 s` | `0.015133 s` |
| Vulkan flat, hot cache | `0.015175 s` | `0.032178 s` | `0.014824 s` |
| HIP flat, cold cache | `0.303026 s` | `0.974313 s` | `0.009619 s` |
| HIP flat, repeat | `0.043706 s` | `0.355552 s` | `0.009915 s` |
| Vulkan transparent mix, hot | `0.013557 s` | `0.036953 s` | `0.012637 s` |
| Vulkan transparent data, hot | `0.019546 s` | `0.019263 s` | `0.013890 s` |

The focused Cycles metadata selected CPU execution, so these figures do not
support a same-device speedup claim. The next full-scene measurement must run
Cycles and Psycles on the same RX 9070 XT, report cold and warm setup
separately, use at least 480p, and attempt 1080p when GPU memory permits.

## Known limitations and next gate

- Render and compare Lone Monk with current Cycles and Psycles on the same AMD
  GPU at 480p or 1080p. Record device selection, samples, total wall time,
  render-only time, peak memory, pass metrics, and triptychs.
- Build the 2026-07-28 Cycles source checkout, or move both source inspection
  and pixels to another exact common revision. Do not silently mix a newer
  source claim with the packaged 5.2.0 LTS binary.
- The flat distribution is implemented, but Cycles light trees are not.
- Environment map importance CDFs and the imported
  `world_sample_map_resolution` are not yet connected to sampling.
- Automatic emissive sampling classification still needs a formal
  Cycles-aligned static analysis; no host pre-evaluation workaround is
  acceptable.
- Imported per-light MIS metadata is preserved, but the remaining
  visible-light forward-MIS path is not complete.
- Complex demo scenes must expand the material, volume, displacement,
  subdivision, motion, denoising, and pass coverage beyond these focused
  probes before any 1:1 feature/quality claim.
