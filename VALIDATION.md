# Psycles validation record — 2026-07-29

This record covers the renderer boundary on Psycles `main` and LuisaCompute
`next@d57720955`. It records the commands, numeric results, real triptychs,
original-resolution visual inspection, and known limitations for the AMD GPU
bring-up, Cycles differential rendering, XIR control-flow repair, and
multilayer OpenEXR output.

## Verdict

- Psycles configures and builds the Luisa fallback, HIP, and Vulkan backends
  together. All three requested backend targets are strict CMake
  postconditions rather than optional best-effort features.
- The complete Psycles gate passes 13/13 after a 32-job build.
- Luisa's focused `restructure_cfg` gate passes 51/51 tests and 1013
  assertions. All 21 structural SPIR-V tests pass, and the RX 9070 XT Vulkan
  runtime gate passes 86/86 tests / 2029 assertions. The current complete
  CTest gate passes 115/116; the sole failure is the pre-existing EASTL
  `fixed_vector` allocation contract described below.
- Vulkan and HIP both render the focused flat-light scene on the Radeon RX
  9070 XT. Vulkan also passes the two transparent-closure probes.
- Psycles now writes one full-float multilayer OpenEXR with
  `ViewLayer.<pass>.<component>` channel names. The end-to-end comparison reads
  Cycles EXR and Psycles EXR directly. Both files identify their RGB values as
  `lin_rec709_scene`.
- The Lone Monk `column marble` blocker is repaired without changing its raw
  closure graph: whole-scene attribute metadata is now device data rather than
  host-recorded shader control flow.
- A module-wide SPIR-V argument-layout ordering defect exposed by outlined
  read-only resource callables is repaired and covered by a red/green Vulkan
  subview regression. The complete 35-material Lone Monk export now reaches a
  strict-native Vulkan first pixel from an empty shader cache.
- The current matched Lone Monk gate is complete at 1440×1080, 256 fixed spp,
  seed zero, using locally built Blender/Cycles 5.3 Alpha
  `main@4fe17ef6`. Cycles HIP and Psycles Vulkan both selected the RX 9070
  XT. Combined RMSE is `0.216918692`, relative RMSE `0.135484421`, and all
  13 compared passes have zero invalid pixels.
- A monolithic 256-spp Vulkan dispatch first triggered the AMDGPU watchdog.
  Psycles now uses a formally exact ordered partition with at most 8 spp per
  synchronized dispatch. Luisa Release builds now reject every non-success
  Vulkan result instead of silently discarding device loss.
- The focused Cycles/Psycles images are visually coincident at normal display
  scale. The committed focused and Lone Monk triptychs include independently
  amplified absolute differences and were inspected at original resolution.

This is not a claim of complete Cycles compatibility. At 1440×1080/256 spp,
Lone Monk's Combined relative RMSE remains `0.1355`; diffuse and glossy
indirect mean luminance remain about 8.63% and 6.07% low. Psycles render-only
throughput is currently `0.7295×` current Cycles on the same GPU, or
`1.3708×` slower. The high-sample pass evidence now replaces the former
pending 1080p gate and defines the next transport-alignment work.

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
| Psycles renderer implementation | `dcb96e3` on published `main` |
| Psycles input boundary | `32d4217dc543b1778729f23a18f3f3143e001a24` |
| LuisaCompute | `d57720955` on published `next` |
| Cycles source inspected | clean Blender `main@4fe17ef6be5d46251fa5e7dbff9018efb1c719d5`, fetched 2026-07-29 |
| Current Cycles render executable | locally built Blender 5.3.0 Alpha Release, hash `4fe17ef6be5d`, built 2026-07-29 |
| Historical 640×480 executable | Blender 5.2.0 LTS, build hash `fbe6228777e7`, built 2026-07-15 |
| OS/kernel | Arch Linux, Linux `7.1.4-zen1-1-zen` |
| CPU/build concurrency | Ryzen 9 9950X3D, 16 cores / 32 threads; build and CTest use 32 jobs |
| GPU | AMD Radeon RX 9070 XT, Navi 48 / `gfx1201` |
| Vulkan | instance 1.4.350; RADV device API 1.4.354; Mesa 26.1.5 |
| HIP | ROCm HIP runtime 7.2.4; HIP reports 7.2.53211 |
| Compiler/build tools | GCC 16.1.1, CMake 4.4.0, Ninja 1.13.2 |
| Fallback dependencies | LLVM 22.1.8, Embree 4.4.1 |
| Image dependencies | OpenImageIO 3.1.12.1, OpenEXR 3.4.13, NumPy 2.5.1, Pillow 12.3.0 |

The committed 1440×1080 pixels come from the locally built current checkout.
The older 640×480 section remains explicitly labeled as a packaged Blender
5.2.0 historical baseline and is not presented as current-`main` output.

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

The `psycles.openexr` regression writes Combined RGBA and Normal XYZ, reopens
the file through OpenImageIO, checks the seven Cycles-compatible channel
names, verifies `oiio:ColorSpace` and `colorInteropID` are both
`lin_rec709_scene`, and compares every float value exactly. This prevents the
unstable `scene_linear` OCIO role from relabeling Rec.709 pixels as ACEScg.
The Blender 4.5/5.2 multilayer-EXR API compatibility regression is also part
of the 12-test gate.

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
| `5cf0c548d` | preserve declared loop merge boundaries |
| `30602e640` | preserve uniquely rooted read-only resource callables instead of duplicating them into the kernel |
| `eb167454a` | freeze the module argument layout before callable emission and add the nonzero-subview ABI regression |
| `d57720955` | check every Vulkan result in Release builds and add a forced-`NDEBUG` device-loss regression |

The focused and complete commands were:

```bash
./build/luisa-tests/bin/test_xir_pass_restructure_cfg
ctest --test-dir build/luisa-tests --output-on-failure -j32
```

The focused binary passes 51 tests / 1013 assertions. With the current
expanded build, the complete suite passes 115/116. `test_eastl_allocation`
fails eight assertions concerning EASTL `fixed_vector` max-size and
move/overflow buffer ownership. It reproduces when run alone and is outside
the XIR/SPIR-V files changed by these commits; it is recorded as an existing
dependency/toolchain baseline failure, not hidden or counted as passing.

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

Canonical scene performance runs use the fixed Cycles CPU/HIP plus
Psycles-Luisa fallback/HIP/Vulkan matrix documented in
[docs/scene-benchmark.md](docs/scene-benchmark.md). The five entries share
the same source scene, final-render export, seed, extent, and fixed sample
count. Fallback is always retained in the report, including when it is slower
than the GPU paths.

The first completed five-renderer Lone Monk result, including cold-stage
timings, all numeric reports, original-resolution inspection, and committed
triptychs, is in
[docs/validation/2026-07-30/lone-monk-five-way](docs/validation/2026-07-30/lone-monk-five-way/).

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

Preserving uniquely rooted read-only resource callables then reduced the
`column marble` module to about 238 thousand words and 1.31 seconds of JIT,
but the first outlined result differed from the known-good pre-preservation
Psycles output by RMS `0.0483804`, maximum `7.020724`, over 383 pixels.
SPIR-V optimization level 2 preserved the error, as did exhaustive SPIR-V
function inlining, placing the defect before the driver-call boundary.

The cause was a module ABI initialization-order violation. SPIR-V callable
post-order emits callees before the kernel, while the direct-buffer metadata
offset had been initialized only inside kernel emission. An outlined
read-only callable therefore loaded its buffer subview metadata from argument
word zero. Luisa now freezes one validated kernel argument-layout plan before
emitting any function and asserts that the kernel observes the same immutable
layout. The regression uses a nonzero buffer subview, a scalar argument that
moves the metadata trailer, and a real `OpFunctionCall`; it failed before the
repair and now returns `{18, 29, 40, 51}`. The historical cold
`column marble` result at `next@eb167454a` is pixel-exact with the known-good
Psycles result, uses
237,944 SPIR-V words, and JITs in 1.30926 seconds. That equivalence checks a
compiler transformation only and is not a Cycles quality reference.

The unmodified 35-material export was then run from an empty cache with
optional XIR/SPIR-V optimization disabled and strict native Vulkan required.
AST-to-XIR took 15.95796 seconds, destructuring 15.36656 seconds,
restructuring 120.05845 seconds, and complete SPIR-V XIR legalization
136.25386 seconds. The validated 2,611,188-word / 26-binding module reached a
RADV pipeline and first pixel: scene compilation was 0.737443 seconds,
main-kernel JIT 226.27 seconds, cold wall time 227.24 seconds, and 64×48/1 spp
rendering 0.0114453 seconds. This is a measurable cold-start optimization
target, not yet acceptable production compilation latency.

The complete commands, stage timings, per-material results, test counts, and
machine-readable measurements are in the
[Lone Monk bring-up report](docs/validation/2026-07-29/lone-monk/bringup.json).
The full-scene 40-channel EXR has 3,072 finite values and zero NaN/Inf values
per channel. Its
[decoded preview](docs/validation/2026-07-29/lone-monk/full-scene-vulkan-1spp.png)
was inspected at original resolution: the central monk and architecture are
recognizable through one-sample noise, without a full-frame clear color,
exposure failure, or obvious stale-buffer pattern.

The first 640×480/64 spp comparison then used the installed Blender 5.2.0 LTS
Cycles binary in HIP mode and Psycles in strict-native Vulkan mode on the same
RX 9070 XT. Adaptive sampling and denoising were disabled, seed zero was
fixed, and all 35 original raw material graphs were retained.

The initial result had Combined RMS `22.190855`, a `0.819773` luminance ratio,
and dark/noisy direct illumination, while Diffuse Color and Normal already
aligned. The scene contains no analytic light and uses a procedural Sky
Texture. Blender 5.2 exports its supported mode as `SINGLE_SCATTERING` and
uses `aerosol_density`; the importer recognized only legacy `NISHITA` and
`dust_density`. Background-ray evaluation still used the raw graph, but the
environment-light descriptor lost explicit sun sampling and fell back to
uniform-sphere sampling.

The versioned importer contract now maps current `SINGLE_SCATTERING` and
legacy `NISHITA` to Psycles' implemented single-scattering path, prefers
`aerosol_density`, and retains `dust_density` as a legacy fallback. It does
not claim simple-world support for Blender's distinct
`MULTIPLE_SCATTERING` mode. The regression imports both current and legacy
spellings and asserts procedural transfer and density. No closure or
environment was pre-baked.

After the repair, Combined RMS is `0.262420535`, relative RMS `0.168320400`,
luminance ratio `1.0228698`, and maximum error `10.245852`. All 40 Psycles
channels contain 307,200 finite values and no NaN/Inf. The complete
[numeric report](docs/validation/2026-07-29/lone-monk/report-640x480-64.json),
[Combined triptych](docs/validation/2026-07-29/lone-monk/triptychs-640x480-64/combined.png),
and the other real pass triptychs are committed with the
[full process and visual record](docs/validation/2026-07-29/lone-monk/README.md).
This is a measured convergence baseline, not a final 1:1 quality pass.

## Numeric comparison

All tables compare linear float channels. Relative RMSE is RMSE divided by the
Cycles RMS for that pass.

### Lone Monk 1440×1080/256 spp, current Blender main

This is the primary full-scene result. Both sides use Blender/Cycles
`main@4fe17ef6be5d46251fa5e7dbff9018efb1c719d5`, the same source scene,
frame 4, camera `cam.001`, seed zero, 256 fixed samples, and the same RX 9070
XT. Psycles consumes the current Blender export with all 35 raw material
graphs and no material pre-bake.

| Pass | RMSE | Relative RMSE | Luminance ratio | Maximum error | Invalid pixels |
|---|---:|---:|---:|---:|---:|
| Combined | `0.216918692` | `13.548442%` | `1.0224346` | `10.427597` | 0 |
| Diffuse Color | `0.011697795` | `6.150709%` | `1.0037321` | `0.318984` | 0 |
| Diffuse Direct | `1.680650711` | `18.344467%` | `1.0218524` | `149.06863` | 0 |
| Diffuse Indirect | `0.247345969` | `61.018448%` | `0.9136612` | `71.994682` | 0 |
| Glossy Color | `0.002655193` | `3.748704%` | `0.9994948` | `0.113261` | 0 |
| Glossy Direct | `0.557576835` | `12.878255%` | `1.0152723` | `138.39240` | 0 |
| Glossy Indirect | `0.190440401` | `49.969987%` | `0.9392910` | `34.497143` | 0 |
| Emission | `0.006447404` | `0.880313%` | `0.9996184` | `0.951205` | 0 |
| Environment | `0.000162833` | `10.518015%` | `0.9914694` | `0.122636` | 0 |
| Normal | `0.029506562` | `5.308572%` | n/a | `1.389746` | 0 |
| Transmission Color / Direct / Indirect | `0` | `0` | both zero | `0` | 0 |

Machine-readable result:
[current 1080p report](docs/validation/2026-07-29/lone-monk/report-1440x1080-256-main-4fe17ef6.json).

Real current-source triptychs:
[Combined](docs/validation/2026-07-29/lone-monk/triptychs-1440x1080-256-main-4fe17ef6/combined.png),
[Diffuse Color](docs/validation/2026-07-29/lone-monk/triptychs-1440x1080-256-main-4fe17ef6/diffcol.png),
[Normal](docs/validation/2026-07-29/lone-monk/triptychs-1440x1080-256-main-4fe17ef6/normal.png),
[Diffuse Direct](docs/validation/2026-07-29/lone-monk/triptychs-1440x1080-256-main-4fe17ef6/diffdir.png),
[Diffuse Indirect](docs/validation/2026-07-29/lone-monk/triptychs-1440x1080-256-main-4fe17ef6/diffind.png),
[Glossy Color](docs/validation/2026-07-29/lone-monk/triptychs-1440x1080-256-main-4fe17ef6/glosscol.png),
[Glossy Direct](docs/validation/2026-07-29/lone-monk/triptychs-1440x1080-256-main-4fe17ef6/glossdir.png),
[Glossy Indirect](docs/validation/2026-07-29/lone-monk/triptychs-1440x1080-256-main-4fe17ef6/glossind.png),
[Emission](docs/validation/2026-07-29/lone-monk/triptychs-1440x1080-256-main-4fe17ef6/emit.png),
[Environment](docs/validation/2026-07-29/lone-monk/triptychs-1440x1080-256-main-4fe17ef6/env.png),
[Transmission Color](docs/validation/2026-07-29/lone-monk/triptychs-1440x1080-256-main-4fe17ef6/transcol.png),
[Transmission Direct](docs/validation/2026-07-29/lone-monk/triptychs-1440x1080-256-main-4fe17ef6/transdir.png),
and
[Transmission Indirect](docs/validation/2026-07-29/lone-monk/triptychs-1440x1080-256-main-4fe17ef6/transind.png).

Every panel was opened at original resolution. Camera/framing, silhouettes,
architecture, texture placement, and large-scale materials/normals align.
There is no flip, missing object, full-frame corruption, or device-reset
residue. Direct illumination appears on the same surfaces but Psycles is
slightly brighter. Diffuse and glossy indirect preserve the same spatial
structure but remain visibly darker, matching the 0.914 and 0.939 mean-energy
ratios. Emission is nearly identical; all transmission panels are exact
black. The detailed visual notes and commands are in the
[Lone Monk process log](docs/validation/2026-07-29/lone-monk/README.md).

### Lone Monk 640×480/64 spp

| Pass | RMSE | Relative RMSE | Luminance ratio | Maximum error | Invalid pixels |
|---|---:|---:|---:|---:|---:|
| Combined | `0.262420535` | `16.832040%` | `1.0228698` | `10.245852` | 0 |
| Diffuse Color | `0.011069954` | `5.873532%` | `1.0036870` | `0.279242` | 0 |
| Diffuse Direct | `2.92519927` | `32.096863%` | `1.0240113` | `144.32452` | 0 |
| Diffuse Indirect | `0.572785676` | `94.700590%` | `0.9115970` | `339.22046` | 0 |
| Glossy Color | `0.003341173` | `4.800315%` | `0.9994941` | `0.090504` | 0 |
| Glossy Direct | `0.836308777` | `20.425919%` | `1.0154054` | `97.110794` | 0 |
| Glossy Indirect | `0.381446093` | `84.778720%` | `0.9441791` | `14.612365` | 0 |
| Emission | `0.014200922` | `1.970841%` | `0.9995498` | `1.187111` | 0 |
| Environment | `0.000476680` | `60.939653%` | `1.4539939` | `0.277972` | 0 |
| Normal | `0.030848974` | `5.677077%` | n/a | `1.206512` | 0 |
| Transmission Color / Direct / Indirect | `0` | `0` | `0` | `0` | 0 |

The indirect relative errors are noise-sensitive, but their mean-energy gaps
are also systematic at this sample count: diffuse indirect is approximately
8.8% low and glossy indirect approximately 5.6% low. Environment has high
relative error only because its absolute reference energy is tiny.

Machine-readable result:
[Lone Monk report](docs/validation/2026-07-29/lone-monk/report-640x480-64.json).

Real triptychs:
[Combined](docs/validation/2026-07-29/lone-monk/triptychs-640x480-64/combined.png),
[Diffuse Color](docs/validation/2026-07-29/lone-monk/triptychs-640x480-64/diffcol.png),
[Diffuse Direct](docs/validation/2026-07-29/lone-monk/triptychs-640x480-64/diffdir.png),
[Diffuse Indirect](docs/validation/2026-07-29/lone-monk/triptychs-640x480-64/diffind.png),
[Glossy Color](docs/validation/2026-07-29/lone-monk/triptychs-640x480-64/glosscol.png),
[Glossy Direct](docs/validation/2026-07-29/lone-monk/triptychs-640x480-64/glossdir.png),
[Glossy Indirect](docs/validation/2026-07-29/lone-monk/triptychs-640x480-64/glossind.png),
[Emission](docs/validation/2026-07-29/lone-monk/triptychs-640x480-64/emit.png),
[Environment](docs/validation/2026-07-29/lone-monk/triptychs-640x480-64/env.png),
and [Normal](docs/validation/2026-07-29/lone-monk/triptychs-640x480-64/normal.png).
The three transmission triptychs are retained beside them and are exactly
black in all panels.

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

- Lone Monk Combined aligns in silhouette, camera, architecture, principal
  colors, and sun/shadow placement after the single-scattering-sky repair.
  The former missing-direct-light/exposure failure is absent. Psycles is
  slightly brighter and the amplified panel contains both radiance-correlated
  sampling noise and coherent edge/detail residuals;
- Lone Monk Diffuse Color aligns brick, marble, wood, foliage, books, and
  texture placement, while Normal aligns large-scale orientation and
  mapped/bump structure. Their amplified residuals concentrate on
  high-frequency detail and visibility edges, not a frame flip, normal-space
  rotation, or material-slot permutation;
- Lone Monk direct-pass illuminated regions align and have mean luminance
  within 2.4%. Diffuse and Glossy Indirect preserve the same spatial
  structure but are visibly darker, matching the measured 0.912 and 0.944
  luminance ratios. This prevents a final visual-acceptance verdict;
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

## GPU timings

The current Lone Monk row is the primary same-physical-device,
1080p-or-higher measurement:

| Measurement | Cycles 5.3 Alpha HIP | Psycles Vulkan |
|---|---:|---:|
| Lone Monk 1440×1080/256 spp render-only | `18.961390479 s` | `25.9918 s` |
| Whole monitored command | `19.439086148 s` | `29.805725258 s` |
| Scene / shader setup reported separately | included in Cycles call | `0.66606 s` scene + `2.2724 s` warm JIT |
| VRAM absolute peak | `6,764,978,176 B` | `5,793,931,264 B` |
| VRAM increase over baseline | `2,659,450,880 B` | `1,711,570,944 B` |

On render-only intervals, Psycles throughput is `0.729514×` current Cycles,
or `1.370775×` slower. There is no same-device speedup yet. Psycles' measured
baseline-relative VRAM increase is 947,879,936 bytes lower (35.64%), while
its absolute peak is 971,046,912 bytes lower (14.35%). Both forms are
reported because desktop VRAM already in use is not renderer memory.

The historical packaged-Blender 5.2 640×480/64 spp row measured `0.96 s`
for Cycles and `1.48413 s` for Psycles. It remains useful as a pre-current-
source checkpoint but is no longer the primary performance boundary.

The remaining rows are small 64×64/256 spp probes and are useful for
shader-size and backend health only.

| Backend/probe | Scene compile | Shader JIT | Render |
|---|---:|---:|---:|
| Vulkan flat, cold cache | `0.025667 s` | `0.559935 s` | `0.015133 s` |
| Vulkan flat, hot cache | `0.015175 s` | `0.032178 s` | `0.014824 s` |
| HIP flat, cold cache | `0.303026 s` | `0.974313 s` | `0.009619 s` |
| HIP flat, repeat | `0.043706 s` | `0.355552 s` | `0.009915 s` |
| Vulkan transparent mix, hot | `0.013557 s` | `0.036953 s` | `0.012637 s` |
| Vulkan transparent data, hot | `0.019546 s` | `0.019263 s` | `0.013890 s` |

The focused Cycles metadata selected CPU execution, so only the Lone Monk row
supports the same-device comparison.

## Current-source Cycles build gate

Blender/Cycles `main@4fe17ef6be5d46251fa5e7dbff9018efb1c719d5`
was built locally as Blender 5.3.0 Alpha with Release, headless Cycles HIP,
and a `gfx1201`-only AOT kernel. The official Linux dependency checkout is
`ecbd06cf6d2a4aa6b00a61ffb479fc81b17aba08`; Git LFS integrity passes.
The 32-job incremental resume completed 2863 tasks in 5:21.19 with no swap,
and the installed binary selects only the RX 9070 XT for HIP.

The first unmodified Lone Monk smoke at 64×48/1 spp completed in
`1.011486 s` of Cycles render time. Its 43-channel multilayer EXR contains
132,096 finite values and no NaN or Inf. At 50 ms sampling, whole-process
VRAM rose from `4,067,905,536` to `6,000,680,960` bytes, an increase of
`1,932,775,424` bytes. Full configuration, the initial missing-LFS failure,
external signal-15 resume, non-system staging install, and smoke command are
recorded in the
[Lone Monk log](docs/validation/2026-07-29/lone-monk/README.md).

The matched 1440×1080/256 spp reference then completed in `18.961390479 s`
of Cycles render time. Its 43-channel EXR contains 66,873,600 finite values,
has no NaN or Inf, and names only the RX 9070 XT HIP device. The current
Blender export retains 350 geometries, 7,543 instances, 35 raw material
graphs, and 47 images; Blender/Cycles does not evaluate or bake any material
for Psycles.

The first Psycles attempt exposed two independent production defects. One
256-spp Vulkan dispatch exceeded the AMDGPU compute watchdog and reset the
queue. Luisa's Release-only `VK_CHECK_RESULT` macro then discarded
`VK_ERROR_DEVICE_LOST`, so the process falsely returned success with an
almost-all-zero EXR and four extreme pixels. The corrupt output and its
`2.19935 s` claimed render time are explicitly excluded from performance and
quality results.

Luisa `d57720955` now checks every Vulkan result in every build configuration.
Its forced-`NDEBUG` regression proves success is evaluated once and device
loss terminates with `SIGABRT`. Psycles `dcb96e3` partitions a requested
sample interval into a contiguous, ordered, non-overlapping exact cover with
at most 8 samples per synchronized dispatch. The exhaustive regression checks
small ranges and the 32-bit boundary; a real 64×48/16 spp Vulkan render is
pixel-equivalent as one 16-spp dispatch or two 8-spp dispatches.

With 32 bounded dispatches, the 1440×1080/256 spp Psycles run completed in
`25.9918 s` render-only. Its 40-channel EXR contains 62,208,000 finite values,
zero NaN/Inf, and no kernel timeout/reset record. Full commands, output
hashes, failure diagnostics, invariants, regressions, VRAM samples, and
triptychs are in the
[Lone Monk log](docs/validation/2026-07-29/lone-monk/README.md).

## Known limitations and next gate

- Diagnose the remaining diffuse/glossy indirect energy deficit and sampler /
  stochastic-distribution differences using Cycles alone as the oracle. Do
  not add a CPU reference sampler or renderer.
- The accepted current-source 1440×1080/256 spp result still has Combined
  relative RMSE `13.55%`, diffuse indirect mean energy 8.63% low, glossy
  indirect 6.07% low, and render throughput below Cycles. It is a production
  validation boundary, not final 1:1 acceptance.
- The bounded sample count limits per-dispatch work for this gate. Extremely
  large images may also need a formally exact pixel/tile partition so one
  8-spp dispatch remains below device watchdog limits.
- Blender 5.2 `MULTIPLE_SCATTERING` sky is distinct from the implemented
  single-scattering equations and is not yet supported by the simple-world
  importance sampler.
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
