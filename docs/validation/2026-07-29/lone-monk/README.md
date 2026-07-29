# Lone Monk bring-up — 2026-07-29

This directory is the durable record for the first full-scene gate. The
machine-readable [bring-up report](bringup.json) records the exact scene hash,
export scale, backend attempts, compiler diagnostics, controlled material
isolation, test results, visual checks, and triptych status.

## Immutable input and comparison policy

The input is scene `daylight`, frame 4, camera `cam.001` from
`lone-monk_cycles_and_exposure-node_demo.blend`, SHA-256
`4250d4205d8d01cefd98c15e81021d6dead540b2923797378bf7b32e96e8b8f7`.
Its original configuration is 1440×1080 at 4096 samples.

The source scene was exported as evaluated geometry and instances plus the
original Blender node graphs, links, sockets, and material metadata. The full
export contains 350 geometries, 7,543 instances, 35 raw material graphs, and
47 images. No Cycles material result was baked into the export. Controlled
variants below retain all geometry, instances, world data, and images; they
reduce only the material-program set to isolate compiler scaling. They are
diagnostic inputs, not quality references.

Cycles is the only rendering oracle. A Lone Monk quality result requires
matching Cycles and Psycles linear EXRs from the same scene, frame, camera,
seed, sample count, integrator settings, pass set, and RX 9070 XT device.

## Initial full-export backend checks

Neither initial Psycles backend reached a first pixel:

- Vulkan spent more than 20 minutes in the cold main-kernel JIT. Disabling
  optional XIR and SPIR-V optimization did not change the two-thread,
  multi-gigabyte code-generation profile.
- HIP spent more than 10 minutes building the 350 HIPRT bottom-level
  acceleration structures at 100% GPU utilization and did not reach the main
  render-kernel JIT.

These are distinct backend paths: HIP was blocked before main-kernel
generation, while Vulkan had already built the scene acceleration structures
and was blocked in the monolithic material/render kernel.

## XIR investigation and formal repair

Pass tracing was added behind `LUISA_XIR_TRACE_PASSES=1`; verbose messages are
visible with `LUISA_LOG_LEVEL=verbose`. The trace records AST-to-XIR,
individual pipeline passes, restructuring stages, block/instruction counts,
raw terminator counts, and SPIR-V validation boundaries.

The first controlled Vulkan input retained the complete scene but selected six
original materials plus the diagnostic dark material. Its render kernel
entered restructuring with 1,799 blocks, 278,375 instructions, 555 raw
conditional branches, and 13 raw indexed branches.

The investigation proceeded as follows:

1. The old `try_restructure_if_batch` implementation collected a candidate
   batch but returned after its first successful rewrite. On this input it
   spent approximately 113.29 seconds repeatedly walking the graph and then
   failed SPIR-V validation.
2. Processing the full stable candidate snapshot reduced legalization to about
   23 seconds, but exposed an independent invalid graph. SPIR-V reported that
   block 3093 branched into a selection construct whose header was block 3087.
3. Disassembly showed header 3087 declaring merge 3088, its true path entering
   block 3171, and a path dominated by the declared merge later branching back
   into 3171. Therefore the graph contained a post-merge selection re-entry;
   batching did not create this defect, it only reached it sooner.
4. The merge inference and its verifier were repaired from structural
   invariants. For a selection `(H, M)`, no edge may leave the region dominated
   by `M` and re-enter the `H`-dominated interior before `M`. Ordinary common
   joins rank ahead of loop-boundary convergence proxies; a boundary proxy is
   used only when it is the real convergence immediately before the enclosing
   loop boundary.
5. Batch progress is now monotonic: every successful rewrite strictly reduces
   the number of raw conditional terminators and creates none. Terminator
   identity, arms, dominance, post-dominance, and merge validity are
   revalidated after each mutation.

This is a CFG-wide rule, not a Lone Monk block-number special case. Two
red/green regressions pin it:

- `restructure_if_batch_consumes_a_linear_diamond_chain` uses 64 diamonds and
  a one-iteration main-loop limit. It failed before the batch repair and now
  proves that one snapshot consumes the entire linear chain.
- `restructure_does_not_reenter_selection_after_its_merge` constructs a loop,
  an outer raw selection, a nested selection, and a shared continue boundary.
  It failed before the merge repair and now proves the no-post-merge-re-entry
  postcondition.

The focused Luisa binary passes 51 tests / 1,013 assertions. At the current
boundary, all 21 structural SPIR-V tests pass, the complete Vulkan SPIR-V
runtime suite passes 86/86 tests / 2,029 assertions on the RX 9070 XT, and
CTest passes 115/116 with 32-way scheduling. The sole CTest failure remains
the independently reproducible, pre-existing EASTL `fixed_vector` allocation
contract. Psycles passes 13/13 using a 32-job build and test schedule. The
published Luisa boundary is `next@d57720955`.

## Result after the XIR repair

The same six-material controlled input completed with optional XIR and SPIR-V
optimization disabled:

```bash
env LUISA_XIR_DISABLE_OPTIMIZATION=1 \
    LUISA_SPIRV_OPT_LEVEL=0 \
  build/bin/psycles_render_blender_scene \
  /tmp/lone-monk-20260729/variants/six-materials \
  /tmp/lone-monk-20260729/variants/six-materials-fixed.ppm \
  vk 64 48 1
```

| Measurement | Result |
|---|---:|
| Full retained scene geometry | 350 geometries / 7,543 instances |
| Renderer material count | 8 |
| Scene compilation | `0.648155 s` |
| Main-kernel JIT | `22.5549 s` |
| SPIR-V binary | 497,049 words / 26 bindings |
| Render | `0.00485903 s` at 64×48 / 1 spp |
| Outputs | PPM, PFM passes, full-float multilayer EXR |

The decoded preview was inspected at original resolution:
[six-material Vulkan smoke](six-materials-vulkan-smoke.png). It contains
finite, decodable image data with no full-frame NaN, Inf, or exposure
explosion. Its sparse cyan point-like appearance is expected from one sample
and the deliberately reduced material set. It is only a first-pixel/compiler
smoke result and is not evidence of Cycles quality alignment.

## Material-program scaling isolation

An eleven-original-material variant retained the same geometry and instances.
It timed out after 180 seconds without a CFG or SPIR-V validation error. The
process remained healthy at approximately 197% CPU. Its trace reached:

| Stage | Measurement |
|---|---:|
| AST to XIR | `11.684 s` |
| Destructure | `11.666 s` |
| Inline pointer arguments | `1.913 s` |
| Reg2mem | `0.176 s` |
| Restructure preflight blocks | 50,020 |
| Restructure preflight instructions | 2,205,924 |
| Raw conditional / indexed branches | 16,611 / 13 |

The timeout occurred during module preflight, before batch restructuring.
This proves that the repaired batch loop is no longer the active scaling
root. One-material-at-a-time variants then isolated the nonlinear growth:

| Original material retained with dark fallback | Vulkan result |
|---|---:|
| `brick pavement` | JIT `7.89866 s`, 305,234 SPIR-V words, rendered |
| `bush` | JIT `18.7265 s`, 365,391 SPIR-V words, rendered |
| `church brick` | JIT `8.60974 s`, 345,257 SPIR-V words, rendered |
| `column marble` | before repair: timed out; after repair: JIT `12.3495 s`, 432,610 SPIR-V words, rendered |
| `copper pipe` | JIT `3.34863 s`, 321,946 SPIR-V words, rendered |

`column marble` alone produces 45,900 blocks, 1,543,437 instructions, 15,266
raw conditional branches, and 10 raw indexed branches before restructuring.
It has 34 nodes / 33 links and includes Bump, Color Ramp, two RGB Curves,
image textures, Mapping, Mix, Math, Normal Map, Object Info, Principled, and
reroutes. Its exported Color Ramp contains 257 lookup samples.

### Column-marble root cause and bounded attribute lookup

The sampled tables were eliminated as a hypothesis before changing the
renderer. Variants retaining the same graph but reducing the Color Ramp, the
two RGB Curves, or all three lookup tables from 257 samples to two still
produced exactly 45,900 blocks, 1,543,437 instructions, and 15,266 raw
conditionals. Dependency cuts then located the expansion:

| Dependency-preserving diagnostic | Values retained | XIR result |
|---|---:|---:|
| Base Color disconnected | 32 | 900 blocks / 137,567 instructions / 266 raw conditionals; JIT `4.79313 s` |
| Only Base Color dependency retained | 63 | 45,420 blocks / 1,500,987 instructions / 15,106 raw conditionals; timed out |

The Base Color path reaches a Vertex Color node. Lone Monk contains 367 named
UV layers and 12 color attributes, for 379 geometry-attribute bindings. The
old `BufferShaderServices::attribute` iterated that host vector while recording
each material callable and emitted one Luisa `$if` per whole-scene binding.
The observed `15,266 / 379 = 40.28` conditionals per binding matches repeated
callable recording. Scene metadata cardinality had incorrectly become shader
control-flow cardinality.

The repair is defined by this code-size invariant: for a fixed shader program
and fixed number of attribute lookup operations, recorded AST/XIR control-flow
size is independent of the number of scene attribute bindings. Attribute
cardinality may affect uploaded data size and device runtime work, but not the
host-recorded shader structure.

Psycles now uploads:

- one compact binding table containing `(attribute id, value-buffer slot)`;
- one range per geometry containing `(offset, count, triangle-buffer slot)`.

The shader reads only the current geometry's range through two bindless device
buffers and searches it with one dynamic loop. A `found` state participates in
the loop predicate, so the lookup has no cross-construct early `break`. The
raw closure graph, its 257-sample tables, and all node dependencies are
unchanged.

The red/green regression records the real `BufferShaderServices::attribute`
implementation, translates it AST-to-XIR, and counts structured selections.
With 512 old host bindings the pre-repair test exceeded the constant bound and
returned failure code 4. The device-table implementation remains at or below
eight `IfInst` nodes for the same logical cardinality. This pins the
cardinality invariant rather than the `column marble` node names or one scene
size.

The original unmodified `column marble` graph then completed:

```bash
env LUISA_XIR_DISABLE_OPTIMIZATION=1 \
    LUISA_SPIRV_OPT_LEVEL=0 \
    LUISA_XIR_TRACE_PASSES=1 \
    LUISA_LOG_LEVEL=verbose \
  build/bin/psycles_render_blender_scene \
  /tmp/lone-monk-20260729/variants/material-column-marble \
  /tmp/lone-monk-20260729/variants/material-column-marble-device-attributes-no-break.ppm \
  vk 64 48 1
```

| Measurement | Before | After |
|---|---:|---:|
| XIR blocks | 45,900 | 1,360 |
| XIR instructions | 1,543,437 | 203,797 |
| Raw conditional branches | 15,266 | 406 |
| Raw indexed branches | 10 | 10 |
| XIR restructure | exceeded the diagnostic bound | `10.47494 s` |
| Vulkan shader JIT | no completion | `12.3495 s` |
| SPIR-V | no completion | 432,610 words / 26 bindings |
| Render | no first pixel | `0.00485309 s` at 64×48 / 1 spp |

The 40-channel full-float EXR has 3,072 finite values per channel and zero
NaN/Inf values. The preview was inspected at original resolution:
[column-marble Vulkan smoke](column-marble-vulkan-smoke.png). It shows the
same sparse cyan point-like one-sample character as the earlier controlled
smoke, without full-frame corruption or exposure failure. It is still a
compiler/first-pixel diagnostic, not a Cycles quality comparison.

## Read-only callable ABI repair

Preserving uniquely rooted read-only resource callables removed the remaining
material inlining explosion, but its first cold `column marble` result exposed
a semantic regression. The outlined shader was finite and fast, yet differed
from the previously correct Psycles XIR-inline result:

| Diagnostic | Result |
|---|---:|
| Outlined shader | 237,940 SPIR-V words; JIT `1.309 s` |
| Mean absolute difference | `0.000991478` |
| RMS difference | `0.0483804` |
| Maximum absolute difference | `7.020724` |
| Affected pixels | 383 / 3,072 |

This older Psycles image was used only to detect a compiler semantic change;
it is not a substitute for Cycles and supports no rendering-quality claim.

Two rejected hypotheses narrowed the boundary:

1. SPIR-V optimization level 2 reduced the module from 237,940 to 141,703
   words but preserved the wrong pixels.
2. Exhaustive SPIR-V function inlining expanded it to 603,537 words and
   remained pixel-exact with the wrong outlined result.

Therefore the error existed before the driver-call boundary. SPIR-V emits
used functions in post-order, so callables precede the kernel. Direct-buffer
offset, size, and address loads in those callables used
`_buffer_metadata_offset`, but that module-wide ABI value had previously been
initialized only inside `_emit_kernel`. The callable therefore read metadata
from argument-block word zero. XIR inlining had hidden the ordering defect by
moving the same load into the later kernel emission.

The formal repair freezes one validated `SpirvKernelArgumentLayoutPlan` before
the first function is emitted. Every callable and the sole kernel now observe
that immutable layout, and kernel emission asserts that the frozen metadata
offset is unchanged. This is a module ABI initialization-order invariant, not
a `column marble` or argument-count special case.

The red/green Vulkan regression
`vk_user_compute_outlined_readonly_buffer_uses_frozen_kernel_argument_layout`
uses a nonzero direct-buffer subview, a scalar kernel argument that moves the
metadata trailer to word two, and a real outlined callable. Before the repair
it read word zero and failed; after the repair it returns
`{18, 29, 40, 51}` and the dumped SPIR-V contains one `OpFunctionCall`.
The existing unique-readonly-resource regression also passes. The focused
runtime results are 3 and 8 assertions respectively; the complete runtime
suite passes 2,029 assertions.

A true cold `column marble` rerun at the historical `next@eb167454a`
boundary produced 237,944
SPIR-V words, compiled in `1.30926 s`, and was pixel-exact with the known-good
pre-preservation Psycles output. It differs from the broken outlined image by
the same RMS `0.0483804`, maximum `7.020724`, and 383 affected pixels, proving
that the regression and repair exercise observable semantics.

## Unmodified full-scene Vulkan result

The complete export was then retried without removing or replacing any of its
35 original material graphs. The run used a separate empty shader cache and
restored the pre-existing cache afterward:

```bash
env LUISA_XIR_DISABLE_OPTIMIZATION=1 \
    LUISA_SPIRV_OPT_LEVEL=0 \
    LUISA_XIR_TRACE_PASSES=1 \
    LUISA_LOG_LEVEL=verbose \
    LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
  build/bin/psycles_render_blender_scene \
  /tmp/lone-monk-20260729/export \
  /tmp/lone-monk-20260729/lone-monk-full-35-material-abi-fix.ppm \
  vk 64 48 1
```

| Measurement | Result |
|---|---:|
| Retained scene | 350 geometries / 7,543 instances / 35 raw material graphs / 47 images |
| Runtime material programs | 37, including renderer-internal programs |
| AST to XIR | `15.95796 s` |
| Destructure CFG | `15.36656 s` |
| Restructure CFG | `120.05845 s` |
| Complete SPIR-V XIR legalization | `136.25386 s` |
| SPIR-V emission and validation | 2,611,188 words / 26 bindings |
| Scene compilation | `0.737443 s` |
| Main-kernel JIT | `226.27 s` |
| Cold wall time | `227.24 s` |
| Render | `0.0114453 s` at 64×48 / 1 spp |
| Cold-cache artifacts | two `.spv` and two `.vk` files |
| Outputs | PPM, PFM passes, 40-channel full-float multilayer EXR |

This is the first unmodified-material-set Vulkan first pixel. The previous
default run had no main-shader cache artifact after 1,245.51 seconds; the
strict-native cold run now validates SPIR-V and completes in 227.24 seconds.
It is still too slow for production cold-start requirements, but the failure
mode has changed from non-completion to a measurable optimization target.

OpenImageIO reports 3,072 finite values and zero NaN/Inf values in every one
of the 40 channels. The [full-scene one-sample preview](full-scene-vulkan-1spp.png)
was inspected at its original 64×48 resolution. The central monk and
architectural silhouette are recognizable through the one-sample noise, with
no full-frame corruption, constant clear color, exposure explosion, or
obvious stale-buffer pattern. This remains a compiler/first-pixel check; it
does not establish agreement with Cycles.

## Triptychs and visual acceptance

The first real full-scene differential baseline is 640×480 at 64 fixed
samples, seed zero. This preserves the scene's 4:3 aspect ratio and satisfies
the minimum 480p gate. Adaptive sampling, denoising, the compositor, and the
sequencer were disabled on the Cycles side. Psycles consumed the same
unmodified export: 350 geometries, 7,543 instances, 35 original raw material
graphs, and 47 images. No material or closure was evaluated or baked by
Blender/Cycles.

The local source reference was fetched and fast-forwarded before inspection:
`/home/mike/Projects/blender-cycles` is clean at Blender
`main@4fe17ef6be5d46251fa5e7dbff9018efb1c719d5`. The pixels below were produced
by the installed Blender 5.2.0 LTS binary, build hash `fbe6228777e7`, because
the current source checkout has not yet been built. Source-inspection and
pixel revisions are deliberately reported separately.

The Cycles command was:

```bash
/usr/bin/blender \
  /home/mike/Downloads/lone-monk_cycles_and_exposure-node_demo.blend \
  --background --python-exit-code 1 \
  --python tools/render_cycles_golden.py -- \
  /tmp/lone-monk-20260729/quality-640x480-64/cycles.exr \
  640 480 64 0 \
  --cycles-device HIP \
  --device-name "Radeon RX 9070 XT"
```

The golden metadata proves that the only enabled Cycles device was
`HIP_AMD Radeon RX 9070 XT_0000:03:00`, `scene.cycles.device` was `GPU`, and
adaptive sampling and denoising were both false. Cycles wrote 43 full-float
channels. Every channel contains 307,200 finite values and no NaN or Inf.

The corresponding strict-native Psycles command was:

```bash
env LUISA_XIR_DISABLE_OPTIMIZATION=1 \
    LUISA_SPIRV_OPT_LEVEL=0 \
    LUISA_XIR_TRACE_PASSES=1 \
    LUISA_LOG_LEVEL=verbose \
    LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
  build/bin/psycles_render_blender_scene \
  /tmp/lone-monk-20260729/export \
  /tmp/lone-monk-20260729/quality-640x480-64/psycles-sun-guiding-rec709.ppm \
  vk 640 480 64
```

The Vulkan log names `AMD Radeon RX 9070 XT (RADV GFX1201)`. Psycles wrote 40
full-float channels; every channel likewise contains 307,200 finite values
and no NaN or Inf. The final EXR reports both `oiio:ColorSpace` and
`colorInteropID` as `lin_rec709_scene`, matching the Cycles EXR. Re-rendering
after changing only this metadata was pixel-exact to the pre-metadata-fix
Psycles EXR.

### Blender 5.2 single-scattering sky diagnosis

The first matched render exposed a systematic failure rather than merely
different random noise:

| Combined measurement | Before | After |
|---|---:|---:|
| RMSE | `22.190855` | `0.262420535` |
| Relative RMSE | `14.23354` | `0.168320400` |
| Mean-luminance ratio | `0.8197728` | `1.0228698` |
| Maximum absolute error | `4505.7529` | `10.245852` |

The material-color and normal passes already aligned closely, while Combined
and direct-light passes were much darker and contained fireflies. The scene
has no analytic lights; it is lit by one procedural sky. Blender 5.2 exports
that node as `sky_type = SINGLE_SCATTERING` and its aerosol control as
`aerosol_density`. Psycles' simple-world recognizer accepted only the legacy
`NISHITA` spelling and read only `dust_density`.

The raw world graph still evaluated the sky for background rays, so the camera
saw a plausible sky. However, failure to construct the procedural environment
descriptor meant that the light distribution had no explicit sun to sample.
Uniform-sphere environment samples almost never hit the small sun disc. This
explains the pass-local signature: material colors and normals were stable,
but direct illumination was dark and high-variance.

Current Cycles source defines distinct
`NODE_SKY_SINGLE_SCATTERING` and `NODE_SKY_MULTIPLE_SCATTERING` modes and
sun-guides a single eligible, untransformed sky node. Psycles currently
implements the single-scattering equations only. The compatibility mapping is
therefore explicit and bounded:

- current `SINGLE_SCATTERING` and legacy `NISHITA` select the supported
  procedural single-scattering path;
- current `aerosol_density` is preferred, with legacy `dust_density` as a
  compatibility fallback;
- `MULTIPLE_SCATTERING` is not claimed or silently mapped by the simple-world
  recognizer.

The importer regression now exercises both the current enum/property pair and
the legacy pair, checks that each stays procedural, and checks the transferred
aerosol/dust value. This is a versioned scene-contract mapping, not a
Lone-Monk-specific material replacement.

The EXR inspection then found a separate interchange error:
`oiio:ColorSpace = scene_linear` is an OCIO role and resolved to
`lin_ap1_scene` with the installed Blender configuration, even though Psycles
pass values are linear Rec.709. The writer now emits the stable
`lin_rec709_scene` identity. Its regression reopens the file and asserts both
the OIIO color-space attribute and the derived OpenEXR `colorInteropID`, in
addition to exact channel names and float values.

The repair plus both regressions pass the complete project gate:

```bash
cmake --build build --parallel 32
ctest --test-dir build --output-on-failure -j32
```

At that 640×480 baseline boundary, 12/12 tests passed in 0.42 seconds. The
later sample-partition regression raises the current gate to 13/13.

### Numeric result

All metrics compare raw linear EXR channels. Relative RMSE is RMSE divided by
the Cycles RMS for that pass. The complete machine-readable result, including
mean vectors, percentiles, invalid counts, orientation guards, and triptych
display scales, is
[report-640x480-64.json](report-640x480-64.json).

| Pass | RMSE | Relative RMSE | Luminance ratio | Maximum error | Invalid pixels |
|---|---:|---:|---:|---:|---:|
| Combined | `0.262420535` | `0.168320400` | `1.0228698` | `10.245852` | 0 |
| Diffuse Color | `0.011069954` | `0.058735319` | `1.0036870` | `0.279242` | 0 |
| Diffuse Direct | `2.92519927` | `0.320968635` | `1.0240113` | `144.32452` | 0 |
| Diffuse Indirect | `0.572785676` | `0.947005900` | `0.9115970` | `339.22046` | 0 |
| Glossy Color | `0.003341173` | `0.048003153` | `0.9994941` | `0.090504` | 0 |
| Glossy Direct | `0.836308777` | `0.204259189` | `1.0154054` | `97.110794` | 0 |
| Glossy Indirect | `0.381446093` | `0.847787201` | `0.9441791` | `14.612365` | 0 |
| Emission | `0.014200922` | `0.019708405` | `0.9995498` | `1.187111` | 0 |
| Environment | `0.000476680` | `0.609396535` | `1.4539939` | `0.277972` | 0 |
| Normal | `0.030848974` | `0.056770775` | n/a | `1.206512` | 0 |
| Transmission Color | `0` | `0` | `0` | `0` | 0 |
| Transmission Direct | `0` | `0` | `0` | `0` | 0 |
| Transmission Indirect | `0` | `0` | `0` | `0` | 0 |

The Environment relative error is large only because its Cycles RMS is
approximately `7.82e-4`; its absolute RMSE is `4.77e-4`. Conversely, the
indirect diffuse and glossy luminance ratios are materially low by about
8.8% and 5.6% respectively and remain real convergence targets.

### Original-resolution visual inspection

Every panel below was opened and inspected at its committed 640×480 panel
resolution. Cycles and Psycles share one diagnostic linear-to-sRGB mapping and
one exposure scale. Normal uses `normal * 0.5 + 0.5`. The third panel is
absolute linear difference amplified independently by the reciprocal of its
99.5th percentile; the multiplier is printed above the panel.

- [Combined](triptychs-640x480-64/combined.png): silhouette, camera,
  architecture, principal colors, and the sun/shadow pattern align. There is
  no longer a missing-light or global-exposure failure. Psycles remains
  slightly brighter and the amplified panel shows radiance-correlated sampling
  noise plus coherent edge/detail residuals.
- [Diffuse Color](triptychs-640x480-64/diffcol.png): brick, marble, wood,
  foliage, books, and texture-coordinate placement align closely. Amplified
  residuals concentrate on edges, foliage, bump detail, and a few bright
  surfaces rather than showing a material-slot permutation.
- [Normal](triptychs-640x480-64/normal.png): large-scale orientation,
  geometry silhouettes, and mapped/bump detail align. Residuals are strongest
  on high-frequency texture and visibility edges; there is no frame flip or
  global normal-space rotation.
- [Diffuse Direct](triptychs-640x480-64/diffdir.png) and
  [Glossy Direct](triptychs-640x480-64/glossdir.png): the formerly absent
  direct sun contribution is present in the same illuminated regions. The
  remaining error is dominated by independent sample noise and edge detail,
  with mean luminance within 2.4% and 1.6%.
- [Diffuse Indirect](triptychs-640x480-64/diffind.png) and
  [Glossy Indirect](triptychs-640x480-64/glossind.png): spatial structure is
  recognizable and aligned, but Psycles is visibly somewhat darker overall,
  consistent with the measured 0.912 and 0.944 luminance ratios.
- [Emission](triptychs-640x480-64/emit.png) and
  [Environment](triptychs-640x480-64/env.png) have low absolute error.
  [Glossy Color](triptychs-640x480-64/glosscol.png) also aligns closely.
- [Transmission Color](triptychs-640x480-64/transcol.png),
  [Transmission Direct](triptychs-640x480-64/transdir.png), and
  [Transmission Indirect](triptychs-640x480-64/transind.png) are exactly zero
  in both renderers for this frame.

This baseline is a major correctness improvement, but it does **not** pass the
1:1 Cycles quality gate. The indirect-pass energy gap, stochastic
distribution/sampler differences, high-frequency normal/material residuals,
and outliers still require investigation at higher samples. The current
Blender/Cycles source build and its 1440×1080 matched run are recorded below;
that higher-sample gate confirms that the indirect-energy gap persists.

### Same-device timing

Both renderers selected the same physical RX 9070 XT, through HIP for Cycles
and RADV Vulkan for Psycles:

| Measurement | Cycles 5.2 HIP | Psycles Vulkan |
|---|---:|---:|
| Internal render-only | `0.96 s` | `1.48413 s` |
| Synchronization | `0.67 s` | reported inside dispatch completion |
| Renderer-reported total / setup | `1.63 s` total | `0.659837 s` scene + `2.14923 s` warm JIT |
| Golden-script elapsed / cold setup | `1.85641 s` | `244.841 s` scene + cold JIT |

On the only directly comparable render-only numbers, the current Psycles
baseline has `0.6468×` Cycles throughput, i.e. it is approximately `1.546×`
slower; there is no speedup claim. The 244-second cold monolithic-kernel JIT
is also far outside a production target. The apparent `1.25×` ratio obtained
by dividing the Cycles Python-call elapsed time by Psycles render-only time is
not reported as a speedup because those intervals have different boundaries.
Peak VRAM was not sampled for this short matched run and remains an explicit
missing measurement for the 1080p gate.

## Current Blender/Cycles source build and HIP smoke

The Blender source remote was refreshed on 2026-07-29 and frozen at clean
`origin/main@4fe17ef6be5d46251fa5e7dbff9018efb1c719d5`, committed the same
day. This is Blender 5.3.0 Alpha, not the packaged 5.2 binary used for the
640×480 reference above.

The checkout is `/home/mike/Projects/blender-cycles`. GNU `time` 1.10 and
Git LFS 3.7.1 were installed from the system package manager. The official
Linux dependency checkout was synchronized with
`make_update.py --no-blender --architecture x86_64` and is pinned at
`lib/linux_x64@ecbd06cf6d2a4aa6b00a61ffb479fc81b17aba08`. The first
configure correctly rejected an LFS pointer in `release/datafiles/startup.blend`;
`git lfs pull` followed by `git lfs fsck` repaired and verified the checkout.

The exact release/headless configuration was:

```bash
cmake -S /home/mike/Projects/blender-cycles \
  -B /home/mike/Projects/blender-build-4fe17ef6 -G Ninja \
  -C build_files/cmake/config/blender_release.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_HEADLESS=ON \
  -DWITH_CYCLES=ON \
  -DWITH_CYCLES_DEVICE_HIP=ON \
  -DWITH_CYCLES_HIP_BINARIES=ON \
  -DCYCLES_HIP_BINARIES_ARCH=gfx1201 \
  -DWITH_CYCLES_DEVICE_HIPRT=OFF \
  -DWITH_CYCLES_DEVICE_OPTIX=OFF \
  -DWITH_CYCLES_CUDA_BINARIES=OFF \
  -DWITH_CYCLES_DEVICE_ONEAPI=OFF \
  -DWITH_CYCLES_ONEAPI_BINARIES=OFF \
  -DWITH_XR_OPENXR=OFF
```

The build was always invoked with `--parallel 32`. An initial timing wrapper
failed before compilation because GNU `time` was not installed. After that
dependency was added, the first long build was externally terminated with
signal 15 at approximately task 3605/6465; there was no compiler error,
kernel OOM report, or swap. Ninja retained the completed graph. The 2863-task
incremental resume passed in 5:21.19 with 2176% average CPU, 3,350,892 KiB
peak resident memory, and no swap. It produced a 7,290,920-byte
`kernel_gfx1201.fatbin` and no unnecessary GPU-architecture fatbins.

The raw build-tree executable then failed its expected pre-install runtime
check because `libopenjph.so.0.25` and the Blender `5.3` data tree were not
staged. The prefix was changed from `/usr/local` to the non-system
`/home/mike/Projects/blender-install-4fe17ef6`; a 32-job refresh and
`cmake --install` produced a self-contained 1.2 GiB installation.
`blender --version` reports Release hash `4fe17ef6be5d`, built 2026-07-29.

A 64×48/1 spp unmodified Lone Monk HIP smoke then selected exactly one
device, `AMD Radeon RX 9070 XT`, and completed:

| Measurement | Current source result |
|---|---:|
| Cycles internal render | `1.011486 s` |
| Process elapsed | `2.359135 s` |
| VRAM baseline | `4,067,905,536 B` |
| Absolute VRAM peak | `6,000,680,960 B` |
| Increase over baseline | `1,932,775,424 B` |
| EXR | 64×48, 43 channels, 132,096/132,096 finite values |

VRAM was sampled every 50 ms through
`tools/measure_amd_vram.py`; the report also records the exact child command
and exit status. This smoke proves that the locally built current-source HIP
path and multilayer EXR are usable. It is not a quality comparison and has no
triptych; triptychs belong to the matched high-sample gate.

## Current-source 1440×1080/256 spp differential

The current-source quality gate uses Blender 5.3.0 Alpha
`main@4fe17ef6be5d46251fa5e7dbff9018efb1c719d5` for both the Cycles pixels
and the evaluated scene exported to Psycles. The source `.blend`, scene,
frame, camera, seed, fixed sample count, and physical RX 9070 XT are shared.
Adaptive sampling and denoising are disabled. The compositor and sequencer do
not participate.

### Current Blender re-export

The `.blend` was re-exported with the locally built current Blender instead
of reusing evaluated geometry from packaged Blender 5.2. This prevents a
version-mismatched geometry evaluation from contaminating a renderer
comparison. The new bundle is
`/tmp/lone-monk-20260729/export-main-4fe17ef6`:

| Export measurement | Result |
|---|---:|
| Export time | `24.59 s` |
| Peak resident memory | `1,357,024 KiB` |
| Geometries / instances | 350 / 7,543 |
| Original material graphs / images | 35 / 47 |
| `scene.json` | 10,834,010 B; SHA-256 `f472dc8b2466b5b11ed3709979ede0144792bc677f267e855e33bf77625f1150` |
| `geometry.bin` | 450,966,096 B; SHA-256 `e5deb7b6e5b7b9d65f119811b7374000727096ce6e0ccda27fde5cbb5078c73c` |

The old and new JSON material/world graph payloads differ only in the
recorded Blender version. The image payloads are byte-identical. Evaluated
geometry differs, as expected across Blender revisions, which is why this
gate uses the current export. Blender still exports raw nodes, sockets,
links, closure topology, and scene metadata; it never evaluates or bakes a
Cycles material result.

### Cycles HIP reference

The reference was rendered through the reusable VRAM wrapper:

```bash
python tools/measure_amd_vram.py \
  --output /tmp/lone-monk-20260729/quality-main-4fe17ef6-1440x1080-256/cycles-vram.json \
  --interval 0.05 -- \
  /home/mike/Projects/blender-install-4fe17ef6/blender \
  /home/mike/Downloads/lone-monk_cycles_and_exposure-node_demo.blend \
  --background --python tools/render_cycles_golden.py -- \
  /tmp/lone-monk-20260729/quality-main-4fe17ef6-1440x1080-256/cycles.exr \
  1440 1080 256 0 \
  --cycles-device HIP \
  --device-name "Radeon RX 9070 XT"
```

The metadata names exactly one enabled device,
`HIP_AMD Radeon RX 9070 XT_0000:03:00`, and records
`scene.cycles.device = GPU`. The 131,175,379-byte EXR has SHA-256
`40e42dedad336882792f9064bb6eecdb65fa09a272c15286ef38fc96692481e3`.
It contains 43 full-float channels and all 66,873,600 values are finite.

### Failed monolithic Psycles dispatch

The first 1440×1080/256 Psycles attempt submitted all 256 samples for every
pixel in one Vulkan compute dispatch:

```bash
env LUISA_XIR_DISABLE_OPTIMIZATION=1 \
    LUISA_SPIRV_OPT_LEVEL=0 \
    LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
  build/bin/psycles_render_blender_scene \
  /tmp/lone-monk-20260729/export-main-4fe17ef6 \
  /tmp/lone-monk-20260729/quality-main-4fe17ef6-1440x1080-256/psycles-vulkan.ppm \
  vk 1440 1080 256
```

RADV reported that the compute shader was cancelled because the context was
guilty of a hard recovery. The kernel journal records a compute-ring timeout,
the `psycles_render_` process as guilty, a device coredump, and a successful
compute-queue reset. This is a bounded-progress failure: one monolithic
dispatch exceeded the AMDGPU watchdog. It is not a shader-quality result.

The process nevertheless returned zero and printed a false render time of
`2.19935 s`. The output proved it was corrupt: 62,207,984 of 62,208,000 EXR
values were zero, only four pixels contained nonzero Combined values, and the
maximum reached 660,171. The observed VRAM range was
4,070,383,616–5,777,596,416 bytes. These numbers are retained as failure
evidence and are excluded from every performance or quality comparison.

### Luisa Vulkan result contract

The false success exposed an independent Luisa backend defect. In Release
builds, `VK_CHECK_RESULT(f)` evaluated `f` and discarded its `VkResult`; a
`VK_ERROR_DEVICE_LOST` from synchronization therefore looked successful to
the caller.

The repair in `LuisaCompute next@d57720955` defines one build-independent
contract:

1. every wrapped Vulkan expression is evaluated exactly once;
2. `VK_SUCCESS` continues;
3. every other result terminates with the expression, symbolic Vulkan error,
   source file, and line.

`test_vk_result_contract` explicitly defines `NDEBUG`, proves the success
expression is evaluated once, and runs `VK_ERROR_DEVICE_LOST` in a child
process that must terminate with `SIGABRT`. The old Release macro exited
normally, so this is a red/green regression. The focused result is 2 tests /
5 assertions. The complete Vulkan SPIR-V path remains 86/86 tests / 2,029
assertions. Luisa CTest passes 115/116 with `-j32`; the sole failure is still
the pre-existing EASTL `fixed_vector` allocation contract.

The regression initially failed to link because its direct use of the Vulkan
logging header also requires `VulkanTools.cpp` for symbolic error strings.
Adding that real dependency to both CMake and xmake fixed the test
infrastructure; no backend behavior was weakened to make the test pass.

### Formally bounded sample dispatches

Psycles now partitions every requested half-open sample interval
`[first, first + count)` into batches `B`. The scheduler maintains these
scene-independent invariants:

```text
B[0].first                  = first
B[i + 1].first              = B[i].first + B[i].count
0 < B[i].count              <= max_samples_per_dispatch
B[last].first + B[last].count = first + count
```

Therefore the batches are an ordered exact cover: no sample is omitted,
duplicated, overlapped, or reordered. Every batch receives the original total
AA sample count and its global first index, so the Luisa device sampler keeps
the same Sobol sequence. Each batch is submitted and synchronized separately,
which bounds watchdog and device-error detection latency. The default maximum
is 8 spp; zero and overflowing intervals are rejected.

`psycles.sample_dispatch_partition` exhaustively checks all first/count/limit
combinations from 0 through 16, the 32-bit endpoint, invalid zero limits,
overflow rejection, exhaustion, minimal batch count, and the exact
256-sample/8-spp decomposition into 32 batches. It is a host scheduling
contract, not a CPU renderer or sampler. A real Vulkan check then rendered
the current Lone Monk export at 64×48/16 spp as one 16-spp dispatch and as
two 8-spp dispatches. Combined PFM was byte-identical and
`oiiotool --diff` passed all channels of the two EXRs. The 32-job Release
build and Psycles CTest pass 13/13. This boundary is published as
Psycles `dcb96e3`.

### Successful bounded Psycles Vulkan run

The accepted run explicitly records the default limit:

```bash
python tools/measure_amd_vram.py \
  --output /tmp/lone-monk-20260729/quality-main-4fe17ef6-1440x1080-256/psycles-vulkan-batched-vram.json \
  --interval 0.02 -- \
  env LUISA_XIR_DISABLE_OPTIMIZATION=1 \
      LUISA_SPIRV_OPT_LEVEL=0 \
      LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
  build/bin/psycles_render_blender_scene \
  /tmp/lone-monk-20260729/export-main-4fe17ef6 \
  /tmp/lone-monk-20260729/quality-main-4fe17ef6-1440x1080-256/psycles-vulkan-batched.ppm \
  vk 1440 1080 256 8
```

The log names `AMD Radeon RX 9070 XT (RADV GFX1201)`, 350 geometries, 7,543
instances, and 37 runtime material programs, including renderer-internal
programs. Scene compilation took `0.66606 s`, warm shader JIT `2.2724 s`,
and render-only `25.9918 s`. The wrapper elapsed time was `29.805725 s`.
No timeout, reset, device loss, guilty context, or AMDGPU recovery appears in
the kernel log for this interval.

The 119,638,054-byte output has SHA-256
`3c887c8ba7221a51951a641f6f63f6d78617341c9bdfe925ec7c1bfdf02cac35`.
It contains 40 full-float channels, both color-space identifiers are
`lin_rec709_scene`, and all 62,208,000 values are finite with zero NaN and
zero Inf.

### Same-device performance and VRAM

| Measurement | Cycles 5.3 Alpha HIP | Psycles Vulkan |
|---|---:|---:|
| Render-only | `18.961390479 s` | `25.9918 s` |
| Whole measured command | `19.439086148 s` | `29.805725258 s` |
| Scene / shader setup reported separately | included in Cycles call | `0.66606 s` / `2.2724 s` |
| VRAM pre-launch baseline | `4,105,527,296 B` | `4,082,360,320 B` |
| Absolute VRAM peak | `6,764,978,176 B` | `5,793,931,264 B` |
| Increase over baseline | `2,659,450,880 B` | `1,711,570,944 B` |

At the comparable render-only boundary, Psycles has `0.729514×` Cycles
throughput and is `1.370775×` slower. There is no speedup claim. Psycles uses
947,879,936 fewer baseline-relative peak bytes (35.64% less) and its absolute
peak is 971,046,912 bytes lower (14.35% less). These are process-window
measurements, not allocator-attributed renderer memory; both absolute and
baseline-relative values are retained in the raw
[Cycles VRAM report](cycles-vram-1440x1080-256-main-4fe17ef6.json),
[accepted Psycles VRAM report](psycles-vulkan-vram-1440x1080-256-main-4fe17ef6.json),
and
[failed monolithic-dispatch report](psycles-vulkan-monolithic-failed-vram-1440x1080-256-main-4fe17ef6.json).

### Numeric pass results

The complete machine-readable report is
[report-1440x1080-256-main-4fe17ef6.json](report-1440x1080-256-main-4fe17ef6.json).
Every compared pass contains 1,555,200 valid pixels and zero invalid pixels.

| Pass | RMSE | Relative RMSE | Mean luminance ratio | Maximum absolute error |
|---|---:|---:|---:|---:|
| Combined | `0.216918692` | `0.135484421` | `1.022434553` | `10.4275970` |
| Diffuse Color | `0.011697795` | `0.061507090` | `1.003732054` | `0.3189842` |
| Normal | `0.029506562` | `0.053085719` | — | `1.3897462` |
| Diffuse Direct | `1.680650711` | `0.183444668` | `1.021852371` | `149.0686340` |
| Diffuse Indirect | `0.247345969` | `0.610184475` | `0.913661167` | `71.9946823` |
| Glossy Color | `0.002655193` | `0.037487039` | `0.999494795` | `0.1132607` |
| Glossy Direct | `0.557576835` | `0.128782547` | `1.015272347` | `138.3923950` |
| Glossy Indirect | `0.190440401` | `0.499699873` | `0.939290990` | `34.4971428` |
| Emission | `0.006447404` | `0.008803128` | `0.999618405` | `0.9512053` |
| Environment | `0.000162833` | `0.105180146` | `0.991469411` | `0.1226361` |
| Transmission Color / Direct / Indirect | `0` | `0` | both zero | `0` |

Compared with the earlier 640×480/64 spp result, Combined RMSE decreases from
`0.262420535` to `0.216918692`, relative RMSE from `0.168320400` to
`0.135484421`, and MAE from `0.071394570` to `0.039865170`. Its luminance
ratio remains high at `1.022434553`. Diffuse and glossy direct means are
2.19% and 1.53% high, while diffuse and glossy indirect remain 8.63% and
6.07% low. The higher-sample result therefore separates persistent transport
energy differences from finite-sample noise; it is not a final 1:1 pass.

### Real triptychs and visual inspection

Every triptych contains the real current Cycles image, the real Psycles
image under the same display mapping, and independently amplified absolute
linear difference. All were generated at one source pixel per panel pixel
and inspected at original resolution:

- [Combined](triptychs-1440x1080-256-main-4fe17ef6/combined.png)
- [Diffuse Color](triptychs-1440x1080-256-main-4fe17ef6/diffcol.png)
- [Normal](triptychs-1440x1080-256-main-4fe17ef6/normal.png)
- [Diffuse Direct](triptychs-1440x1080-256-main-4fe17ef6/diffdir.png)
- [Diffuse Indirect](triptychs-1440x1080-256-main-4fe17ef6/diffind.png)
- [Glossy Color](triptychs-1440x1080-256-main-4fe17ef6/glosscol.png)
- [Glossy Direct](triptychs-1440x1080-256-main-4fe17ef6/glossdir.png)
- [Glossy Indirect](triptychs-1440x1080-256-main-4fe17ef6/glossind.png)
- [Emission](triptychs-1440x1080-256-main-4fe17ef6/emit.png)
- [Environment](triptychs-1440x1080-256-main-4fe17ef6/env.png)
- [Transmission Color](triptychs-1440x1080-256-main-4fe17ef6/transcol.png)
- [Transmission Direct](triptychs-1440x1080-256-main-4fe17ef6/transdir.png)
- [Transmission Indirect](triptychs-1440x1080-256-main-4fe17ef6/transind.png)

The visual findings agree with the metrics:

- camera, framing, instance/geometry silhouettes, architecture, principal
  texture placement, and large-scale material assignment align; there is no
  flip, displaced scene, missing object, full-frame corruption, or
  watchdog-residue pattern;
- Diffuse Color, Glossy Color, and Normal are close at ordinary viewing
  scale. Their amplified residuals concentrate on visibility edges,
  high-frequency bump/texture detail, foliage, books, and a few bright
  surfaces rather than a material-slot permutation or normal-space rotation;
- direct diffuse and glossy illumination appears on the same surfaces.
  Psycles is subtly brighter and the amplified difference contains
  high-variance highlights and edges;
- diffuse and glossy indirect spatial structure aligns, but Psycles is
  visibly and numerically darker overall. This is the primary persistent
  algorithm-alignment target;
- Emission is visually almost identical. Environment contains only sparse,
  very low-energy pixels and has low absolute error;
- all three transmission panels and differences are black because both
  renderers produce exact zero for this frame.

No exposure compensation, denoising, material pre-bake, pass recombination,
or image-space fitting was applied. The next diagnosis must use these direct
and indirect pass residuals plus current Cycles semantics to identify a
Luisa-path implementation defect and add its regression before changing the
renderer.
