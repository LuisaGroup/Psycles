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
CTest passes 114/115 with 32-way scheduling. The sole CTest failure remains
the independently reproducible, pre-existing EASTL `fixed_vector` allocation
contract. Psycles passes 12/12 using a 32-job build and test schedule. The
published Luisa boundary is `next@eb167454a`.

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

A true cold `column marble` rerun after `next@eb167454a` produced 237,944
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

There are intentionally no placeholder Lone Monk triptychs. The controlled
material variants, including the repaired single-material run, have no
equivalent Cycles quality reference and cannot be presented as one. The
required triptych for each linear pass is:

1. Cycles;
2. Psycles using the same display transform and exposure;
3. independently amplified absolute linear difference, with the multiplier
   printed in the panel title.

The accompanying JSON must record RMSE, relative RMSE, energy or luminance
ratio, maximum error, invalid pixels, device inventory, cold/warm setup time,
render-only time, peak memory, and same-device speedup. Images must be
inspected at original resolution for silhouette, geometry-edge, missing-light,
texture-coordinate, normal, systematic shading, and color/exposure errors.

The triptych pipeline itself is already covered by committed Cycles/Psycles
checks:

- [flat-light Combined triptych](../flat-light-vk/triptychs/combined.png);
- [transparent-mix Combined triptych](../transparent-mix-vk/triptychs/combined.png);
- [transparent-data Combined triptych](../transparent-data-pass-vk/triptychs/combined.png).

Those focused images are not substituted for Lone Monk. The unmodified
Psycles material set now produces a linear EXR, but Lone Monk triptychs remain
pending until current Cycles and Psycles are rendered with matching settings
at 480p or higher. A 1080p run will be attempted if GPU memory permits.
