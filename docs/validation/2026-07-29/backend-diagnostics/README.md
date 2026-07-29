# Lone Monk backend diagnostics — 2026-07-29

This record separates three backend findings made after the Filter Glossy
alignment boundary:

- the historical full-scene Luisa HIP result is faster than Psycles Vulkan by
  its reported render timer, but is not a valid Cycles performance comparison
  because it dropped visible scene instances;
- the GFX12 large-TLAS traversal defect responsible for that structural miss
  is fixed and regression-tested in Luisa
  `next@3dc93b4c632e423e2ff44e308a2ff4656332f78a`;
- the current strict-native Vulkan cold JIT is dominated by formal XIR CFG
  restructuring, not by CMake compilation, SPIR-V validation, or rendering.

The full HIP quality and performance result remains unaccepted: a separate
cumulative HIPRT high-quality BLAS-build failure still prevents a clean
350-geometry rerender with the traversal fix.

Cycles remains the only rendering oracle. No CPU renderer, sampler, BSDF
mirror, material baking, exposure fit, or image-space correction was used.
The compared scene is the current Blender/Cycles
`main@4fe17ef6be5d46251fa5e7dbff9018efb1c719d5` Lone Monk export with 350
geometries, 7,543 instances, 35 original raw material graphs, and 47 images.

## HIP timing is nominal, not accepted

The full 1440×1080/256 spp HIP attempt used the same bounded 8-spp dispatch
contract as the accepted Vulkan result:

```bash
python tools/measure_amd_vram.py \
  --output psycles-hip-1440x1080-256-vram.json \
  --interval 0.05 -- \
  build/bin/psycles_render_blender_scene \
  export-filter-glossy-main-4fe17ef6 \
  psycles-hip-1440x1080-256.ppm \
  hip 1440 1080 256 8
```

The process returned zero and reported 350 geometries, 7,543 instances, and
37 runtime material programs. Scene compilation took `2.91319 s`, the main
shader JIT took `75.8138 s`, render-only took `24.8549 s`, and the complete
monitored command took `104.524574 s`. Peak VRAM was 6,267,981,824 bytes,
2,164,301,824 bytes above the pre-launch baseline. The raw measurement is
[psycles-hip-1440x1080-256-vram.json](psycles-hip-1440x1080-256-vram.json).

If the image were correct, the render timer would be `1.310816×` Cycles HIP,
or 31.08% slower with `0.762883×` throughput. It would also be 9.43% lower
than Psycles Vulkan's `27.4415 s`. Those are diagnostic timer ratios only.
They are excluded from accepted performance claims because the rendered
workload is observably incomplete. The last valid same-device comparison
therefore remains Psycles Vulkan at `1.447230×` Cycles, or 44.72% slower.

The HIP EXR SHA-256 is
`ee9d239c227ff74b0bd92ca4e6a77f5155b2164be875c761343c67b49b173afa`.
The complete numerical comparison is
[hip-cycles-full-report-1440x1080-256.json](hip-cycles-full-report-1440x1080-256.json).
Every pass contains 1,555,200 finite pixels, but finite output is not the same
as correct output:

| Pass | HIP/Cycles luminance | RMSE | Relative RMSE |
|---|---:|---:|---:|
| Combined | `1.035310×` | `0.936863` | `0.585152` |
| Diffuse Color | `0.970970×` | `0.0619081` | `0.325513` |
| Diffuse Direct | `0.741783×` | `5.57446` | `0.608458` |
| Diffuse Indirect | `0.756938×` | `0.310153` | `0.765124` |
| Glossy Direct | `0.748428×` | `2.43645` | `0.562742` |
| Glossy Indirect | `0.738040×` | `0.243809` | `0.639734` |
| Emission | `0` | `0.732399` | `1.0` |
| Normal | — | `0.140065` | `0.251993` |

The Combined mean happens to be close because missing shaded geometry is
replaced by a bright sky contribution. Pass-local energy and spatial error
show that this is cancellation, not agreement.

### Original-resolution visual inspection

The following real triptychs were generated with one source pixel per panel
pixel and opened at original resolution:

- [Combined](hip-triptychs/combined.png)
- [Normal](hip-triptychs/normal.png)
- [Diffuse Color](hip-triptychs/diffcol.png)

All three independently show the same structural defect. The upper central
church storey and roof present in Cycles are absent in Psycles HIP, and the
background is visible through the resulting hole. The Normal panel contains
the same missing region, proving that the failure precedes lighting-pass
classification and EXR packing. Foreground books, arches, columns, and many
other instances remain visible, so this is not a total acceleration-structure
failure.

At the exact 64×48/1 spp seed used for a backend smoke check, HIP and Vulkan
differ in 2,487 of 3,072 pixels above `1e-6`; the same missing geometry is
visible. A controlled diagnostic changed both mesh and top-level
acceleration-structure build hints from the default high-quality mode to
`PREFER_FAST_BUILD`. It changed 507 pixels relative to the default HIP result
but did not restore the church/roof instances. The source was then restored
and rebuilt with all 32 jobs. This rejects build-hint selection as a
sufficient explanation and motivated separating instance transforms, TLAS
cardinality, and traversal state in the next isolation.

The first 12-instance isolation attempt then exposed an earlier HIPRT
boundary while retaining all 350 BLAS inputs. Near the 288th high-quality
geometry build, HIPRT's `oroModuleLaunchKernel` returned `invalid argument`
and the process aborted. An immediate repeat reached the same region, stayed
at 100% GPU, and eventually produced an AMDGPU `gfxhub` page fault for an
unmapped page in the HIPRT build. Geometry 288 (`Plane.017`) has 313,392
finite vertices and 104,464 triangles; all indices are in range, with no
duplicate-index or zero-area triangles. Thus malformed basic mesh data is not
a sufficient explanation. This builder/lifetime failure and the full-scene
missing-instance result may share a cause, but that relationship is not yet
claimed. The next regression boundary is the smallest ordered mesh set that
reproduces the HIPRT fault.

## GFX12 large-TLAS traversal overflow

The controlled invariant is independent of Lone Monk:

> Adding instances whose visibility mask excludes every tested ray must not
> change any previously visible hit.

The regression builds one visible triangle twice. The baseline TLAS contains
only that instance. The padded TLAS contains 1,024 overlapping instances,
with instance 8 visible and every other instance assigned visibility mask
zero. For the same ray and mask, the exact result must remain a surface hit:

| Query form | Baseline | Padded |
|---|---:|---:|
| closest instance | `0` | `8` |
| any hit | `true` | `true` |
| committed ray-query instance | `0` | `8` |
| committed hit type | `Surface` | `Surface` |

Before the repair, the padded result could become a miss even though the
baseline passed. Increasing the GFX12 LDS short stack merely moved the
failure to a larger TLAS and was rejected as a solution.

RDNA4's `ds_bvh_stack_push8_pop1_rtn` short stack is circular. Its raw
overflow sentinel is `0xffffffff`, while normal terminal underflow is
`0xfffffffe`. The previous HIP wrapper exposed both outcomes to traversal as
the same `InvalidValue`, so overflow silently discarded unvisited BVH nodes.
The repair preserves this distinction and follows two callback-safe policies:

- closest-hit and unsuccessful any-hit traversal restart from the original
  ray on a cold, out-of-line checked software stack; a witnessed any-hit is
  already sufficient and need not restart;
- resumable ray queries retain a software traversal stack in their persistent
  state, because restarting after a candidate callback could replay observable
  user code.

Both software paths trap on their explicit capacity boundary rather than
converting stack exhaustion into a miss. This is a traversal-state invariant,
not a scene-, instance-, transform-, or node-index special case.

The final 1,024-instance regression passed six HIP runs on the RX 9070 XT
(five consecutive runs plus one after rebasing onto the latest Luisa
`origin/next`), with 131 assertions per run. The same test passed Vulkan with
131 assertions. The following HIP suites also passed:

| Suite | Assertions |
|---|---:|
| accel build modes | `300` |
| motion ray query | `145` |
| static motion instance | `276` |
| matrix motion instance | `80` |
| SRT motion instance | `104` |
| motion mesh | `470` |

All affected targets were rebuilt with `cmake --build ... --parallel 32`.
Exploratory overlapping-TLAS runs at 7,543 and 16,384 instances also passed
before the diagnostic size override was removed from the committed
regression.

### Real-scene isolation and visual inspection

The real-scene check retains the 12 Lone Monk church/roof geometries and all
7,543 TLAS entries. The other 7,531 entries reference a valid geometry with
visibility mask zero. This makes the large input observationally equivalent
to the 12-instance compact control while retaining the production 37-program
material kernel. The large-scene `scene.json` SHA-256 is
`9738546aa71039ad9fda341a6ff0c5a92387896e436fcd95d44665f28743b100`.

The production shader cache was disabled for exactly one post-repair render,
then immediately restored and rebuilt with all 32 jobs. At 128×96/1 spp, the
old large TLAS produced only the environment. The repaired large TLAS restores
the same church and roof structure visible in the compact control. The real
triptych below was opened and inspected at original nearest-neighbor scale:

[![HIP GFX12 visibility-padding triptych](hip-tlas-overflow/visibility-padding-triptych.png)](hip-tlas-overflow/visibility-padding-triptych.png)

The triptych SHA-256 is
`c2516418af2a0e479178293139b4959603a70350b03ffdb87a4ea17b0c9c1ec5`.
The repaired PPM and multilayer EXR SHA-256 values are
`2194385806e16d383736e9f3d2254fca112bb24bd2451d483a36b5be72f7e9f0`
and
`1a44df0bbe50fcb69be4345e1de58809565df7869d5904c6e4193efa6d1783b2`.
This low-spp isolation proves structural traversal correctness only; it is
not a Cycles quality or speed acceptance result.

The uncached command completed in `352.20 s` at 98% average CPU and
2,626,968 KiB peak resident memory. Psycles reported `351.277 s` for shader
JIT and `0.108312 s` for the tiny render. HIP LLVM code generation took
`26.5791 s` and emitted 4,180,808 bytes; the timestamped interval from code
emission to the completed LTO/module-load message was another `268.872 s`.
Thus this production HIP cold JIT is also effectively single-core and needs
separate performance work. The tiny 1-spp render timer is deliberately not
compared with Cycles. The machine-readable record is
[hip-gfx12-tlas-overflow.json](hip-gfx12-tlas-overflow.json).

## Strict-native Vulkan cold-JIT profile

The timing input retains all 350 geometries, 35 raw material graphs, and 47
images, but selects only 12 church/roof instances. This keeps the production
37-program material kernel while providing the next HIP isolation boundary.
Its `scene.json` SHA-256 is
`21ad3042c3e06eb2ccdc4294ef550cd1ca5efe5965eae7946f5ac461f7d95ce1`.

The production shader cache was temporarily disabled for this one run so that
an interrupted earlier compile could not turn the measurement into a cache
hit. The exact production kernel and strict-native configuration were used:

```bash
env LUISA_XIR_DISABLE_OPTIMIZATION=1 \
    LUISA_SPIRV_OPT_LEVEL=0 \
    LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
    LUISA_XIR_TRACE_PASSES=1 \
    LUISA_LOG_LEVEL=verbose \
  build/bin/psycles_render_blender_scene \
  church-roofs vk-128x96-1.ppm vk 128 96 1 1
```

The cache setting was restored immediately afterward, followed by a
32-thread rebuild. The worktree was clean after restoration. The complete
bounded measurement is
[vulkan-jit-stages.json](vulkan-jit-stages.json).

| Cold-JIT stage | Seconds | Share of `240.23 s` |
|---|---:|---:|
| AST to XIR | `16.1140` | `6.71%` |
| Complete SPIR-V XIR legalization | `148.5515` | `61.84%` |
| └ `destructure-cfg` | `15.6820` | `6.53%` |
| └ `restructure-cfg` | `132.0029` | `54.95%` |
| Native SPIR-V emission | `37.9740` | `15.81%` |
| SPIR-V validation | `1.1970` | `0.50%` |
| Result/property handoff | `0.1610` | `0.07%` |
| RADV compute-pipeline creation | `36.0230` | `15.00%` |
| Other timestamp gap | `0.2095` | `0.09%` |

The stages account for the entire reported JIT within timestamp resolution.
The full process took `241.13 s` and peaked at 3,471,716 KiB resident memory.
The emitted module contains 2,672,528 SPIR-V words and 26 bindings.

The legalization input to the large definition contains 1,654 blocks,
208,018 instructions, 534 raw conditional terminators, and one raw indexed
terminator. `destructure-cfg` lowers 2,262 ifs, five switches, ten loops,
33 simple loops, and 39 breaks. Pre-restructure `reg2mem` then reports
361,084 hoisted allocas. `restructure-cfg` consumes `132.0029 s` while
creating 2,483 structured selections, 46 loops, and eight switches. The
post-boundary `mem2reg` promotes 359,804 allocas and completes in only
`0.4133 s`.

The process averaged approximately 198% CPU during the long passes. Thus
`cmake --build --parallel 32` correctly uses all requested build workers, but
cannot make this single shader's largely two-core JIT use 32 threads. The
formal CFG round trip is the first optimization target: eliminating only
SPIR-V validation would save about 0.5%, while improving
`restructure-cfg` addresses about 55% of cold JIT. Any repair must preserve
the existing structured-CFG invariants and regressions; replacing it with
scene-specific shortcuts would be invalid.
