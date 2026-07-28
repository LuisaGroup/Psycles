# Psycles validation record — 2026-07-29

This record covers the renderer boundary published as Psycles
`main@a10d686` and LuisaCompute `next@f83725d27`. It records the commands,
numeric results, visual inspection, and known limitations for the AMD GPU
bring-up, Cycles flat-light distribution, XIR control-flow repair, and
multilayer OpenEXR output.

## Verdict

- Psycles configures and builds the Luisa fallback, HIP, and Vulkan backends
  together. All three requested backend targets are strict CMake
  postconditions rather than optional best-effort features.
- The complete Psycles gate passes 12/12 after a 32-job build.
- Luisa's focused `restructure_cfg` gate passes 49/49 tests and 1003
  assertions. The complete Luisa unit gate passes 87/88; the sole failure is
  the pre-existing EASTL `fixed_vector` allocation contract described below.
- Vulkan and HIP both render the focused flat-light scene on the Radeon RX
  9070 XT. Vulkan also passes the two transparent-closure probes.
- Psycles now writes one full-float multilayer OpenEXR with
  `ViewLayer.<pass>.<component>` channel names. The end-to-end comparison reads
  Cycles EXR and Psycles EXR directly.
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
| Psycles implementation | `a10d686` on `main` |
| Psycles input boundary | `32d4217dc543b1778729f23a18f3f3143e001a24` |
| LuisaCompute | `f83725d27502f79e30aac50eba851e410912bfc5` on published `next` |
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

The relevant published Luisa commits are:

| Commit | Result |
|---|---|
| `6ead8e714` | backend resource paths work when Luisa is built as a subdirectory |
| `83a04feb8` | preserve CFG and SSA invariants during restructuring |
| `f83725d27` | preserve executable semantics for update regions, re-entry, affine state, and disconnected edges |

The focused and complete commands were:

```bash
./build/luisa-tests/bin/test_xir_pass_restructure_cfg
ctest --test-dir build/luisa-tests --output-on-failure -j32
```

The focused binary passes 49 tests / 1003 assertions. The complete suite
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

The flat distribution follows the current Cycles flat-light construction:
world-space emissive-triangle area weighting, uniform analytic/background
lamp probability, and a 50/50 class split when triangles and lamps both
exist. Selection and PDF lookup use one uploaded CDF and one Luisa upper-bound
callable. Material emission-sampling metadata and world-sampling metadata are
imported with the original material graphs.

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
