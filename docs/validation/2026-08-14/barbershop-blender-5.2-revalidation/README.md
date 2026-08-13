# Barbershop Blender 5.2 reference and HIP performance revalidation

## Outcome

The official Barbershop Interior file is not corrupt. The structured dark
floor, ceiling, and cupboard differences previously treated as a traversal
oracle came from comparing outputs evaluated by different Blender revisions:
a Blender 5.3 Alpha export against a Cycles reference from another build.
Re-evaluating both sides with Blender 5.2 LTS restores the expected scene and
invalidates the synthetic whole-scene candidate-completion path.

The corrected Psycles image has the same broad geometry, textures, ceiling,
floor, and cupboard structure as Cycles 5.2. At 640x480 and 64 fixed samples,
Combined relative RMSE is `0.107473` and mean luminance is `0.999559x`
Cycles CPU. Coherent indirect/glossy residuals remain and are not classified
as mere floating-point noise.

On the RX 9070 XT, the best current Psycles HIP mode is staged wavefront.
Fresh three-run medians are `5.155 s` for Psycles versus `1.748 s` for
Cycles HIP at 64 spp (`2.95x` slower). The preceding 256-spp checkpoint was
`20.010 s` versus `6.085 s` (`3.29x` slower). These are render-only intervals
with adaptive sampling and denoising disabled.

## Exact reference identity

The input file was downloaded again from the official benchmark directory.
The old and new copies are byte-identical:

```text
SHA-256 95972b56180462cac47ec82f3a755bd9111ec18ca37a6196a319c013db994130
```

The canonical reference and export use Blender 5.2.0 LTS at
`fbe6228777e7` (`blender-v5.2-release`). The locally checked-out source is at
`/home/mike/Projects/blender-cycles`; the configured build enables HIP and
precompiled HIP kernels, disables CUDA/OptiX/oneAPI, and installs under
`/home/mike/Projects/blender-install-5.2`.

The same file produces materially different final render dependency graphs:

| evaluator | triangle meshes | curves | instances | triangles | geometry payload |
| --- | ---: | ---: | ---: | ---: | ---: |
| Blender 5.3 Alpha `ec438d7429e5` | 1,649 | 6 | 2,565 | 23.75 M | 4.9 GB |
| Blender 5.2 LTS `fbe6228777e7` | 1,055 | 4 | 1,109 | 13.00 M | 2.7 GB |

Materials, lights, and image datablocks are essentially unchanged; the
evaluated geometry/instance population is not. Toggling the 5.3 scene from
Cycles to EEVEE and back did not change its result, so the distinction is the
versioned final-depsgraph evaluation rather than a renderer-engine refresh.
The 5.2 and 5.3 Cycles Combined images have RMSE `0.087516`; 5.2 is `2.2773x`
brighter on average. Normal and Diffuse Color differ much less, which locates
the largest change in transport/visibility rather than UV orientation.

The tooling now records `{version, version_cycle, version_tuple, build_hash,
build_branch, build_type}` in both golden metadata and `scene.json`. The
benchmark runner requires exact equality before starting Psycles, including
for `--reuse-export`. It also preserves authored-disabled view layers instead
of silently enabling them. Host and Blender-side regressions cover both
contracts.

## Traversal correction

Let `E_B(ray)` be the ordered candidate sequence enumerated by acceleration
backend `B`, and `P(ray, candidate)` be the shared exact Cycles triangle
predicate, including visibility, source/light exclusion, closed interval,
and stable identity ordering. Traversal is now defined only as

```text
Accepted_B(ray) = Order(Filter(P(ray, c), c in E_B(ray))).
```

The removed implementation replaced `E_B` with a union containing host-built
whole-support and closed-AABB relations. Even though every injected candidate
was subsequently tested geometrically, the union was not the candidate set
used by Cycles' selected device backend. It could therefore manufacture a
closed-endpoint blocker that neither HIPRT nor the matched Cycles render
enumerated. This was a semantic error, not an insufficiently precise overlap
classification.

Removing completion deletes the dense/sparse source lookup, per-instance
coincident rings, primitive overlap tables, and device-side whole-scene
candidate walks. The ordinary exact per-candidate resolver, widened
closed-distance tie query, Pluecker predicate, and stable order remain. The
focused fallback/HIP/Vulkan traversal and direct-light tests all pass.

On the old mismatched 5.3 bundle, removing the synthetic scan changed
Barbershop HIP from `54.50 s` to `14.68 s` at 640x480/64 spp. That number is
useful only as evidence of the implementation cost; it is not a current
Cycles comparison. On the corrected 5.2 bundle, staged wavefront renders in
about `5.2 s`.

## Corrected image comparison

The 5.2 export contains 1,055 meshes, four curve geometries, 1,109 instances,
564 runtime materials, and 189 surface queue keys. At 640x480/64 spp:

| pass | relative RMSE | Psycles/Cycles mean luminance |
| --- | ---: | ---: |
| Combined | 0.107473 | 0.999559 |
| Normal | 0.018546 | 1.003999 |
| Diffuse Color | 0.080598 | 0.997175 |
| Diffuse Direct | 0.073545 | 0.999622 |
| Diffuse Indirect | 0.404321 | 1.004660 |
| Glossy Color | 0.004834 | 1.000667 |
| Glossy Direct | 0.176833 | 0.996218 |
| Glossy Indirect | 0.540470 | 1.001158 |
| Emission | 0.001513 | 1.000027 |
| Environment | 0.005992 | 1.000958 |

The identity orientation is decisively best for every spatial pass. Visual
inspection of the original-resolution Combined, Diffuse Color, Normal,
Diffuse Direct, and indirect/glossy triptychs confirms that the large black
ceiling/floor holes are gone and the right cupboard is no longer replaced by
a different evaluated object population. Remaining residuals are coherent in
indirect transport and high-energy glossy paths; those remain active parity
work.

![Combined: Cycles 5.2, Psycles HIP, absolute difference](triptychs/combined.png)

![Diffuse Color: Cycles 5.2, Psycles HIP, absolute difference](triptychs/diffcol.png)

![Normal: Cycles 5.2, Psycles HIP, absolute difference](triptychs/normal.png)

![Diffuse Direct: Cycles 5.2, Psycles HIP, absolute difference](triptychs/diffdir.png)

![Diffuse Indirect: Cycles 5.2, Psycles HIP, absolute difference](triptychs/diffind.png)

## Scheduler matrix and profiler result

All modes below use the same 5.2 bundle, RX 9070 XT, 640x480, 64 samples,
Tabulated Sobol sequence, 64 samples per host batch, 32-thread continuation
blocks, and no adaptive sampling or denoising. Scene compilation and shader
JIT are excluded.

| renderer/mode | render-only | slowdown vs Cycles HIP |
| --- | ---: | ---: |
| Cycles 5.2 HIP | 1.748 s | 1.00x |
| Psycles staged wavefront | 5.155 s | 2.95x |
| Psycles per-sample megakernel | 6.605 s | 3.77x |
| Psycles pixel-loop megakernel | 8.160 s | 4.66x |
| Psycles ordinary wavefront | 10.245 s | 5.85x |
| Psycles graph wavefront | 13.600 s | 7.77x |
| Psycles persistent | 23.435 s | 13.39x |

Staged wavefront is `20.7%` lower latency than the same per-(pixel,sample)
topology without coroutines, so this checkpoint demonstrates a real scheduler
benefit. Graph counter readback is already batched: its largest 64-spp chunk
read only 8,064 bytes in 72 readbacks. Its loss comes from sweep/queue policy
and its optional giant state-machine tail, not bulk PCIe transfer.

At 256 spp, rocprofv3 maps the dominant continuation kernels using their
scheduler dispatch counts:

| continuation | GPU kernel time | share of 20.057 s wall | private segment per work-item | VGPR |
| --- | ---: | ---: | ---: | ---: |
| `shade_surface` | 10.570 s | 52.7% | 40,384 B | 256 |
| `intersect_closest` | 6.530 s | 32.6% | 1,344 B | 256 |
| `shade_volume` | 0.476 s | 2.4% | 1,656 B | 256 |
| `shade_light_forward` | 0.118 s | 0.6% | 1,056 B | 256 |

The first two continuations account for `85.3%` of render wall time.
`shade_surface`'s 40 KB private segment proves severe register spilling;
the 784-byte coroutine frame alone does not explain the gap. The next
performance work is therefore closure-local live-range contraction and
generation of only reachable closures, followed by HIPRT closest-hit path
experiments. Block-size or queue-readback tuning cannot cure this profile.

Graph and persistent cold compilation remain separate compiler problems. The
graph tail kernel spent about 79% of sampled host compile time in LLVM IPSCCP,
which materializes a dense per-member lattice over the giant aggregate state
machine. A formal sparse aggregate projection or an earlier demanded-mask
lowering is required; disabling SCCP for this scene would be an ad hoc fix.

## Post-IPO HIP large-return ABI correction

The 189 mutually exclusive surface-topology callables return a roughly
144-byte `SurfaceSampleCall` aggregate. AMDGPU's generated-function return
convention has only 32 32-bit VGPR locations. A return beyond that boundary is
lowered to a hidden caller stack object; because the old IR had one independent
object per static call site, the 189 alternatives accumulated linearly even
though only one executes for a hit.

The Luisa HIP backend now applies a post-IPO ABI transform to every supported
generated callable whose conservatively legalized return exceeds 32 VGPR
locations:

```text
Ret f(args...)  ->  void f(private Ret *result, args...)
```

All calls of one exact return type in a caller share one private result slot,
and each call is followed immediately by the defining load. Hence the slot
live intervals are disjoint; recursive calls still have distinct machine
frames. Running after IPO keeps the callable body in SSA form during ordinary
optimization. The pass atomically rejects external/address-taken functions,
non-default calling conventions, COMDAT/GC or exceptional ABIs, semantic
metadata, operand bundles, tail annotations, fast-math call assumptions, and
indexed `allocsize` attributes rather than partially remapping an unproved
contract.

For the production Barbershop kernel, the pass transformed 184 surviving
generated functions at 189 call sites into one shared result slot, moving
35,328 bytes of aggregate return ABI behind explicit storage. Same-command
rocprofv3 traces at 640x480/64 spp report:

| variant / continuation | calls | GPU kernel time | private segment per work-item | VGPR |
| --- | ---: | ---: | ---: | ---: |
| before / `shade_surface` | 583 | 2700.162 ms | 40,384 B | 256 |
| after / `shade_surface` | 583 | 2688.286 ms | 4,288 B | 256 |
| before / `intersect_closest` | 475 | 1662.468 ms | 1,344 B | 256 |
| after / `intersect_closest` | 475 | 1667.342 ms | 1,344 B | 256 |

Thus the static private segment falls by `89.38%`, but `shade_surface` kernel
time changes by only `-0.44%` and the warm render median by about `-0.52%`,
both near run-to-run noise. The old ABI reserved 189 disjoint static objects,
while a dynamic path touched only one return; sharing removes the capacity
pathology without removing the remaining dynamic store/load work. The
unchanged 256-VGPR count identifies live-range/instruction pressure inside
surface evaluation as the next target. `intersect_closest` remains the other
large cost and is unaffected, as expected.

An exact pass-disabled/pass-enabled A/B was compared against a repeated
pass-disabled render for all Combined, Normal, Albedo, twelve light passes,
and volume guiding outputs. The two comparisons have the same notable noise:
Combined relative RMSE `4.4494e-5`, Diffuse Indirect `0.00270209`, and Glossy
Indirect `0.000218899`; every p99 pixel error is zero. The isolated bright
pixels therefore come from existing atomic/scheduling nondeterminism, not the
ABI transform. Original-resolution visual inspection of Combined, Diffuse
Indirect, and Glossy Indirect triptychs found no structural difference. The
scene-level Cycles/Psycles triptychs above remain the visual parity record.

The formal boundary and rejection-domain regressions contain 122 assertions.
The HIP LLVM pipeline and callable graph suites pass, as do full parallel
Luisa and Psycles builds. Film/light regressions pass on fallback, HIP, and
strict native XIR-to-SPIR-V Vulkan, together with the sample-dispatch
partition test.

## Commands and gates

The corrected staged run is:

```text
./build/bin/psycles_render_blender_scene \
  <blender-5.2-export> out.ppm hip 640 480 256 64 \
  - 0 0 0 0 256 - 1 0 wavefront-staged \
  32 32768 32 1 1 0 4 2 4096 131072 0 0 1
```

The profiler adds:

```text
LUISA_CORO_WAVEFRONT_STATS=1 rocprofv3 \
  --kernel-trace --memory-copy-trace --scratch-memory-trace --stats -- \
  <command-above>
```

All available host cores were used for builds and tests. The focused gates
cover scene planning, traversal, direct lighting, shader identity, benchmark
identity, and Blender export/golden behavior. The final traversal tests pass
on fallback, HIP, and strict native XIR-to-SPIR-V Vulkan.

The complete suite reports 264/266 passing on this host. The two failures are
existing strict Vulkan numeric baselines (`luisa_area_light_forward_vk` and
`luisa_volume_path_vk`), with stable errors between roughly `1e-5` and
`1.3e-4`. A clean worktree at the pre-change `origin/main` commit `06575e4`,
built against the identical Luisa commit and native XIR-to-SPIR-V route,
reproduces every reported value exactly on the same RX 9070 XT/RADV stack.
They are therefore recorded as a pre-existing backend/driver baseline drift,
not hidden by widening tolerances and not attributed to this traversal
change.
