# Native SVM default and geometry/lamp consumers

The native Cycles 5.2.1 SVM is the scene compiler's default surface path.
Neither an unset `PSYCLES_NATIVE_CYCLES_SVM_SURFACE` nor an explicit `0`
selects the old executor. Both cases are checked by the zero-BSDF whole-render
regression before its unchanged original-Cycles trace/image checks.

This is not a claim that SurfaceProgram has been completely deleted.

## Changes and structural corrections

- Delete the old 496-line main surface geometry dispatcher. Both main and
  shadow setup use the native KernelObject/triangle/curve images, not eager
  SurfacePoint attribute evaluation. A scene without curves records no curve
  setup branch; curve-only scenes do not record triangle accesses.
- Delete the legacy shadow-material replay branch. The production shadow
  traversal factory is separate from shadow shading, so traversal-only
  callers do not construct a material program or require shader resources.
- Port original `curve_shader_setup`: containing-curve identity, packed
  segment, endpoint-key clamping, ribbon u/v, object-space D*t normalization,
  rounded ribbon normal, world transforms, Ng, dPdu/dPdv, and the common
  backfacing/differential tail. Primitive type reaches SVM without a hardcoded
  triangle conversion. Other curve shapes/motion remain outside scene admission.
- Replace forward analytic-lamp graph replay with the native light-domain
  SVM and KernelShader constant-emission fast path. Black and signed constant
  records skip the stream; no legacy material/parameter resources are needed.
- Remove three light metadata fallbacks to SurfaceProgram. Mesh-light and
  analytic-light metadata use the native compiler's material facts.
- Use the native identity plan for instance, light and background object IDs.
  Authored scenes without explicit IDs previously let the resource loader and
  SVM image choose different coordinates. A mixed explicit/implicit-ID case
  is now checked after actual device upload. Graph-bearing lamps use their
  compiled shader index; non-node unit lamps retain exported trace identity.

The default suite exposed two fixture/integration issues. The authored
displacement regression's invalid object coordinate caused a HIP memory
fault and is corrected by the shared identity plan. The area-light fixture
provided only a legacy attribute hash: it now also supplies the existing
attribute's symbolic name required by Cycles SVM. Original expected pixels
and traces were not changed. A separate non-node lamp trace check guards
against losing its exported shader identity during material remapping.

The traversal-only regression previously constructed unused legacy shading
alongside its intersection callable. It now uses the same extracted
INTERSECT_SHADOW factory as the renderer, retaining all original intersection,
storage, coroutine reuse and resource-shape checks. Native shadow shading is
covered separately by its original-Cycles full-state oracle.

## Independent curve oracle

Authority: `/home/mike/Projects/blender-cycles-trace-5.2`,
`cb168525138fecc792cc393f94afc39582b0103c`,
`kernel/geom/curve_intersect.h` and `kernel/geom/shader_data.h`.

`tests/cycles_svm_curve_setup_fixture.h` supplies inputs only. The original
HIP `ribbon_intersect` and `shader_setup_from_ray` produce the checked-in
`tests/data/cycles_svm_curve_setup.txt`. No alternate reference renderer or
Psycles-generated expected values are involved.

```sh
/opt/rocm/bin/hipcc --offload-arch=gfx1201 -DHIPCC -std=c++20 -O3 -ffast-math \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles -I include \
  tools/cycles_svm_curve_setup_oracle.hip -o /var/tmp/curve-setup-oracle
/var/tmp/curve-setup-oracle
```

Twenty-four cases cover first/interior/last segments, signed ribbon v,
non-uniform scaling, negative scaling, pre-applied transforms, nonzero
curve/key offsets, permuted object IDs, zero/nonzero differentials and time.
Both production main setup and shared ray/shadow setup compare all 28 float
and eight integer fields. The test has no Accel, triangle image, legacy curve
key/attribute arrays, SurfaceProgram or parameter buffers. Fast math is on.
The existing complete curve-path render additionally checks Hair Info/UV,
primary-hit coordinates and curve shadows in mixed geometry.

## Validation evidence

Evidence directory: `/var/tmp/psycles-native-volume-svm-06XnDX`.

- All-target build passes with `cmake --build build --parallel 32`
  (`native-default-subsurface-build.log`).
- Complete HIP selection: 171/171, 490.95 s
  (`native-default-verified-hip.log`).
- Complete fallback initially passes 172/173, 498.84 s: the persistent case
  exceeds the backend's fixed 4 MiB LLVM barrier-coroutine arena
  (`native-default-verified-fallback.log`). The 2026-09-08 Luisa follow-up
  fixes that allocator with reusable, pointer-stable chunks, not per-lane
  malloc/free. A 4.5 MiB live-frame regression uses one overflow chunk;
  subsequent epochs perform no additional heap allocation. Individual frames
  larger than 4 MiB are also tested. This storage is distinct from the small
  main application coroutine frame.
- The post-repair complete fallback selection is 172/173, 190.91 s
  (`fallback-arena-final-integration-fallback.log`). The persistent kernel
  now completes; a later wavefront/megakernel comparison fails at the first
  event's `light_ng.z` trace value, -0.62323 versus -0.623204. This remaining
  numerical discrepancy is not resolved or hidden by changing tolerance,
  inlining policy or floating-point implementation.
- Strict native-XIR Vulkan: SSS queue transitions, curve setup, triangle
  setup and lamp emission, 4/4 (`native-default-verified-vulkan.log`). All
  three native-XIR/require-SPIR-V/disable-DXC switches are enabled. An
  additional curve run with `LD_DEBUG=libs` loads neither libdxcompiler nor
  libdxil (`native-curve-vulkan-loader.log`).
- After the final allocator repair, focused HIP and strict native-XIR Vulkan
  pass 4/4 each (`fallback-arena-final-integration-{hip,vulkan}.log`). The
  allocator lifetime/reuse and actual LLVM barrier-frame regressions pass
  2/2 (`fallback-arena-reuse.log`), as do existing shared-memory and fallback
  shader-cache/boolean/minimal-codegen tests. No HIP/Vulkan code or inlining policy changes.
- `native-curve-focused-hip.log`: curve setup, original triangle surface setup
  and full curve-path renderer, 3/3.
- `native-shadow-split-focused-hip.log`: stored/local traversal, native curve
  and triangle setup, and original-Cycles native shadow state, 5/5.
- Before the curve extension, the default/identity/lamp consumer checkpoint
  passes six focused fallback tests (`native-consumers-fallback.log`) and
  strict native-XIR Vulkan lamp emission (`native-lamp-strict-vulkan.log`).
- Fresh uncached staged HIP canaries below use the default SVM path (opt-in
  unset), direct-light queue, absolute samples [0, 256), and fast math. All
  46 output channels are finite. These integration times exclude cold JIT
  and are not a paired performance benchmark.

| Scene | Resolution / spp | Render time | Main subroutines / fields / bytes | Combined / DiffInd relative RMSE |
| --- | --- | --- | --- | --- |
| Lone Monk | 1440x1080 / 256 | 13.5925 s | 4 / 55 / 220 | 0.01239594 / 0.12881646 |
| Monster (SSS) | 1080x1080 / 256 | 15.1410 s | 6 / 71 / 284 | 0.00547924 / 0.02552799 |

Logs and EXRs: `monk-final-staged-hip.*`, `monster-final-staged-hip.*`.
Reports: `monk-final-compare.json`, `monster-final-compare.json`.
The original Cycles HIP references are respectively
`/var/tmp/psycles-shadow-queue-cycles-aBpy7U/cycles.exr` and
`/var/tmp/psycles-multiscene-52-Vw3BkE/monster/cycles1080/cycles.exr`.
The comparison tool validates both Blender build identities, without an
unverified-identity override. Scene bundles are
`/var/tmp/psycles-fullres-hip-20260830/exports/lone-monk-current` and
`/var/tmp/psycles-native-volume-svm-06XnDX/monster-export`.

The repeated complete host selection passes 148/149
(`native-default-verified-host.log`). The stale probe-runner/oracle registry
(Map Range and newer oracle-only probes) is corrected without changing
expected shading results. The remaining source-size failure reports four
pre-existing files above its 2,000-line limit. The policy is not weakened
or described as passing.

Curve oracle SHA-256:
`30746b8e5cb6cd955f03f32c1ac3dde2f3f2a09bde0f12063004e8e9d2fb0426`.

## Remaining deletion boundary

The 2026-09-08 [native background follow-up](../../2026-09-08/native-background/README.md)
removes the background SurfaceProgram consumer and ports importance baking.
The loader still builds transitional material resources for the
stacked-volume renderer and displacement prepass. Their consumers must
move to the original domain-specific SVM state/control flow before deleting
the resources. Old surface callables, graph/value-program compiler/runtime
and corresponding tests are still pending removal. No completed legacy-path
removal, full SVM coverage or renderer-speed parity is claimed here.
