# Light-tree selection and forward-MIS checkpoint

This checkpoint enables Blender's `use_light_tree` setting in the Luisa
integrator. The implementation was checked against official Cycles source
commit `f0562f12ec27ff680bd770dd34f576790a71071e` (Blender main, 2026-08-06),
principally `intern/cycles/scene/light_tree.*` and
`intern/cycles/kernel/light/tree.h`.

It is a production estimator checkpoint, not a claim that the finite-sample
proposal is already bit-identical to Cycles.

## Implemented contract

- Emissive triangles, analytic lights, and the optional environment share one
  stable dense emitter identity. Spatial construction may reorder emitters,
  but NEE and forward-hit MIS address the same identity through explicit
  reverse mappings.
- Local emitters use a 12-bucket spatial/orientation hierarchy; distant lights
  and the environment occupy the dedicated distant branch. The compact GPU
  ABI uses aligned `float4`/`uint4` lanes and is shared by fallback, HIP, and
  Vulkan.
- Traversal uses Cycles' equal mixture of maximum- and minimum-importance
  proposals, including the zero-minimum uniform reservoir and random-number
  remapping. Surface and volume forms are separate Luisa callables, so the
  host-stage specialization emits no runtime surface/volume mode branch.
- Reverse probability follows the selected leaf-to-root path and reproduces
  the same leaf mixture. Surface, analytic-light, environment, and volume
  forward hits use this probability in their existing power-heuristic MIS.
- Emissive triangle forward hits use a sorted `(Cycles object, Cycles
  primitive)` lookup followed by the stable emitter mapping. Selection never
  scans the emitter population; the lookup is logarithmic.
- Volume forward MIS carries Cycles' full `ray.previous_dt` independently of
  the displacement from the segment start to the sampled collision.
- Tree bounds are built after displacement from the final intersection
  vertices, with the source instance transform and effective instance material
  override. The material query is only Cycles-style emission metadata for
  proposal construction. The original closure graph is still evaluated on the
  device for radiance; no Blender/Cycles pre-bake or surrogate material is
  introduced.
- The legacy flat CDF remains available when the Blender setting is disabled.
  Import now accepts and preserves `use_light_tree=true` instead of silently
  dropping or rejecting it.

## Formal invariants and regressions

`psycles_light_tree_tests` checks bounding and orientation-cone union, energy
conservation, the local/distant root split, leaf capacity, and both inverse
emitter mappings. Sparse or duplicate stable identities are rejected.

`psycles_luisa_light_tree_tests` additionally checks:

- host construction from final displaced support and an instance material
  override;
- surface and volume sampling normalization over 8192 stratified selections;
- sampled probability against the independently reconstructed reverse PDF;
- behind-surface rejection versus transmission admission;
- surface and volume forward-PDF reconstruction, including `previous_dt`;
- the sorted triangle-to-emitter lookup.

`psycles_luisa_area_light_forward_tests` runs a scene-compiled, fused path
kernel with the tree enabled. Its one-emitter rectangle must retain the same
image and direct-light trace as the established flat-distribution Cycles
oracle fixture. `psycles_luisa_emissive_triangle_forward_pdf_tests` locks the
separation between selection probability and conditional triangle solid-angle
density.

The Blender importer regression now requires a valid import and a preserved
true setting. This replaces the obsolete negative capability test.

All commands below completed with exit code zero on the RX 9070 XT:

```bash
cmake --build build --target \
  psycles_blender_import_tests \
  psycles_light_tree_tests \
  psycles_luisa_light_tree_tests \
  psycles_luisa_area_light_forward_tests \
  psycles_luisa_emissive_triangle_forward_pdf_tests \
  --parallel 32

build/psycles_blender_import_tests
build/psycles_light_tree_tests

for backend in fallback hip vk; do
  build/bin/psycles_luisa_light_tree_tests "$backend"
  build/bin/psycles_luisa_area_light_forward_tests "$backend"
  build/bin/psycles_luisa_emissive_triangle_forward_pdf_tests "$backend"
done
```

## HIP backend defect found by this checkpoint

The first complete volume-forward test differed only on HIP: the reconstructed
probability was `0.291219` instead of `0.277084`, while fallback and Vulkan
matched. Reducing the test did not expose a Callable ABI error. Dumping the
final HIP LLVM modules showed that a dynamically normalized `float3` dot
product still used `llvm.vector.reduce.fadd.v3f32`; in a sufficiently large,
divergent kernel the AMDGPU target miscompiled that reduction.

Luisa `next` commit `f2718bab2` lowers fixed-vector floating-point dot products
to explicit ordered component additions and increments the HIP shader-cache
codegen revision. Its regression sends normalized vectors through a large
out-of-line Callable, checks 16 runtime values, and asserts that final LLVM IR
contains no target-unstable reduction intrinsic. After rebasing onto the then
current `next`, the callable suite passed 39 assertions in three tests and the
shader-cache suite passed 35 assertions. The Psycles probability then matched
on all three backends.

## Compilation observation

The full forward fixture also isolates the current cold-JIT gap. HIP LLVM
code generation took about 2.7--5.2 seconds per large variant and bitcode
linking about 0.7--2.0 seconds. Vulkan's expensive stage was the compute
SPIR-V optimization/generation of roughly 404k--479k final words, taking
about 10--18 seconds per large variant. The small standalone forward-PDF
kernel compiled immediately on both backends. This points to large fused
shader optimization, rather than tree construction or runtime traversal, as
the next Vulkan compile-time target.

## Remaining Cycles alignment work

The current hierarchy is one flattened world tree. Current Cycles additionally
uses mesh-emitter representatives, per-mesh subtrees, instance/reference
nodes, object-local queries, bit trails, receiver light-link roots, and
specialized per-light importance parameters. Psycles' precomputed generic
bounds/cones make sampling and reverse PDFs reciprocal and therefore preserve
the estimator, but they do not yet guarantee the same proposal probability or
tree topology as Cycles for a complex scene. Motion-time emitter bounds,
light linking, and environment-map importance sampling are also outstanding.

Consequently, the old Monster under the Bed and Lone Monk renders are not
promoted by this focused test. Current-HEAD 480p-or-larger Cycles CPU/HIP and
Psycles fallback/HIP/Vulkan reruns, numerical reports, and inspected triptychs
remain required before either scene can be called correct.
