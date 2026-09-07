# Native Cycles SVM volume consumers

This checkpoint replaces production stacked-volume, collision, shadow and
majorant material evaluation with Cycles 5.2.1 SVM. It does not complete the
renderer-wide deletion of SurfaceProgram: the scene loader and displacement
prepass still contain legacy dependencies. No Luisa, inlining, floating-point
emulation or fast-math policy changes are part of this checkpoint.

Authority: `/home/mike/Projects/blender-cycles-trace-5.2`, revision
`cb168525138fecc792cc393f94afc39582b0103c`. Implementation base: Psycles
`a9d4015d`; Luisa remains the published `fb315d27d`.

## State and control flow

| Consumer | Cycles authority | Preserved behavior |
| --- | --- | --- |
| Ray setup | `kernel/geom/shader_data.h` | P=ray.P+D*tmin; ray_P=ray.P; N=Ng=wi=-D; volume primitive type; zero compact/parametric differentials and ray length |
| Ordered stack | `kernel/integrator/volume_shader.h` | One allocator for the entire stack; native shader/object flags; shadow visibility before transforms and SVM; merge after every entry beyond entry zero, including hidden entries |
| Main coefficients | `kernel/integrator/shade_volume.h` | Extinction/emission from ShaderData flags; scattering only from successfully allocated closures; a null phase destination does not disable allocation |
| Phase copy | `kernel/integrator/volume_shader.h` | Copy the first eight allocated/merged phase records without reclamping weights or recomputing sample weights; native ClosureType IDs |
| Runtime extrema | `kernel/integrator/shade_volume.h` | Native light-path and homogeneity flags; one/four samples; zero E/T before each entry; no allocations; original heterogeneous floor and expansion |
| Density bake | `kernel/bake/bake.h` | One/sixteen samples; padded Sobol-Burley positions; time 0.5; coefficient-only state; volume feature mask without LIGHT_PATH; object density divided out |

Equal-phase merging is family-specific and stable. It reduces the live count
without refunding allocation budget. The typed volume-only move does not read
surface normals or undefined closure tails. Absorption does not allocate a
scattering closure. Emission/terminate/shadow states retain Cycles' zero
allocation budget rather than being inferred from which output a caller uses.

One native ShaderData persists across entries and candidate positions. Texture
LCG initialization uses the original main-volume `0x15b4f88d` and shadow-volume
`0xd9111870` seeds. Density bake uses the cell-index seed and preserves the
original object-space ray_P while transforming sampled P when required.
Removing the LIGHT_PATH feature for baking does not erase the Light Path
opcode or its visibility-derived outputs.

The scene's native compiler metadata supplies the opcode-case set, stack
high-water mark and global closure budget. No profiling or preliminary render
selects local-array sizes. Fast math stays enabled; oracle comparisons permit
small native arithmetic differences.

## Ownership and deletions

Majorant planning reads the native KernelShader metadata. Baking happens after
the session's final RenderKernelParameters exist, so volume Camera/Window
nodes receive actual camera transforms and projection. Each session owns its
own immutable majorant buffers; the compiled scene retains only the host plan.
This is Cycles' acceleration-data bake, not a material/radiance substitute.

Production volume consumers no longer construct BufferShaderServices,
SurfacePoint or GraphSurface, or derive inverse transforms from the TLAS.
`path_kernel_volume_point.cpp/.h` and the old per-surface volume-flag GPU table
have been deleted. The former point test now exercises the native consumer
against original GPU results; it does not retain the deleted surrogate.
The generic homogeneous/collision transport accepts a host-stage shader
provider and records the same transport equations.

## Independent GPU regressions

Typed fixture headers contain inputs only. Expected results come from the
original Cycles HIP code, never from a Psycles or CPU reference renderer.

| Oracle | Coverage | SHA-256 of checked-in stdout |
| --- | --- | --- |
| `cycles_svm_volume_stack.txt` | 300 capacity/stack/path combinations, setup state, raw closures and copied phases | `e76a983aea3daf183c0482ab7a6c0a3729391900b745112fca049a923f483263` |
| `cycles_svm_volume_density.txt` | 12 native bake combinations: one/sixteen samples, P/Object/Generated, Light Path feature mask, applied/unapplied transforms | `04cac24fa02089600c6463377fc7440b84a0509e204ebc719708122fee30b054` |

The stack output is checked both through the core evaluator and through the
actual production provider after a coefficient-only query. The latter checks
that full-stack evaluation restores its allocation budget. Existing prepass,
majorant scene/hierarchy and heterogeneous transport tests now use native SVM.
Older transport tests retain their original float expectations; their phase
identity assertions now use native enum values instead of private HG=0 IDs.

Reproduce each oracle from the worktree root:

```sh
/opt/rocm/bin/hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC \
  -std=c++20 -O3 -ffast-math \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles -I include \
  tools/cycles_svm_volume_stack_oracle.hip -o /var/tmp/volume-stack-oracle
/var/tmp/volume-stack-oracle

/opt/rocm/bin/hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC \
  -std=c++20 -O3 -ffast-math \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles -I include \
  tools/cycles_svm_volume_density_oracle.hip -o /var/tmp/volume-density-oracle
/var/tmp/volume-density-oracle
```

Evidence directory: `/var/tmp/psycles-native-volume-svm-06XnDX`.
Focused native/transport validation passed 8/8 on HIP and 8/8 on fallback.
Final strict native-XIR Vulkan passed 6/6, including the complete volume path
render test, in 139.96 seconds with
`LUISA_VULKAN_USE_XIR=1`, `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1` and
`LUISA_VULKAN_DISABLE_DXC=1`. Full 32-thread builds pass.
An additional uncached density test with `LD_DEBUG=libs` compiled native
SPIR-V (13,372 words) and loaded neither DXC nor DXIL
(`native-volume-vulkan-loader.log`).

Final full HIP suite: **174/174**, 405.93 seconds
(`native-volume-validated-all-hip.log`). Full fallback: **175/176**, 307.90
seconds (`native-volume-validated-all-fallback.log`). Its sole failure is the
previously observed sample-dispatch film trace at slot 30, component 2:
`light_ng.z` expected `-0.62323 (0xbf1f8bfd)`, observed
`-0.623204 (0xbf1f8a50)`. No tolerance or arithmetic patch was added.

Host checks: **148/149**, 14.73 seconds. The existing source-size guard still
reports four unchanged oversized files (`cycles_svm_nodes.cpp`,
`test_cycles_svm_compiler.cpp`, `test_luisa_compact_surface_preparation.cpp`,
`test_luisa_cycles_svm.cpp`). The limit was not relaxed. Both external GPU
oracle outputs were regenerated and compare byte-for-byte with the checked-in
data. No new large-scene performance measurement is claimed at this checkpoint.

## Remaining scene blocker

At this checkpoint the Classroom 1920x1080/256 canary stopped before JIT: its Object Index
outputs are implemented by native SVM, but the loader still first compiles
every material through SurfaceProgram, which rejects those outputs. This is
not a missing native Object Info handler and does not justify extending the
legacy executor. Removing that remaining compilation/resource dependency is
the next structural task. No new Classroom rendering/performance result is
claimed from this failed attempt.

Resolved by the subsequent [native scene admission checkpoint](../native-scene-admission/README.md),
which records the successful 1920x1080/256 run and the remaining indirect residual.
