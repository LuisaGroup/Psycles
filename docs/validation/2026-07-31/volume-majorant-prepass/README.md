# Raw volume majorant prepass

This checkpoint reproduces the current Cycles heterogeneous-volume density
bake as a Luisa shader pass. Blender/Cycles main
`b82c3f0da6c1813dabedc563d64e536f4d83e868` is the sole oracle. There is no
Psycles CPU reference renderer and no material or closure pre-bake.

## Cycles contract

The reference path is
`intern/cycles/kernel/bake/bake.h::kernel_volume_density_evaluate`,
`kernel/sample/sobol_burley.h`, and `kernel/tables.h`. For every
heterogeneous cell it defines:

- a fixed `128^3` grid with x-fast flattened addressing;
- sixteen samples in a voxel padded by twenty percent on every side;
- Sobol index `cell_index * 16 + sample`, dimension set zero, seed zero, and
  mask `0xffffffff`;
- object-space sample construction followed by the object-to-world point
  transform;
- zero ray direction, time `0.5`, camera visibility, no path flags, and zero
  path depths/events/ray length;
- evaluation of the original volume graph with emission enabled;
- scalar reduction
  `max(max_component(extinction), max_component(emission))`; and
- minimum/maximum reduction followed by division by the entry-invariant
  `object_volume_density` scale.

`VolumeMajorantPrepass` records exactly this computation through
`SurfaceDispatch` and `VolumeStackEntryPointProvider`. Those are host-stage
polymorphic AST builders: the resulting device kernel directly evaluates the
raw `GraphSurface` closure at every sample. The extrema are acceleration
metadata only. Runtime heterogeneous transport will evaluate the same graph
again at every candidate collision.

The finite sixteen-sample extrema are Cycles' estimate, not a mathematical
proof of a global bound. The runtime `majorant_exceeded` contract therefore
remains explicit.

## Exact sampler regression

The 3D Sobol-Burley Luisa implementation carries only the first three Cycles
direction dimensions needed by this pass, with the dimension-zero Van der
Corput path and the same Owen scrambling. A temporary oracle executable was
compiled directly against the official Cycles headers and then removed; it
was not a Psycles reference implementation.

Nine indices exercise low, middle, and highest input bits. Each x/y/z result
is compared by its IEEE-754 bits, not by tolerance:

| index | x | y | z |
| ---: | ---: | ---: | ---: |
| `0` | `3f505454` | `3ebbc2e7` | `3f7de3ea` |
| `1` | `3d7e88f3` | `3f65bb61` | `3e8f2b81` |
| `2` | `3f202b30` | `3f12fa29` | `3e042c5b` |
| `15` | `3f31ab47` | `3ecbc20c` | `3f0aef01` |
| `16` | `3f3dff98` | `3f521c79` | `3ecde3a0` |
| `197520` | `3f32f9da` | `3e98f0a4` | `3ce4c243` |
| `197535` | `3eb0a99e` | `3df2adb5` | `3db0070b` |
| `0xfffffff0` | `3f201fb0` | `3f5fe037` | `3f4478f9` |
| `0xffffffff` | `3edd6ee4` | `3ded72f7` | `3f0d578f` |

## Raw-graph regression

The fixture constructs an ordinary graph rather than a synthetic coefficient
callback:

```text
Texture Coordinate.Generated -> Volume Coefficients.EmissionCoefficients
```

It applies a translated object transform and an object-density scale of two.
The point provider emits a large sentinel value if any camera visibility,
zero-direction, path-state, or time field differs from the Cycles bake state.
After the density scale is divided back out, the extrema are checked for
three distant cells:

| cell | minimum | maximum |
| ---: | ---: | ---: |
| `0` | `0.0031634965` | `0.00928486325` |
| `12345` | `0.749039471` | `0.759348571` |
| `2097151` | `0.995441556` | `1.00154698` |

The production scene point provider also has an explicit regression for the
zero-direction bake ray. Its transposed object normal transform now uses
Cycles' zero-preserving `safe_normalize()` definition rather than producing
NaNs.

Run:

```sh
cmake --build build \
  --target psycles_luisa_volume_majorant_prepass_tests \
  --parallel 32
ctest --test-dir build --output-on-failure \
  -R 'psycles\.luisa_volume_majorant_prepass_(fallback|hip|vk)' \
  --parallel 1
```

All focused prepass and volume-point tests pass on fallback, HIP, and Vulkan.

## Visual status and remaining connection

This pass produces internal scalar acceleration metadata, not pixels, so an
EXR triptych or visual judgement would be misleading at this checkpoint.
Cycles/Psycles triptychs remain mandatory when the heterogeneous estimator is
connected to the production render path.

Scene-side dispatch, hierarchy flattening, and retained root/node/range buffers
are now complete and recorded in
[`../volume-majorant-scene-resources`](../volume-majorant-scene-resources/README.md).
Overlapping-root interval reduction and full collision/phase/direct-light
transport follow. Cycles' per-cell LCG state for stochastic texture sampling
also remains an explicit prerequisite before arbitrary stochastic volume
textures can be claimed.
