# Production volume-majorant provider

This checkpoint connects scene transforms, object-density scaling, and
runtime Light Path extrema evaluation to the ordered overlapping-majorant
traversal. Blender/Cycles main
`b82c3f0da6c1813dabedc563d64e536f4d83e868` is the sole behavioral oracle.
The implementation evaluates each original `GraphSurface` Volume closure
through Luisa DSL; it adds neither a Psycles CPU reference renderer nor a
material pre-bake.

## Cycles contract

The exact reference is
`intern/cycles/kernel/integrator/shade_volume.h`:
`volume_estimate_extrema`, `volume_object_get_extrema`, and the object-space
setup in `volume_octree_setup`.

`SceneVolumeMajorantEntryProvider` records the corresponding policy:

- object entries transform the world ray origin and direction with the
  instance inverse transform, preserving the world-ray distance parameter;
- the World entry performs no nominal TLAS lookup and keeps the world ray
  unchanged;
- ordinary shaders and camera rays use baked leaf extrema multiplied by the
  entry's object-density scale;
- a non-camera Light Path shader evaluates the original raw Volume closure
  along the world ray;
- homogeneous closures use one midpoint sample;
- heterogeneous closures use four samples at
  `t_min + (shade_offset + i) * interval_length / 4`; and
- the heterogeneous maximum becomes
  `max(0.5, 1.5 * sampled_max)`, while the sampled minimum is retained.

Extinction and emission are reduced to their largest RGB channel before the
maximum of the two is taken. This matches Cycles' closure fields rather than
inventing a scalar density surrogate.

The provider is a real host-stage `.h`/`.cpp` component. Its virtual
interface is exercised while Luisa records the enclosing kernel AST, so the
device still executes one fused shader without a device virtual call.
Per-surface heterogeneous and Cycles-compatible Light Path flags are uploaded
in a compact device buffer keyed by the structural `GraphSurface` tag.

## Shader-flag semantics

Current Cycles does not restrict `SD_HAS_LIGHT_PATH_NODE` to the Volume
closure. `scene/shader.cpp` scans every node in the finalized shader and even
contains a TODO to check Volume reachability in the future. Psycles therefore
sets the runtime-majorant flag for a Light Path node connected only to the
Surface closure as well. Using a narrower dependency proof would be a
plausible optimization, but it would not be current Cycles behavior.

Homogeneity remains a separate structural proof over values reachable from
the original Volume closure. Its traversal visits every dependency edge and
reduces the result without making the shader-flag decision depend on input
ordering.

The regression contains an ordinary spatial Volume without Light Path, a
homogeneous Volume `IsCameraRay` dependency, a spatial Volume expression with
both Generated and `RayDepth`, and a homogeneous Volume paired with a
surface-only `RayDepth`. It proves all four combinations, including Cycles'
deliberately broad surface-only flag.

## Exact device regression

The production scene fixture uses a translated, uniformly scaled TLAS
instance. For world origin `(2.5, -2, 6.5)` and direction `(1, 0, 0)`, all
three backends produce object origin `(1, 2, 3)` and direction `(2, 0, 0)`.
The World probe remains unchanged.

The heterogeneous raw graph is:

```text
Texture Coordinate.Generated -> Vector to Scalar
Light Path.RayDepth ----------> Multiply
Vector to Scalar.Value -------> Multiply
Multiply.Value ---------------> Volume Absorption.Density
```

For interval `[0, 2]`, shade offset `0.25`, path depth `1`, and the fixture's
Generated transform, the four exact samples are at
`0.125`, `0.625`, `1.125`, and `1.625`. Their scalar extrema are
`1.54166667` and `2.04166667`; the provider returns
`[1.54166667, 3.0625]` after Cycles' safety expansion. The homogeneous
non-camera Light Path probe returns `[0, 0]`, while its camera probe preserves
the baked `[0.25, 0.75]`.

Run:

```sh
cmake --build build --parallel 32 \
  --target psycles_luisa_volume_majorant_scene_tests
ctest --test-dir build --output-on-failure --parallel 3 \
  -R 'psycles\.luisa_volume_majorant_scene_(fallback|hip|vk)$'
```

The focused regression passes on fallback, HIP, and Vulkan on the local
RX 9070 XT in `0.14 s` total with warm shader caches. A full 32-thread build
completed in `8.17 s`; the final incremental rebuild completed in `0.28 s`.
Serial CTest passed `105/105` tests in `8.96 s`, including the source-size
gate. Machine-readable measurements are recorded in `backend-report.json`.

## Visual status

This checkpoint emits transforms and scalar interval extrema, not pixels.
An EXR or triptych would not exercise the component, so no image is claimed.
The first production heterogeneous collision/phase/direct-light connection
must be followed by official Cycles/Psycles EXR comparison and a documented
visual inspection before the heterogeneous scene gate can be removed.
