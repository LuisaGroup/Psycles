# Voronoi Texture validation

## Outcome

Psycles now retains Blender's original Voronoi Texture nodes and evaluates
them in generated Luisa shaders on fallback, HIP, and Vulkan. Blender/Cycles
does not bake coordinates, colors, texture results, or closures for Psycles.
The implementation covers 1D--4D, F1, F2, Smooth F1, Distance to Edge,
N-Sphere Radius, all four distance metrics, Normalize, all valid outputs, and
all nine dynamic sockets.

This work was selected from the official
[Barbershop Interior scene](https://svn.blender.org/svnroot/bf-blender/trunk/lib/benchmarks/cycles/barbershop_interior/barbershop_interior.blend).
The repository copy is the unmodified 287,574,804-byte file with SHA-256
`95972b56180462cac47ec82f3a755bd9111ec18ca37a6196a319c013db994130`.
The server reports the same byte length. The asset inventory contains 28
Voronoi occurrences; its reachable material compilation formerly emitted 22
unsupported-Voronoi diagnostics, all of which are now gone.

The source oracle was current Blender/Cycles main
`72e50464a3cf00ee93954ec74ee8d9dbe9f42ab8`. Golden EXRs were rendered by
official Cycles CPU in Blender 5.2.0 LTS `fbe6228777e7`; a source diff confirms
that the relevant Voronoi kernel and node implementation are unchanged between
that build and current main. No Psycles CPU renderer or CPU reference model
exists or was introduced.

## Formal implementation contract

The implementation is one dimension-parameterized construction rather than a
set of scene-specific patches:

1. The Blender adapter retains the raw Vector, W, Scale, Detail, Roughness,
   Lacunarity, Smoothness, Exponent, and Randomness expressions together with
   dimension, feature, metric, Normalize, and requested-output properties.
   An unlinked Vector retains Cycles' Generated-coordinate behavior.
2. The typed `SurfaceProgram` stores every dynamic dependency explicitly and
   encodes the finite static configuration. Distance, Color, Position, W, and
   Radius are distinct typed operations. Graph lowering emits only an output
   that is actually requested, so unused outputs do not duplicate the AST.
3. While Luisa traces the shader AST, the host-side Voronoi component expands
   dimension, feature, metric, Normalize, and output into a cached typed
   `Callable`. No weakly typed `float4` property array or runtime property
   switch is carried through each shading point.
4. A single dimension-aware lattice construction supplies Cycles' exact hash
   family, neighborhood traversal, distance ordering, feature-point selection,
   and Color/Position recovery for 1D through 4D. Euclidean, Manhattan,
   Chebychev, and Minkowski are projections of the same metric contract.
5. F1, F2, and Smooth F1 share the same octave abstraction. The fractal layer
   follows Cycles' fractional Detail recurrence, Roughness/Lacunarity frequency
   and amplitude evolution, position interpolation, maximum-distance bound,
   Normalize behavior, and kernel-side socket clamps.
6. Distance to Edge and N-Sphere Radius use their own formally bounded
   neighborhood reductions, including the 1D midpoint construction and the
   multidimensional perpendicular/nearest-neighbor definitions.

The public component is split across a real header, implementation, and graph
node factory. The largest new C++ implementation is 868 lines; no `.inl`
splitting or material-specific specialization was used.

## Probe coverage and results

`voronoi_texture_distance` contains 32 cells covering F1/F2, 1D--4D, every
metric, zero/signed/large Scale, Randomness endpoints, implicit Generated
coordinates, and every dimension-valid Distance/Color/Position/W output.

`voronoi_texture_fractal` contains 16 cells covering Smooth F1, fractional and
clamped Detail, normalization, signed Lacunarity, zero/signed Scale, linked
dynamic scalar inputs, socket clamp boundaries, position/color/W outputs, and
zero-input recurrence exits.

`voronoi_texture_edges` contains 16 cells covering Distance to Edge and
N-Sphere Radius in every dimension, base and fractal paths, normalization,
signed Scale/Lacunarity, and Randomness endpoints.

Every run used 64x64, 4 spp, box filtering, raw Blender graphs, and the same
official Cycles CPU EXR. Combined and Emit are hard-gated to relative RMSE at
most `1e-5`; distance and fractal are additionally gated to luminance ratio
`0.99999..1.00001`. Every recorded pass has zero invalid pixels.

| Probe | Backend | Combined RMSE | Relative RMSE | Luminance ratio | Max error |
|---|---|---:|---:|---:|---:|
| distance | fallback | 6.407301e-8 | 4.820229e-8 | 1.000000089 | 1.192093e-6 |
| distance | HIP | 5.901227e-8 | 4.439509e-8 | 1.000000000 | 8.344650e-7 |
| distance | Vulkan | 7.250566e-8 | 5.454620e-8 | 1.000000089 | 8.344650e-7 |
| fractal | fallback | 1.207541e-7 | 1.181785e-7 | 1.000000000 | 8.702278e-6 |
| fractal | HIP | 5.446303e-7 | 5.330140e-7 | 1.000000000 | 4.762411e-5 |
| fractal | Vulkan | 5.455904e-7 | 5.339535e-7 | 1.000000071 | 4.762411e-5 |
| edge/radius | fallback | 2.373320e-5 | 3.578588e-6 | n/a | 8.964539e-5 |
| edge/radius | HIP | 2.373320e-5 | 3.578588e-6 | n/a | 8.964539e-5 |
| edge/radius | Vulkan | 2.323091e-5 | 3.502850e-6 | n/a | 8.773804e-5 |

The edge/radius matrix deliberately projects signed Position/W values into
emission along with scalar distances. Its mean is negative, so a positive
luminance-ratio gate is not meaningful; relative RMSE remains the hard gate.

## Oracle-domain regressions

Two probe-construction issues were found and made explicit instead of being
papered over in the renderer:

- Blender's host constant folder omits the Detail/Roughness clamps present in
  the Cycles SVM/HIP kernel path. A fully constant graph can therefore produce
  a different result from the renderer that is supposed to be the oracle.
  Multidimensional fractal cases leave Vector shader-varying through Generated
  coordinates, proving the actual Cycles kernel path; the 1D boundary cases
  remain inside the declared socket ranges.
- Repeated high Detail and Lacunarity can move a lattice coordinate outside
  the signed 32-bit domain before hashing. Float-to-integer conversion there
  is backend-dependent. The regression still exercises Detail clamping at 15
  and high-frequency recurrence, but selects coordinates/Lacunarity whose
  cells remain in the defined integer domain on every oracle/backend.

These constraints are documented in the probe source and preserve broad
algorithmic coverage. They are not Psycles-side special cases.

## Visual inspection

I opened all nine 1552x582 triptychs at original resolution. Cycles and
Psycles have the same per-cell values and boundaries, dimensional selection,
F1/F2/Smooth F1 structure, metric behavior, feature-point color/position,
fractal transitions, edges, and radii on every backend. Difference panels use
automatic amplification between roughly 10,000 and 3.77 million times. They
show only floating-point residuals; there is no shifted boundary, missing
branch, orientation error, discontinuity, or backend-specific artifact.

Distance/F1/F2 matrix:

- [fallback triptych](distance-fallback-triptych.png)
- [HIP triptych](distance-hip-triptych.png)
- [Vulkan triptych](distance-vk-triptych.png)

Fractal/Smooth F1/dynamic-input matrix:

- [fallback triptych](fractal-fallback-triptych.png)
- [HIP triptych](fractal-hip-triptych.png)
- [Vulkan triptych](fractal-vk-triptych.png)

Distance-to-edge/N-sphere-radius matrix:

- [fallback triptych](edges-fallback-triptych.png)
- [HIP triptych](edges-hip-triptych.png)
- [Vulkan triptych](edges-vk-triptych.png)

Machine-readable reports are stored beside this document as
`{distance,fractal,edges}-{fallback,hip,vk}-report.json`.

## Barbershop full-scene audit

The existing raw Barbershop bundle contains 1,649 geometries, 2,555 instances,
547 original materials, and 190 images. All 547 raw closure graphs compile;
there is no Blender/Cycles material bake. The previous checkpoint emitted 52
material diagnostics, including 22 reachable `TEX_VORONOI` failures. With the
same bundle and this implementation, the final inspector emits 30 diagnostics
and zero errors: every Voronoi failure disappeared. The audit took 2.66 s,
peaked at 5.19 GB RSS while reading the 4.9-GB geometry bundle, and did not
swap.

All 28 inventoried nodes are 3D, F1, Euclidean, and non-normalized; 16 request
Color and 12 request Distance. The broader probes above prevent the
implementation from degenerating into a Barbershop-only case.

The remaining 30 diagnostics are independent blockers: two Refraction
closures, one Subsurface Scattering closure, four Hair Info nodes, four
Attribute nodes, one Magic Texture, six implicit conversions, two true
displacement requests, and ten unavailable image references. This checkpoint
does not claim full Barbershop image parity or a scene-level speedup while
those features remain incomplete.

## Compile diagnostics and regression gates

A cache-cleared distance-matrix diagnostic took 8.27 s for HIP. About 2.10 s
was LLVM code generation and 5.66 s was bitcode-to-AMDGPU-object linking, so
the link dominated that cold run. Vulkan took 8.04 s; SPIR-V optimization took
about 3.86 s (290,118 to 247,436 words), with the remaining time in
DXC/SPIR-V generation and driver pipeline creation. On the final warm-cache
rerun, distance JIT/render times were 2.65 s/0.0034 s for HIP and
3.99 s/0.0056 s for Vulkan. These are probe compile diagnostics, not
Barbershop throughput claims.

The Release build used all 32 build jobs and completed in 8.9 s. The complete
suite passed `139/139` tests with 32 parallel lanes in 4.22 s. Structural
regressions assert the closed static configuration encoding, all nine typed
dynamic dependencies, requested-output specialization, and absence of unused
Voronoi operations. The source-size gate passed for 373 first-party files
(121,474 lines, 2,000-line per-file limit). The versioned node inventory now
records 44 `cycles_verified` and 46 complete Cycles nodes.

## Commands

```text
cmake --build build --parallel 32
ctest --test-dir build --output-on-failure --parallel 32
python3 tools/run_cycles_shader_probes.py voronoi_texture_distance voronoi_texture_fractal voronoi_texture_edges --blender /usr/bin/blender --psycles-render build/bin/psycles_render_blender_scene --output-dir /tmp/voronoi --backend fallback|hip|vk --cycles-device CPU --width 64 --height 64 --samples 4
build/bin/psycles_inspect_blender_material /tmp/barbershop-export '*'
```
