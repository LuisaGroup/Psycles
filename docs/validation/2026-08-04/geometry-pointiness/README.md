# Geometry Pointiness validation

## Outcome

Psycles now imports Blender's original Geometry Pointiness output and matches
the current official Cycles CPU oracle on fallback, HIP, and Vulkan. Blender
does not evaluate or bake the material graph: Psycles retains the raw closure
graph and performs the same mesh-synchronization construction as Cycles before
the generated Luisa shader interpolates the resulting standard attribute.

The implementation was selected from the official Barbershop Interior asset
at `assets/official-blender-scenes/barbershop-interior/barbershop_interior.blend`.
This is the unmodified 287,574,804-byte file supplied for the project, with
SHA-256 `95972b56180462cac47ec82f3a755bd9111ec18ca37a6196a319c013db994130`.
Its 165 unsupported Pointiness diagnostics are now zero.

The source oracle was current Blender/Cycles main `4671e6977336`. The golden
EXRs were rendered by the locally built Blender 5.3 Alpha `b82c3f0da6c1`; its
Pointiness mesh-sync implementation is unchanged from that current source.
No Psycles CPU renderer or reference model exists or was introduced. Host-side
mesh synchronization is part of the Luisa scene implementation, not a second
renderer.

## Formal implementation contract

The implementation follows the four-stage Cycles construction over an
evaluated mesh, rather than approximating curvature case by case:

1. Sort vertices by coordinate sum, preserve Cycles' reverse-index ordering
   for exactly coincident coordinates, find duplicate chains with the same
   `3 * FLT_EPSILON` scan bound and squared-distance predicate, and compress
   every chain to its canonical original vertex.
2. Sum the evaluated Blender point normals over each duplicate equivalence
   class, normalize once, and copy the welded normal to every representative.
3. Canonicalize and deduplicate edges after welding, accumulate the normalized
   incident edge directions, and evaluate Cycles' one-ring angle divided by
   pi.
4. Add every unique welded neighbor once, divide by degree plus one to obtain
   Cycles' two-ring approximation, and copy canonical values back to duplicate
   vertices.

The exporter detects a linked Geometry Pointiness output structurally,
including nested node groups. Only meshes assigned such a material retain
evaluated point normals and Blender's original edges, captured before exporter
tessellation. The shader graph itself is unchanged. Scene import validates
the binary cardinalities and indices; a geometry that needs Pointiness but
lacks this source fails with a named `Geometry.Pointiness` diagnostic instead
of silently shading as zero.

The generated Luisa AST uses the reserved `geom:pointiness` standard attribute
ID, so a user attribute named `pointiness` cannot alias it. Values are uploaded
on the point domain and use the existing typed triangle interpolation service.
Volume shading has no surface primitive (`PRIM_NONE`) and therefore observes
the same missing-attribute zero. No weakly typed runtime property edge or
material-side precomputation was added.

## Canonical probe

`geometry_pointiness` is a smooth split height field containing a convex bump,
a concave depression, a high-frequency ripple, and outer boundaries. Two
disconnected patches meet at an exactly coincident center seam, deliberately
duplicating both vertices and geometric edges. This exercises the quotient,
normal welding, post-weld edge deduplication, one-ring angle, and neighbor blur
in one raw Blender graph.

All runs used 64x64, 16 spp, box filtering, Tabulated Sobol, and the same
latest Cycles CPU EXR. Combined and Emit are hard-gated to luminance ratio
`0.99999..1.00001` and relative RMSE at most `1e-5`; all other passes retain
the runner's exact/structural gates. Every result has zero invalid pixels.

| Backend | Combined RMSE | Relative RMSE | Luminance ratio | Max error |
|---|---:|---:|---:|---:|
| fallback | 2.825225e-8 | 5.648139e-8 | 1.000000000 | 1.788139e-7 |
| HIP | 2.921445e-8 | 5.840500e-8 | 1.000000000 | 1.788139e-7 |
| Vulkan | 2.921445e-8 | 5.840500e-8 | 1.000000000 | 1.788139e-7 |

HIP's cold shader JIT took 5.44 s, of which the log attributed about 0.79 s
to LLVM code generation and 4.44 s to linking; the actual 16-spp render took
0.0054 s. Vulkan's cold JIT took 3.08 s, producing 181,281 SPIR-V words before
and 153,151 after optimization; rendering took 0.0142 s. Fallback's cold JIT
took 1.33 s and rendering took 0.0114 s. These are compile diagnostics for the
probe, not full-scene throughput claims.

## Visual inspection

I opened all three 1552x582 triptychs at original resolution. Cycles and
Psycles show the same convex and concave response, ripple, welded center seam,
and boundary blur on every backend. The difference panels contain only
float32 rounding residuals and are automatically amplified by approximately
7.55 million times; there is no shifted seam, edge discontinuity, curvature
sign error, missing region, or backend-specific structure.

- [fallback triptych](pointiness-fallback-triptych.png)
- [HIP triptych](pointiness-hip-triptych.png)
- [Vulkan triptych](pointiness-vk-triptych.png)

Machine-readable reports are stored beside this document as
`pointiness-{fallback,hip,vk}-report.json`.

## Barbershop full-scene audit

The final export contains 1,649 geometries, 2,555 instances, 547 original
materials, and 190 images. It completed in 4:15.97 with no swap. All 547 raw
materials then compiled successfully. The diagnostic count fell from 217 to
52: all 165 Pointiness warnings disappeared, without changing any material
graph.

Pointiness source data is demand-driven. It is present on 885 geometries and
contains 9,867,338 point normals plus 19,748,970 original edges. The exact
binary cost is 118,408,056 bytes of normals plus 157,991,760 bytes of edges,
or 276,399,816 bytes total. `geometry.bin` grew from 4,879,523,152 to
5,155,922,968 bytes by exactly that amount; plain meshes incur no Pointiness
payload. The export cost 9.13 s more than the preceding Wave checkpoint
(+3.7%). A full C++ import and formal construction over all 885 sources passed
in 4.11 s with no invalid cardinality or edge index.

The 52 remaining diagnostics are now sharply localized: 22 Voronoi nodes, six
implicit socket conversions, four Hair Info nodes, four Attribute nodes, four
missing `guilder_ornament.png` references, two Refraction closures, two true
displacement requests, two missing `generic_scratches.png` references, four
missing agent-face images, and one each of Subsurface Scattering and Magic
Texture. These remain explicit blockers; this checkpoint does not claim full
Barbershop image parity.

The official file also emits historical migration warnings under Blender 5.3:
its old Filmic display/view settings and `sRGB EOTF` names are unavailable,
Smoke is deprecated, and embedded `generate_customprops.py` is skipped because
scripts are disabled. Those messages predate this implementation and are kept
separate from Psycles' 52 material diagnostics.

## Regression gates

The regression set includes an analytic split-cube quotient test, isolated and
invalid-topology cases, a Blender quad export proving that the original four
edges are retained without a tessellation diagonal, C++ bundle import and
lowering, and backend scene tests that reject missing sources and accept valid
ones on fallback, HIP, and Vulkan. The Release build used all 32 build jobs.
The complete suite passed `139/139` tests with 32 parallel lanes: 32.63 s on
the first backend-cache population and 4.22 s on the final warm-cache run. The
source-size gate passed for 369 first-party files (119,858 lines, 2,000-line
per-file limit).

Geometry remains `device_partial` in the versioned Cycles node inventory
because its independent Tangent and Parametric outputs are not complete.
Pointiness itself is nevertheless covered by the strict canonical probe and
the tests above; the inventory remains 43 `cycles_verified` and 45 complete
nodes rather than overstating the whole Geometry node.

## Commands

```text
cmake --build build --parallel 32
ctest --test-dir build --output-on-failure --parallel 32
python tools/run_cycles_shader_probes.py geometry_pointiness --blender /path/to/blender --psycles-render build/bin/psycles_render_blender_scene --output-dir /tmp/pointiness --backend fallback|hip|vk --cycles-device CPU --width 64 --height 64 --samples 16
build/bin/psycles_inspect_blender_material /tmp/barbershop-export cash_register_wood --require-pointiness-source
```
