# Complex Blender scene ingestion

This checkpoint exercised Psycles' raw Blender scene exporter against the
official Blender 4.1 splash and Classroom demo assets before attempting a
release-quality render. It records ingestion facts and compatibility gates;
it is not a visual-parity claim.

## Reference revisions

- Psycles began this audit at `42b5be8`; compact-domain geometry completed at
  `151bb96`.
- Blender/Cycles reference checkout:
  `b82c3f0da6c1813dabedc563d64e536f4d83e868`.
- Exporting Blender build: 5.3.0 Alpha,
  `16d7a3a413e7` (2026-07-29).
- Source assets:
  `assets/official-blender-scenes/blender-4.1-splash.blend` and
  `assets/official-blender-scenes/classroom/classroom.blend`.

## Blender 4.1 splash

The complete source scene exports 144 evaluated geometries, 232 instances,
76 materials, 120 images, 26 analytic lights, and 28,089,460 triangles. The
initial v1 contract expanded every triangle corner into an independent vertex;
v2 retains 14,135,152 points and 84,268,380 true corners:

| Measurement | v1 flattened | v2 domain-aware |
|---|---:|---:|
| `geometry.bin` | 8,841,529,456 B | 7,046,437,660 B |
| Complete export bundle | 9,018,751,868 B | 7,223,688,451 B |
| Position stream | 1,011,220,560 B | 169,621,824 B |
| Generated-coordinate stream | 1,011,220,560 B | 169,621,824 B |
| Named UV value/tangent streams | 3,074,860,368 B | 3,074,860,368 B |

Geometry storage decreased by 20.30% and the complete bundle by 19.90%.
The v2 export took 353.9 s and peaked at 25,051,412 KiB RSS. That peak remains
an engineering problem: the required corner-domain UV/tangent arrays need a
streaming exporter, not vertex averaging or discarded attributes.

The scene enables Cycles light-tree sampling and connects seven Displacement
roots; both remain explicit render gates. No volume root is connected.

## Classroom

The first export failed because images owned by linked asset libraries were
resolved relative to the main `classroom.blend`. Blender and Cycles instead
resolve `//` paths relative to the owning `Library`. After using
`bpy.path.abspath(image.filepath, library=image.library)`, export reached a
second real source type: the 2048x2048 `checker` image is `GENERATED` and has
no external file. Generated pixels are now copied into a temporary Blender
image datablock and encoded without mutating the source scene.

The repaired export completed with:

| Measurement | Value |
|---|---:|
| Geometries / instances | 252 / 894 |
| Materials / images / lights | 85 / 51 / 10 |
| Points / triangle corners | 256,269 / 1,489,272 |
| Triangles | 496,424 |
| Export bundle, v1 / v2 | 232,131,023 B / 191,619,255 B |
| `geometry.bin`, v1 / v2 | 140,996,704 B / 100,440,364 B |
| Source render dimensions | 1920x1080 |

Classroom has no connected Volume root in its active `_mainScene`; its legacy
`_volumeLight` scene is a separate Blender Internal setup. Seven materials
connect Displacement. The active Cycles scene disables adaptive sampling,
denoising, and the light tree, so after displacement handling it is a useful
1080p path-tracing benchmark.

## Regression coverage

- `psycles.blender_export_linked_images` creates a nested linked `.blend`
  whose texture path is only valid relative to that library.
- `psycles.blender_export_generated_images` exports a generated RGBA texture
  and checks its PNG payload, digest, dimensions, and decoded pixel values.
- `psycles.blender_export_attribute_domains` checks a shared-index quad with
  point and corner attributes, binary section cardinalities, and C++ v2
  import.
- The full path area-light oracle reads POINT, CORNER, and FACE attributes and
  passes on fallback, HIP, and Vulkan.
- The full 32-way build and all 47 CTest cases pass.

The same contract was rendered and visually inspected on Lone Monk; its
same-binary v1/v2 A/B, official Cycles comparison, backend timings, and
triptychs are recorded in
[the compact-geometry report](../compact-geometry/README.md). Splash and
Classroom triptychs remain deferred until their displacement/light-tree gates
are implemented. Publishing images with substituted behavior would not be a
Cycles-parity result.
