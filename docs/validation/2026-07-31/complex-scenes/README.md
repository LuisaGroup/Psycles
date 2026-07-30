# Complex Blender scene ingestion

This checkpoint exercised Psycles' raw Blender scene exporter against the
official Blender 4.1 splash and Classroom demo assets before attempting a
release-quality render. It records ingestion facts and compatibility gates;
it is not a visual-parity claim.

## Reference revisions

- Psycles began this audit at `42b5be8`.
- Blender/Cycles reference checkout:
  `b82c3f0da6c1813dabedc563d64e536f4d83e868`.
- Exporting Blender build: 5.3.0 Alpha,
  `16d7a3a413e7` (2026-07-29).
- Source assets:
  `assets/official-blender-scenes/blender-4.1-splash.blend` and
  `assets/official-blender-scenes/classroom/classroom.blend`.

## Blender 4.1 splash

The complete source scene exported 144 evaluated geometries, 232 instances,
76 materials, 120 images, 26 analytic lights, and 28,089,460 triangles. The
current v1 geometry contract expanded every triangle corner into an independent
vertex:

| Measurement | Value |
|---|---:|
| Exported triangle corners | 84,268,380 |
| `geometry.bin` | 8.3 GiB |
| All referenced geometry sections | 8,841,529,440 bytes |
| Position stream | 1,011,220,560 bytes |
| Generated-coordinate stream | 1,011,220,560 bytes |
| Named UV value/tangent streams | 3,074,860,368 bytes |

This is now the motivating real-scene case for replacing the flattened v1
layout with Cycles-style point, corner, and face attribute domains. The scene
also enables Cycles light-tree sampling and connects seven Displacement roots;
both remain explicit render gates. No volume root is connected.

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
| Triangles / flattened corners | 496,424 / 1,489,272 |
| Export directory | 222 MiB |
| `geometry.bin` | 135 MiB |
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
- The full 32-way build and all 44 CTest cases pass after both repairs.

Visual triptychs are intentionally deferred until the outstanding scene gates
are implemented. Publishing an image with substituted displacement, light-tree
sampling, or volume behavior would not be a Cycles-parity result.
