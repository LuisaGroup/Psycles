# Blender Nishita sky model

These files are unmodified copies of Blender's bundled Nishita sky model at
Cycles/Blender commit `9e2066aef7ef7e20c142ad7bd3303138a4304c93`:

- `intern/sky/include/sky_nishita.h`
- `intern/sky/source/sky_math.h`
- `intern/sky/source/sky_single_scattering.cpp`
- `intern/sky/source/sky_multiple_scattering.cpp`

Psycles uses them for the same host-side 512 x 256 generated image and solar
disc constants consumed by Cycles 5.2.1 `NODE_TEX_SKY`. The generated image is
uploaded as a native Luisa texture; shader evaluation remains in the Luisa SVM
interpreter on every backend.

The per-file SPDX headers are authoritative. The single-scattering source is
Apache-2.0, the multiple-scattering source is MIT, and Blender's shared sky
declarations/math support are GPL-2.0-or-later.
