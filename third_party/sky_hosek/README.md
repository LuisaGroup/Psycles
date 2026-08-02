# Hosek-Wilkie sky model

These files are unmodified copies of Blender's bundled Hosek-Wilkie XYZ sky
model at Blender commit `c2acd5638ea00c2f5843d34e2049eae60e4cdae3`:

- `intern/sky/include/sky_hosek.h`
- `intern/sky/source/sky_hosek.cpp`
- `intern/sky/source/sky_hosek_data.h`

The source and coefficient tables are used only to cook immutable model data
on the host. Direction-dependent sky evaluation remains in the Luisa shader
and therefore executes through every selected Psycles device backend.

The files are licensed under BSD-3-Clause. See [LICENSE](LICENSE).
