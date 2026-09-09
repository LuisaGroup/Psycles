# Native shadow-origin staging

This checkpoint restores the two production stages used by Cycles 5.2.1 for
surface direct-light rays. `shadow_ray_offset` now keeps its smooth-triangle,
positive-cutoff and positive-amount guards around the terminator arithmetic
and smooth-surface geometry. Direct-light transport then derives finite-light
direction and distance from that displaced position. The exact source-triangle
certificate runs afterward with the final setup direction; it may change the
origin, but never re-aims the ray.

The implementation is in `include/psycles/luisa/surface_ray.h`,
`src/luisa/path_kernel_surface_geometry.cpp`, and
`src/luisa/path_kernel_direct_light_transport.cpp`. The existing combined
helper remains available for low-level compatibility tests; production direct
light uses the split methods on `SurfaceGeometryContext`.

Validation completed on the RX 9070 XT with the root build configured for HIP,
fallback and Vulkan:

- all-target root build with `cmake --build build --parallel 32`;
- `psycles.luisa_surface_ray_fallback` and `psycles.luisa_surface_ray_hip`;
- `psycles.luisa_cycles_shadow_pipeline_hip`;
- `psycles.luisa_cycles_svm_native_shadow_hip`;
- `psycles.luisa_cycles_svm_shadow_surface_hip`;
- the three shading-terminator recording/scene/fixture controls.

All listed tests passed. The added surface-ray assertions compare the split
terminator-plus-certificate path with the established combined construction
for both certified and ambiguous triangle origins. No full-scene timing pair
is claimed by this checkpoint; fresh exporter identity and SDK317 scene gates
remain prerequisites for that measurement.
