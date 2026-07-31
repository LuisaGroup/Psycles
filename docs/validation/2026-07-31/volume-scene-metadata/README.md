# Volume scene metadata

This checkpoint validates the host preprocessing that specializes Psycles'
Luisa path AST for a scene's volume nesting and camera placement. It does not
evaluate radiance and is not a CPU reference renderer.

## Cycles contract

The implementation is pinned to official Blender/Cycles main
`b82c3f0da6c1813dabedc563d64e536f4d83e868`:

- `scene/geometry.cpp` derives `Geometry::has_volume` from the effective
  shader set.
- `scene/object.cpp` transforms object bounds and marks objects whose bounds
  overlap a volume object.
- `scene/scene.cpp::get_volume_stack_size()` reserves the background and
  terminator slots, reserves one object slot for otherwise disjoint volumes,
  increments for each intersecting volume object, and clamps to
  `MAX_VOLUME_STACK_SIZE == 32`.
- `scene/camera.cpp::viewplane_bounds_get()` conservatively expands the active
  near view plane by the near clip and maximum aperture radius before testing
  object-volume bounds.
- `scene/shader_graph.cpp::get_num_closures()` counts ordinary surface
  allocations and reserves one `MAX_VOLUME_STACK_SIZE == 32` block per
  reachable volume closure node, saturating at `MAX_CLOSURE == 64`.

Psycles resolves instance material overrides before classifying a boundary,
transforms all eight corners of the object-space AABB so negative scale is
safe, and retains Cycles' intentionally conservative pairwise overlap count.
It does not substitute a nesting solver whose smaller answer would change
kernel state sizing relative to Cycles.

At kernel construction, the camera bound uses the same projection, aspect,
sensor fit, lens shift, near clip, aperture radius/ratio, and focal distance
already used by the Luisa camera sampler. A world-only volume needs the
background and terminator but no enclosure ray query. A volume-free scene uses
host stack size zero only to omit all volume locals and queries from its
specialized Luisa AST.

## Regression result

`psycles.volume_scene_metadata` covers:

- overlapping object volumes, including Cycles' conservative per-object
  count;
- multiple disjoint volumes sharing the one required object slot;
- world-only and fully volume-free scenes;
- effective volume-to-surface and surface-to-volume material overrides;
- transformed bounds with a reflected, non-uniform scale;
- saturation at 32 stack slots;
- perspective, orthographic, and panorama camera overlap;
- camera translation, near-plane extent, and anamorphic aperture expansion.
- closure-allocation budgeting for one and multiple reachable volume nodes.

The focused test passes. The full build uses 32 parallel jobs; the complete
72-test CTest suite and the handwritten-source-size gate are run before the
checkpoint is committed.

## Visual scope

There is no triptych for this host-state checkpoint because it emits no
radiance. Scene volume materials remain release-gated while surface boundary
updates, stacked-medium transport, phase continuation, and volume direct
lighting are incomplete. Linear-EXR Cycles/Psycles triptychs begin when those
stages are enabled together.
