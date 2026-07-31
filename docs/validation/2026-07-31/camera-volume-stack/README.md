# Camera volume-stack traversal

This checkpoint validates Cycles' camera-enclosure traversal as an actual
Luisa ray-query program. It is a state/geometry regression, not a volume
render or a visual-parity claim.

## Reference and contract

- Official Blender/Cycles reference:
  `b82c3f0da6c1813dabedc563d64e536f4d83e868`.
- Reference code:
  `intern/cycles/kernel/integrator/intersect_volume_stack.h`,
  `volume_stack.h`, `geom/shader_data.h`, and
  `bvh/intersect_filter.h`.
- Probe ray: world-space `+Z`, `tmin = 0`, `tmax = FLT_MAX`.
- Traversal limit: `2 * volume_stack_size`.
- Candidate policy: reject non-volume primitives and the immediately
  preceding exact `(instance, primitive)` identity.
- Progress policy: Cycles' one-ULP `intersection_t_offset`, not an arbitrary
  epsilon.

The fixture constructs four visible triangle objects plus one hidden
negative-scale instance:

1. an open back-facing volume boundary enclosing the camera;
2. a nearer non-volume boundary which any-hit must reject;
3. a closed volume with a front entrance and back exit, which must not be
   placed in the initial stack;
4. a second open enclosing volume;
5. a hidden reflected instance used to check Cycles' geometric-normal
   orientation convention.

It also seeds a world-volume entry with independently exported Cycles object
and shader identities. The expected stack is therefore world, object 101,
object 104, followed by the terminator. The fixture invokes the same
`PathVolumeStateComponent` owned by the production path sample twice. The
first sample takes the enclosure path above; the second disables the probe
and must contain only the independently initialized world entry. This checks
both host-stage branches and proves the two move-only local stacks do not
alias.

## Backend result

The focused CTest group is:

```text
psycles.luisa_camera_volume_stack_fallback
psycles.luisa_camera_volume_stack_hip
psycles.luisa_camera_volume_stack_vk
```

All three pass. The thirteen output records check stack/intersection/enclosure
counts, all retained identities and raw graph dispatch handles, volume
candidate filtering, entrance/exit normals, negative-scale orientation, the
fixed probe direction, background-only initialization, and sample-local
ownership. The full Psycles build used 32 parallel jobs, and all 62 CTest
cases passed with `-j32`; the source-size gate also confirms every handwritten
implementation file remains below 2,000 lines.

## Luisa fallback defect found

The first fallback run crashed while reading the instance transform after the
ray query; HIP and Vulkan passed. The fault was not an Embree callback
classification problem. Fallback encoded an `AccelView` when the shader
command was submitted, while the preceding TLAS build was still queued. A new
accel therefore contributed a null instance-array snapshot even though the
build populated that array before shader execution.

Luisa `next` commit `c55b8d57b` changes the accel view to retain a stable
pointer to the backend-owned instance-data descriptor. Builds publish the
current vector storage only after all reallocating operations, and the
embedded device library dereferences the descriptor at execution time. The
fallback shader-cache ABI is incremented so an old inlined object cannot be
reused.

The Luisa regression builds and queries a fresh TLAS in one stream, then
checks tracing plus exact visibility, user-id, and all four transform columns.
`test_accel_visibility fallback` passes all 144 assertions.

## Visual scope

There is intentionally no triptych for this checkpoint: it emits stack state,
not radiance. Although the component is now owned by the real path sample,
scene volume materials remain release-gated until the downstream transport
stages are complete. Volume EXR comparisons and inspected triptychs begin
after surface boundary updates, free-flight, phase continuation, and volume
NEE are active together.
