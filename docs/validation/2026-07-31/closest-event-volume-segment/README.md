# Closest-event volume-segment foundation

This checkpoint establishes the total state and estimator contracts needed to
integrate a homogeneous medium before every closest mesh, lamp, or background
event. It intentionally keeps scene volume materials release-gated until the
path-kernel stage and volume direct lighting are complete; this is not yet a
volume-render or visual-parity claim.

## Official Cycles contract

The implementation is pinned to official Blender/Cycles main
`b82c3f0da6c1813dabedc563d64e536f4d83e868`, principally
`integrator/intersect_closest.h`, `integrator/intersect_closest_functions.h`,
`integrator/shade_volume.h`, `integrator/volume_shader.h`,
`integrator/path_state.h`, and `scene/shader_graph.cpp`.

- A closest surface, lamp, or background distance first bounds the active
  volume segment. A real volume collision discards that event and restarts
  closest-event traversal from the collision.
- Cycles uses dimensions 3/4/5 of the current 16-dimensional path block for
  phase, reservoir/channel, and scatter distance respectively. Analytic lamps
  split a ray into segments without advancing that block.
- Volume `ShaderData::wi` is `-ray.D`; phase sampling uses `-sd->wi`, hence the
  phase axis is the propagation direction `ray.D`.
- A volume collision increments the ordinary and volume bounce counters,
  advances the Sobol offset by 16, switches visibility to volume scatter,
  updates MIS/pass flags atomically, and applies the scene-synced
  `volume_bounces + 1` limit.
- Continuation probability is exactly
  `min(sqrt(max(abs(throughput))), 1)`, subject to Cycles' distinct ordinary
  and transparent minimum-bounce predicates.
- On a background miss, only a world entry may survive
  `volume_stack_clean()`. Leaked object entries must not attenuate the ray to
  infinity.
- Shader closure storage has a global allocation budget separate from its
  live count. Surface closures consume their ordinary allocation, while every
  reachable volume closure node reserves a 32-entry volume-stack block; the
  result saturates at `MAX_CLOSURE == 64`.

## Architecture

`HomogeneousVolumeSegmentComponent` is a genuine `.h + .cpp` host-stage
interface. Its implementation composes `StackedVolumeEvaluator`,
`HomogeneousVolumeTransport`, and `VolumePhaseSet` through ordinary C++
polymorphism while Luisa records one fused device AST. It owns no film,
closest-event, Sobol, or path-state policy, keeping those concerns in the
forthcoming path stage.

`cycles_path_state::next_volume()` is a total transition: flags, visibility,
bounce counters, volume counter, and RNG offset cannot be updated piecemeal.
The shared `continuation_probability()` function is the single source for
surface and volume roulette. Scene sync now carries the Cycles-adjusted volume
bounce limit into kernel parameters. `VolumeStack::clean_for_background()`
models the background cleanup as a separate stack operation.

## Regression coverage

The focused regressions cover:

- fallback, HIP, and Vulkan end-to-end raw graph → stacked coefficients →
  homogeneous collision → reservoir-selected HG continuation;
- the exact official HG direction/PDF, propagation axis, selected closure,
  rescaled random, roughness, distance, event PDF, throughput, emission, and
  empty-stack identity;
- total volume path-state transition and continuation probability on
  fallback, HIP, and Vulkan;
- world-preserving and world-free background stack cleanup on fallback, HIP,
  and Vulkan;
- one-volume-node allocation of 33 slots including its surface closure, and
  saturation at 64 for two volume nodes; and
- Blender-to-kernel synchronization of `volume_bounces + 1`.

## Remaining release gate

The next checkpoint connects this component before every closest event,
preserves the same random block across transparent analytic-light segments,
routes volume emission and volume passes, and implements the terminate-on-next
surface exception. Volume NEE and heterogeneous grid transport still remain
after that. No EXR or triptych is claimed here because this checkpoint does
not yet emit production scene radiance.
