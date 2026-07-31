# Closest-event volume-segment foundation

This checkpoint integrates a homogeneous medium before every closest mesh,
lamp, or background event in the production path kernel. Constant raw Volume
closure graphs are now accepted by scene compilation. Spatially varying raw
graphs remain explicitly gated until heterogeneous integration is present;
volume direct lighting is still open, so this is a finite-segment transport
checkpoint rather than a complex-scene visual-parity claim.

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
closest-event, Sobol, or path-state policy. `PathVolumeSegmentStage` owns those
policies and composes this estimator through the same host-stage virtual
pipeline as surface stages while Luisa records one fused kernel AST.

`cycles_path_state::next_volume()` is a total transition: flags, visibility,
bounce counters, volume counter, and RNG offset cannot be updated piecemeal.
The shared `continuation_probability()` function is the single source for
surface and volume roulette. Scene sync now carries the Cycles-adjusted volume
bounce limit into kernel parameters. `VolumeStack::clean_for_background()`
models the background cleanup as a separate stack operation.

`VolumeProgramCapabilityComponent` walks dependencies reachable from the raw
volume-closure root. Production scene compilation accepts only programs proven
independent of spatial shader values; it reports a capability diagnostic for a
Generated-coordinate-dependent density instead of silently evaluating that
heterogeneous graph once per segment.

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

The new `psycles.luisa_volume_path_{fallback,hip,vk}` regression exercises the
actual `compile_scene → build_path_kernel → device.compile → dispatch` path.
Its camera starts inside a finite transparent-boundary box, so the medium is
initialized by the production TLAS enclosure query. The kernel attenuates the
1.9-unit segment before the exit surface, crosses the raw transparent closure,
and then evaluates the white world. It also checks Environment routing, zero
Volume Direct/Indirect for absorption-only transport, and the heterogeneous
capability rejection.

## Official Cycles EXR oracle

The oracle is reproducible without a CPU reference renderer:

```text
blender --background --factory-startup \
  --python tools/create_cycles_volume_path_oracle.py -- \
  /var/tmp/psycles-volume-path-oracle.exr
oiiotool --dumpdata /var/tmp/psycles-volume-path-oracle.exr
```

Blender 5.2.0 LTS/Cycles CPU, one sample, Box filter, and 32-bit scene-linear
OpenEXR produced the same constant value in all 16 pixels:

```text
(0.467666417, 0.621885061, 0.826959133, 1.000000000)
```

Fallback, HIP, and Vulkan reproduce every channel within `3e-6`; the focused
tests pass on all three backends. The EXR was inspected with `oiiotool --stats`
and `--dumpdata`: all channels are finite, standard deviation is zero, and no
NaN or infinity is present. A triptych would contain three uniform swatches
and add no visual evidence, so the first meaningful volume triptych remains
the lit/scattering scene after volume NEE is connected.

## Remaining release gate

Closest-event free flight, phase continuation, volume emission/pass routing,
roulette reuse at an attenuated emissive surface, and the terminate-on-next
surface exception are active. Volume NEE, VSPG history, and heterogeneous grid
transport still remain. Until volume NEE is complete, complex Blender volume
scenes stay outside the compatibility claim even though homogeneous materials
now execute in production.
