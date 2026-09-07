# Coroutine boundaries versus Cycles GPU queues

The boundaries are not yet fully isomorphic. Presence of correctly named
continuations does not prove the transition graph or scheduled work matches
Cycles. This audit checks actual call sites against Cycles 5.2.1
`cb168525138fecc792cc393f94afc39582b0103c`.

## Existing cuts

The staged renderer records `intersect_closest`, `shade_surface`,
`shade_background`, reachable `shade_light_forward`, `shade_volume` and
`intersect_subsurface` boundaries. Unreachable scene stages are omitted on
the host. Surface geometry and SVM closure population begin after the surface
cut, and finish before any subsequent suspension.

The independent direct-light queue has three separately dispatched stages:
`shade_light_nee`, `intersect_shadow`, `shade_shadow`. Constant emitters bypass
the first; opaque shadow hits terminate without shade; a full transparent-hit
batch can return from shade to intersection. The sequential diagnostic path
uses the corresponding named coroutine cuts instead of the side queue.

Monk's reported four main subroutines do not include those three auxiliary
stages. Main coroutine frame size is not Cycles' complete wavefront SoA state
size, nor the total memory of main and shadow paths.

## Fixed: unnecessary closest queue after SSS

Original `kernel/integrator/subsurface.h::subsurface_scatter` stores the exit
ray/intersection and queues `SHADE_SURFACE` directly in the ordinary admitted
case. MNEE and raytrace variants select their own queues. Failed transport
terminates in `INTERSECT_SUBSURFACE`.

The Psycles loop previously suspended unconditionally at `intersect_closest`
before examining `pending_subsurface_exit`. The following bounce setup did
correctly consume the stored hit without tracing. Thus it added a scheduling
edge and frame round trip, not an extra geometric intersection.

The production cut now tests that pending-hit state before suspending. On a
successful SSS exit, the existing hit-consumption setup runs in the SSS
continuation and its next suspension is `shade_surface`. No hit reconstruction,
RNG, closure arithmetic or Luisa compiler behavior is changed. The guard is
not generated in scenes without BSSRDF.

The reduced permanent GPU test invokes this same production cut helper and
reads actual Wavefront scheduler continuation counts. It tests ordinary,
repeated-SSS, alternating and failed-SSS paths, 43 logical paths with capacity
19, both AoS/SoA frames, repeated dispatch and a no-SSS negative control.

Before the fix, the two-successful-SSS case visits closest 129 times instead
of the original scheduling contract's 43. The alternating case visits 129
instead of 86. Both layouts fail; after the fix the focused HIP test passes.
Logs: `subsurface-cut-red-hip.log`, `subsurface-cut-green-hip.log` under
`/var/tmp/psycles-native-volume-svm-06XnDX`.

The permanent queue-count regression also passes fallback and strict native
XIR-to-SPIR-V Vulkan, including both frame layouts and repeated dispatches.
The complete HIP selection passes 171/171. The complete fallback selection
has one separate persistent-scheduler failure: its LLVM block-barrier frames
overflow the backend's fixed 4 MiB arena. This is not the main path's small
Luisa coroutine frame and does not make the complete fallback suite green.

The original Monster scene is rerendered at 1080x1080 / 256 spp after this
fix, with SVM enabled by default, native fast math and the separate direct
light queue. All 46 channels are finite. Combined / DiffInd relative RMSE
against the original Cycles HIP EXR is 0.00547924 / 0.02552799; the main
coroutine remains six subroutines, 71 fields / 284 bytes. Its single 15.1410 s
render is an integration check, not a paired speedup measurement. Exact
sources and logs are in the [default-path checkpoint](../native-default/README.md).

This cannot explain Monk's residual or establish a Monk speedup: its native
scene feature mask does not generate the SSS stage in the first place.

## Remaining concrete discrepancies

| Cycles boundary or edge | Current Psycles state |
| --- | --- |
| INIT_FROM_CAMERA -> INTERSECT_VOLUME_STACK when required | Camera volume-stack initialization is still inline in sample setup. |
| Separate SHADE_VOLUME / SHADE_VOLUME_RAY_MARCHING routing | A single `shade_volume` continuation contains the current volume consumer. |
| SHADE_LIGHT_FORWARD -> INTERSECT_CLOSEST | The event loop still reuses its previously obtained mesh/background hit and resolves the next lamp/event without the Cycles closest requeue. |
| SHADE_SURFACE_RAYTRACE, MNEE and dedicated-light stages | Not fully implemented/admitted; no complete correspondence claimed. |

The forward-light difference follows directly from
`kernel/integrator/shade_light.h::integrator_shade_light_forward` versus the
Psycles event loop. Matching the final image on an isolated lamp does not
prove this scheduling edge equivalent for all volume/geometry interactions.
It remains alignment work, not a claimed optimization or an explanation of
unrelated scene performance.

The existing stage-ABI validator checks reachable stage presence and rejects
unreachable names. It does not by itself prove all edges. Future boundary
changes must retain original Cycles state transitions and add queue-execution
oracles, followed by full original-scene validation. Scheduler tuning stays
in Psycles through generic Luisa coroutine extensions/handlers.
