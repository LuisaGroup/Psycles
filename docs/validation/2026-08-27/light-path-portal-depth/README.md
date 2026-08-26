# Light Path Portal Depth

## Outcome

Psycles now preserves Blender 5.2's `Light Path / Portal Depth` output as a
distinct typed surface operation. It is no longer silently aliased to
`Transmission Depth` by the Blender adapter.

Psycles does not yet admit Cycles' Ray Portal closure, so every path in the
currently supported program domain has portal depth zero. The Luisa evaluator
therefore specializes the new operation to zero. This is a semantic
specialization, not a socket fallback: when Ray Portal support is added, this
operation is the explicit point that must read the new path-state field.

## Formal correspondence

Cycles increments `portal_bounce` only when a sampled closure carries
`LABEL_RAY_PORTAL`. Let `P` be the set of paths reachable from the closures
currently admitted by Psycles and let `portal(p)` be the portal-bounce count.
Because no admitted closure can produce `LABEL_RAY_PORTAL`, induction over path
transitions gives

```text
portal(camera) = 0
not exists c in admitted_closures: label(c) contains LABEL_RAY_PORTAL
therefore forall p in P: portal(p) = 0
```

Aliasing Portal Depth to Transmission Depth violates this invariant whenever a
supported transmissive closure has already been sampled. The adapter now uses
an exhaustive socket-to-operation table and rejects unknown Light Path sockets
through the existing named unsupported-output diagnostic instead of selecting
a catch-all depth operation.

## Permanent regressions

The regression has three independent boundaries:

- the core graph and compact lowering contain one `path_portal_depth`
  instruction and no `path_transmission_depth` instruction;
- a Blender JSON graph linked from the exported `Portal Depth` socket lowers
  to the same distinct operation;
- the compact Luisa evaluator runs the same material twice with a path whose
  transmission depth is deliberately 13: Portal Depth emits zero while
  Transmission Depth emits 13.

The last check prevents a test from passing merely because the two enum names
are distinct. It executes on fallback, HIP, and strict native Vulkan
XIR-to-SPIR-V.

## Validation

The targeted all-thread build completed successfully. The focused tests passed:

```text
psycles.surface_program_metadata                       passed
psycles.blender_import                                 passed
psycles.luisa_compact_surface_preparation_fallback     passed
psycles.luisa_compact_surface_preparation_hip          passed
psycles.luisa_compact_surface_preparation_vk           passed
```

The Vulkan canary was run with
`LUISA_VULKAN_USE_XIR=1`,
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and
`LUISA_VULKAN_DISABLE_DXC=1`.
