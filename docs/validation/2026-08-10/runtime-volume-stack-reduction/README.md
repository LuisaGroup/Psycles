# Runtime volume-stack predicate reduction

Date: 2026-08-10

## Problem

The path-volume and shadow-volume pipelines each classified a volume stack by
walking `VolumeStack::maximum_entries()` in a C++ host loop. Luisa therefore
recorded one copy of the surface-flag predicate for every possible stack slot,
even though the active count and entries are device-local path state. The
scene-selected stack capacity changed shader code shape and render-cache
identity.

This was not a small fixed-vector expansion. The capacity is clamped only by
`maximum_volume_stack_size == 32`, so the same predicate body could be cloned
up to 31 times in each of two transport paths.

## Formal replacement

Heterogeneous classification is the existential reduction

```text
exists e in stack[0 .. count): heterogeneous(surface_flags(e)).
```

`VolumeStack::any` now records that relation as one device loop with an early
exit after the predicate becomes true. Both callers supply their original
typed predicate as a host-stage lambda. The stack remains device-local and no
entry, shader flag, or material value is copied to the host.

For every legal storage capacity `S`, the code-shape invariant is

```text
instructions(any, S) = instructions(any, 2)
loops(any, S) = 1.
```

The XIR regression constructs independent kernels at the minimum and maximum
capacities. Both contain exactly 149 instructions and one loop. It therefore
fails if a later implementation reintroduces capacity-dependent host
unrolling, even if numeric tests happen to use a small stack.

## Validation

`psycles_luisa_volume_stack_tests` also evaluates true and false predicates on
populated stacks. The pre-existing cases continue to cover Cycles stack
identity, entry/exit transitions, capacity, duplicate rejection, camera
enclosure discovery, shadow copying, cleanup, and sample-method reduction.

| Check | fallback | HIP | Vulkan native XIR/SPIR-V |
| --- | ---: | ---: | ---: |
| min/max-capacity XIR shape | pass | shared host check | shared host check |
| populated-stack predicate and existing numeric matrix | pass | pass | pass |

HIP selected the RX 9070 XT. Vulkan selected the same GPU through RADV and
compiled the test through native XIR-to-SPIR-V (`12,963 -> 8,831` words), with
no DXC route.

Lone Monk has no active volume pipeline, so this correction is not included in
its surface-only JIT speedup. Its purpose is to bound scene-driven kernel
growth for Classroom, Barbershop, Splash, and other volume-capable scenes.
