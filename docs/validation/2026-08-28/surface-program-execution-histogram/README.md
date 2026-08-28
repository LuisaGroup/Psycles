# Hit-weighted surface-program execution histogram

## Question

The static Barbershop surface image contains 8,870 value instructions, but
static program size does not say which operations dominate rendered work.  The
diagnostic answers the exact dynamic question without adding one atomic to
every SVM instruction.

For topology `t`, let `H[t]` be the number of compact populate-once surface
executions and let `P[t]` be its immutable preparation program.  Every valid
population executes every instruction of `P[t]` once in topological order.
Therefore the execution count of an instruction or exact handler class `c` is

```text
E[c] = sum_t H[t] * multiplicity(c, P[t]).
```

The device records exactly one topology event per population.  It distributes
events over 64 float-atomic lanes per topology, and the host accepts a lane
only when it is finite, nonnegative, integral, and below the `2^24`
consecutive-integer boundary.  It then projects the accepted counts through
the final bytecode and uses checked 64-bit addition.  Thus `exact=true`
implies the equation above; saturation, malformed ranges, overflow, and
unsupported non-compact/non-populate-once execution fail closed.

Closure counts use the same projection for the first topological traversal.
They are explicitly reported as instruction visits: a zero-weight leaf is
visited by the traversal but does not invoke its decoder.  This distinction
prevents the diagnostic from overstating physical-closure allocation.

The option is host/JIT specialized.  A production renderer that does not
request the histogram contains neither the atomic nor a device-side enable
branch.

## Complex-scene evidence

All measurements used the Blender 5.2 exports, HIP, compact surface values,
populate-once, 320x240, 16 spp, and the staged wavefront scheduler:

```text
PSYCLES_COMPACT_SURFACE_VALUES=1 PSYCLES_POPULATE_SURFACE_ONCE=1 \
build/bin/psycles_render_blender_scene <export> <output.ppm> hip \
  320 240 16 16 - 0 0 0 0 16 - 1 0 wavefront-staged \
  32 32768 32 1 1 0 4 2 4096 0 0 0 1 1048576 \
  - <surface-program-histogram.json>
```

| Scene | populations | nonzero topologies | value executions | value instructions / population | closure visits / population |
|---|---:|---:|---:|---:|---:|
| Barbershop | 3,355,128 | 163 | 303,583,517 | 90.48 | 3.14 |
| Classroom | 2,911,443 | 54 | 83,674,008 | 28.74 | 1.73 |
| Monster Under the Bed | 2,529,325 | 14 | 59,698,110 | 23.60 | 1.30 |
| Lone Monk | 2,529,912 | 22 | 46,118,230 | 18.23 | 1.13 |

The seven hottest Barbershop value operations account for 193,880,590 of
301,392,157 handler executions (64.33%; the remaining 2,191,360 value
executions are explicit surface-normal transitions):

| operation | executions | handler share |
|---|---:|---:|
| Mix | 48,408,387 | 16.06% |
| Color to Scalar | 36,555,696 | 12.13% |
| Math | 28,275,527 | 9.38% |
| Color Ramp | 24,714,283 | 8.20% |
| Mapping | 19,184,309 | 6.37% |
| Scalar to Color | 18,569,465 | 6.16% |
| Image Color | 18,172,923 | 6.03% |

This rules out Principled closure decoding as the sole explanation for the
Barbershop surface gap.  Its value interpreter performs roughly four times as
many instructions per hit as the other complex scenes, and the hot work spans
typed conversion, arithmetic, table lookup, coordinate mapping, and image
sampling.  The next optimization must reduce common interpreter fetch,
dispatch, and operand-address work across this measured mixture; a patch for
one closure or one opcode cannot address the dominant structure.

## Regression matrix

The per-sample dispatch test connects a real value node to the surface and
checks all of the following:

- one-shot and chunked renders produce identical exact histograms;
- topology counts sum to the known number of surface events;
- handler executions plus normal transitions partition all value executions;
- closure kinds partition all first-traversal visits;
- leaf variants partition all leaf visits.

Commands and results:

```text
cmake --build build --parallel 32
ctest --test-dir build -R psycles.luisa_sample_dispatch_film_fallback \
  --output-on-failure
build/bin/psycles_luisa_sample_dispatch_film_tests hip
LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
build/bin/psycles_luisa_sample_dispatch_film_tests vk
```

All passed.  The Vulkan log reports native SPIR-V generation and contains no
DXC route.
