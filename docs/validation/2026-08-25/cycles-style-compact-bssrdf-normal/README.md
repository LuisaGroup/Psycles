# Cycles-style compact BSSRDF exit-normal evaluation

## Outcome

The random-walk BSSRDF exit-normal query now executes the same compact typed
surface program as surface population. It no longer records one
topology-expanded Luisa callable body per material graph. The source graph and
its original closures remain device data and device expressions; Blender and
Cycles are not used to pre-evaluate or bake any material value.

This is an intermediate step toward the Cycles 5.2 architecture in which one
SVM program evaluates the surface graph and populates the local closure array.
It removes the last BSSRDF-normal topology switch, but it does not yet claim
that every downstream BSDF consumer has been converted to the single compact
population.

## Formal reduction

Let the surface closure program emit source-ordered candidates
`C = (c_0, ..., c_(m-1))`, and let `K` be the physical closure capacity. The
allocation recurrence is

```text
n_0 = 0
keep_i = scattering(c_i)
         and allocation_weight(c_i) >= closure_weight_cutoff
         and n_i < K
n_(i + 1) = n_i + (keep_i ? 1 : 0)
```

Only retained BSSRDF closures contribute to the exit-normal reduction:

```text
w_i = abs(average(weight(c_i)))
W = sum(w_i | keep_i and bssrdf(c_i))
N = sum(w_i * normal(c_i) | keep_i and bssrdf(c_i))
result = (W == 0) ? ShaderData::N : normalize(N)
```

The exact-zero test is intentional and matches Cycles. The capacity is clamped
to `[1, maximum_surface_closure_capacity]`, so the compact interpreter and the
topology-expanded diagnostic route share one allocation state machine and
cannot disagree at the capacity boundary.

Cycles guards this work with the per-shader `has_bssrdf_bump` flag. A topology
tag alone is insufficient because multiple parameter blocks can share a graph
while making all BSSRDF branches unreachable. Psycles therefore carries the
exact material predicate in `MaterialBindingGpu::flags`; when it is false, the
device branch returns the incoming shading normal without evaluating the
surface program.

## Implementation structure

- `SurfaceBssrdfNormalAccumulator` is the shared allocation and reduction
  state machine used by compact and expanded execution.
- `make_compact_surface_bssrdf_normal_callable()` evaluates the typed bytecode
  once and streams original closure records into that accumulator.
- Surface-value runtime construction and upload were moved into
  `path_tracer_surface_value_runtime.cpp`. The interpreter implementation is
  now 1,653 lines and the runtime module is 462 lines; neither source file
  exceeds the project's 2,000-line maintenance ceiling.
- Runtime topology metadata now has a generic `topology_flag` contract instead
  of encoding automatic bump semantics in the buffer-slot name.

## Regression matrix

The compact preparation test adds two real BSSRDF bump graphs. One exercises a
nonzero weighted normal reduction and one exercises exact-zero fallback to the
graph-evaluated `ShaderData::N`. The same topology is also run with the
per-material predicate disabled, proving that it neither evaluates nor changes
the normal. Existing surface-program metadata tests cover the exact host-side
predicate for closure reachability and displacement bump.

The complete project was built with all 32 hardware threads:

```sh
cmake --build build --parallel 32
```

Focused host and fallback validation passed 8/8:

```sh
ctest --test-dir build --output-on-failure -j 6 \
  -R 'psycles\.(surface_program_metadata|surface_closure_execution_plan|luisa_(surface_closure_collection|surface_population|compact_surface_preparation|principled_setup_callable|principled_thin_wall|subsurface_exit)_fallback)$'
```

The corresponding six AMD HIP and six strict native Vulkan XIR-to-SPIR-V
tests passed 12/12. The Vulkan route used:

```sh
LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1
```

```sh
ctest --test-dir build --output-on-failure -j 1 \
  -R 'psycles\.luisa_(surface_closure_collection|surface_population|compact_surface_preparation|principled_setup_callable|principled_thin_wall|subsurface_exit)_(hip|vk)$'
```

No DXC compilation was observed in the strict Vulkan run.

## Next compiler boundary

Cycles compiles a `ShaderGraph` into one data-driven SVM stream. Its compiler
walks dependencies from output roots, schedules the DAG with a deterministic
Sethi-Ullman heuristic, reuses stack storage after the final consumer, inserts
runtime branch skips for exclusive Mix dependencies, and emits closure
population once. Psycles already has the required host graph and typed device
bytecode; the next change is to make that existing IR obey the same scheduling
and single-population model rather than adding another graph representation.
