# Cycles-style surface value DAG scheduling

## Result

Psycles now schedules its existing `SurfaceProgram` graph IR with the same
deterministic Sethi-Ullman DAG heuristic used by the Cycles 5.2 SVM compiler,
then performs exact last-use allocation in that selected order. This is a
graph-compiler change: no shader value is evaluated on the host, no closure is
pre-baked, and the compact device instruction count is unchanged.

The scheduler operates on immutable graph structure only. Material parameters,
resolution, sample count, dispatch chunk size, and coroutine frame capacity do
not affect the schedule or shader cache identity. Original
`ValueExpressionId` is the deterministic tie breaker.

## Formal model

Every active non-parameter value is a vertex in a DAG. An edge `u -> v` means
that `v` reads `u`; terminal closure inputs have one synthetic consumer at the
end of the value stream. Let `w(v)` be the result width in 32-bit words and
`c(v)` the number of distinct live consumer instructions plus that optional
terminal consumer. For producers `p_i`, sorted by descending
`current(p_i) - w(p_i)`, the recurrence is

```text
current(v) = c(v) > 1 ? w(v) : SU(v)

SU(v) = max_i(sum_(j<i) w(p_j) + current(p_i),
                  sum_i w(p_i) + w(v))
```

This ordering is optimal for trees and a heuristic for DAGs. A dependency walk
from all sinks emits each value exactly once. The existing storage allocator
then enforces the stronger read-before-write contract: all operands must have
live locations before any final-use location can be reused by the result.

Reordering is semantics-preserving because value instructions are pure. Their
complete device inputs are operand values, immutable bytecode metadata,
late-bound material parameters, `ShaderServices`, and `SurfacePoint`.

## Typed-bank refinement

Cycles allocates one scalar SVM stack, while Psycles has separate scalar,
vector, and uint64 banks. Blindly selecting the scalar-stack heuristic is not
formally monotonic for that representation. A four-node counterexample is:

```text
vector root -> scalar chain output
           \-> vector output
```

Cycles ordering keeps two vector values live and requires 28 B across typed
banks, while the original legal topological order requires 16 B. The planner
therefore measures exact bank pressure for both candidate schedules and
selects the original order only when its payload is strictly smaller. Equal
pressure selects the Cycles order. This proves that the change cannot increase
the planned typed-local payload relative to the previous compiler.

The regression suite contains both this counterexample and a scalar graph in
which source order needs three slots but Sethi-Ullman needs two.

## Exact scene A/B

The baseline is commit `b7d330c`, rebuilt in the same worktree before applying
the scheduler. Both sides use the same Blender 5.2 exports and the same
`psycles_inspect_blender_material` executable source. Bytecode sizes and
instruction counts are identical; only the legal instruction order and slot
addresses change.

| Scene | Topologies | Max payload before | Max payload after | Sum before | Sum after | Sum reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Barbershop | 189 | 144 B | 128 B | 10,360 B | 8,460 B | 18.34% |
| Monster Under the Bed | 18 | 64 B | 64 B | 596 B | 500 B | 16.11% |
| Classroom | 59 | 64 B | 60 B | 1,496 B | 1,468 B | 1.87% |

Barbershop's maximum bank dimensions change from 6 scalar / 10 vector to
6 scalar / 9 vector. Classroom changes from 4 scalar / 4 vector to
3 scalar / 4 vector. Monster's maximum is unchanged, but lower-pressure
non-maximum programs reduce the aggregate payload.

Reproduction:

```sh
cmake --build build --parallel 32 --target \
  psycles_inspect_blender_material \
  psycles_surface_program_metadata_tests

./build/bin/psycles_inspect_blender_material \
  /home/mike/Projects/psycles-benchmarks/barbershop-480p-64spp/export '*'
./build/bin/psycles_inspect_blender_material \
  /home/mike/Projects/psycles-benchmarks/monster-480p-64spp/export '*'
./build/bin/psycles_inspect_blender_material \
  /home/mike/Projects/psycles-benchmarks/classroom-480p-64spp/export '*'
```

## Validation

The complete project builds with all 32 hardware threads. The focused surface
matrix passes 20/20 across host, fallback, AMD HIP, and strict native Vulkan
XIR-to-SPIR-V:

```sh
cmake --build build --parallel 32

LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
ctest --test-dir build --output-on-failure -j 1 \
  -R 'psycles\.(surface_program_metadata|surface_closure_execution_plan|luisa_(surface_closure_collection|surface_population|compact_surface_preparation|principled_setup_callable|principled_thin_wall|subsurface_exit)_(fallback|hip|vk))$'
```

## Rejected scene-sized local arrays

A follow-up experiment wired the proved scene maxima directly into Luisa
`Local` extents at host/JIT construction time. Barbershop therefore recorded
6 scalar / 9 vector / 1 uint64 slots instead of the fixed 8 / 12 / 1 ceiling.
This was semantically valid and passed fallback, HIP, and strict native-XIR
Vulkan tests, but it is not committed.

The exact HIP A/B used the same scheduler revision, export, RX 9070 XT,
640x480, 64 spp, and `wavefront-staged` configuration. Two ordinary warm runs
were 3.60243/3.60155 s with fixed arrays and 3.59830/3.59682 s with scene-sized
arrays: a 0.12% mean difference inside run-to-run noise. An additional fixed
run was 3.59641 s and an additional scene-sized run was 3.60823 s, confirming
that the distributions overlap.

`rocprofv3 --kernel-trace --scratch-memory-trace --stats` reported:

| Main path kernel metric | Fixed 8/12/1 | Scene 6/9/1 |
| --- | ---: | ---: |
| Packaged HIP cache artifact | 641,287 B | 624,135 B |
| Private segment | 6,000 B | 5,856 B |
| SGPR / VGPR | 128 / 256 | 128 / 256 |
| Profiled render-only | 3.61619 s | 3.60907 s |
| Main-kernel mean invocation | 5,341.812 us | 5,420.390 us |

The object and private segment became modestly smaller, but occupancy-limiting
register allocation did not change and no runtime gain was measurable. Exact
scene maxima would also create a distinct shader-cache identity for every
capacity tuple. That cache fragmentation is not justified by a 0.12% noisy
mean, so the experiment is preserved only as
`rejected-scene-specialized-svm-banks-no-runtime-gain` in the local stash.
Production retains the fixed upper-bound arrays; this commit claims the exact
graph working-set reduction, not a render speedup.

Profiler databases and A/B images are outside the source tree at:

```text
/var/tmp/rocprof-psycles-svm-banks-fixed/fixed_results.db
/var/tmp/rocprof-psycles-svm-banks-specialized/specialized_results.db
/var/tmp/psycles-svm-bank-ab/
```
