# Sparse XIR verifier dominance

Date: 2026-08-10

## Outcome

LuisaCompute `next@f6a9b2728` replaces the XIR verifier's materialized
all-dominators relation with block numbering, a sparse predecessor CSR, and a
sparse immediate-dominator tree. The verifier still checks the complete pass
boundary at the beginning and end of `restructure_cfg`; it does not weaken,
sample, or disable verification.

On the unchanged 37-material Lone Monk production kernel, whose generated
SPIR-V is byte-count-identical in size to the preceding checkpoint, cache-cold
Vulkan shader JIT falls from `569.378 s` to `180.533 s`, a `3.15x` speedup.
The new run spends `1.009 s` in the native SPIR-V handoff verifier. Peak RSS
remains `9,416,224 KiB` (`8.98 GiB`) because this repair changes verifier
complexity rather than the still-large generated shader.

This is not the end of the compiler performance work. `restructure_cfg`
still takes `53.138 s`, now dominated by repeated mutation-oriented scans in
its post-restructure fixed point. The driver then takes another `86.780 s` to
create the monolithic RADV pipeline. Those are separate remaining problems;
neither should be hidden by attributing all time to verification.

## Formal representation

Let the verifier-sanitized reachable CFG be `G = (V, E)`. The old verifier
stored a hash set `D(b)` for every block and iterated

```text
D(entry) = {entry}
D(b) = {b} union intersection(D(p) for p in predecessors(b)).
```

Initializing every non-entry `D(b)` to `V` already requires quadratic
storage. Repeated hash-set copies and intersections make the time cost much
worse on the thousands of blocks emitted by a production path kernel.

The replacement is:

1. Traverse only locally owned, reachable successor edges. This is required
   because a verifier must diagnose malformed cross-function targets without
   dereferencing them as part of a valid function CFG.
2. Assign each reachable block one reverse-postorder value number.
3. Convert predecessor pointers once to a CSR of numeric block IDs. The
   fixed-point loop performs no pointer hashing.
4. Run Cooper-Harvey-Kennedy over one `idom[id]` value per block.
5. Encode the resulting `V - 1` tree edges with compact first-child and
   next-sibling arrays, then assign preorder/subtree intervals iteratively.
6. Answer `dominates(a, b)` by tree ancestry:

```text
pre[a] <= pre[b] < subtree_end[a].
```

The CFG and construction storage is `O(V + E)` and retained dominance state
is `O(V)`; there is no `V x V` dominance relation. Each completed dominance
query is `O(1)` after two block-ID lookups. Cooper construction can require
multiple sweeps in its worst case, so the verifier exposes the sweep count
rather than claiming a false worst-case linear-time bound. A
Lengauer-Tarjan replacement is justified only if measured CFGs show the
fixed-point convergence itself becoming material; Lone Monk's current
remaining cost is elsewhere.

The implementation is isolated in real `.h` and `.cpp` files rather than
growing `verifier.cpp` or hiding the code in an `.inl` fragment.

## Regression coverage

The new large-CFG regression builds a chain of 2,048 diamonds:

| Structural quantity | Required value |
| --- | ---: |
| Reachable blocks `V` | 6,145 |
| Reachable CFG edges `E` | 8,192 |
| Stored idom tree edges | 6,144 (`V - 1`) |
| CHK sweeps | at most 3 |

Each merge also consumes a value defined at the dominating diamond header, so
the test covers interval-query semantics as well as storage shape. A separate
malformed-module regression branches from one function into another and
requires that the illegal edge be diagnosed while contributing zero edges to
either numbered dominance CFG.

Focused measurement after a full rebuild:

```text
test_xir_verifier: 46/46 tests, 211 assertions
task-clock:         21.49 ms
cycles:             104,896,355
instructions:       219,034,379
cache misses:       366,152
```

The complete `unit_xir` label passes `48/48` tests in `0.28 s` real time.
Five native SPIR-V structure/handoff tests pass, and
`test_vk_spirv_codegen_path vk` passes `92/92` runtime tests with `2,096`
assertions on the RX 9070 XT.

An earlier invocation of the complete XIR suite was deliberately discarded:
only the focused executable had been relinked after the public verification
statistics layout changed, so old test binaries called the new shared library
with a stale return-object ABI and stack-smash diagnostics. Rebuilding all
targets with all hardware threads removed that invalid variable before the
reported `48/48` run.

## Lone Monk cold-JIT measurement

Command:

```sh
PSYCLES_DISABLE_SHADER_CACHE=1 \
LUISA_LOG_LEVEL=verbose \
LUISA_VULKAN_PROFILE_COMPILATION=1 \
LUISA_XIR_TRACE_PASSES=1 \
build/bin/psycles_render_blender_scene \
  /var/tmp/psycles-lone-monk-transmission-dbdcb17/export \
  /var/tmp/psycles-sparse-dom-20260810/vk.ppm \
  vk 1 1 1 1
```

The 1x1/1-spp launch is a compile smoke: Psycles still records and compiles
the complete scene path kernel, so resolution and sample count do not reduce
the shader. No rendering-algorithm or scene input changed in this checkpoint.

| Boundary | Previous/current evidence | Sparse-idom run |
| --- | ---: | ---: |
| Raw SPIR-V | 1,431,985 words | 1,431,985 words |
| Optimized SPIR-V | 1,116,158 words | 1,116,158 words |
| Complete shader JIT | 569.378 s | 180.533 s |
| `restructure_cfg` | previously verifier-dominated in profiled large kernels | 53.138 s |
| Restructure input verifier | not separately timed | 315.846 ms |
| Restructure output verifier | not separately timed | 320.179 ms |
| Native handoff verifier | 289.373 s on the larger rejected outline experiment | 1.009 s |
| Native AST-to-SPIR-V | not separately comparable | 93.597 s |
| RADV pipeline creation | about 87 s | 86.780 s |
| Peak RSS | about 8.98 GiB | 8.98 GiB |

The old `289.373 s` handoff value came from the rejected outlined-closure
experiment with a larger 1,726,431-word raw module, so it is evidence for the
old verifier pathology, not a strictly apples-to-apples speedup denominator.
The `569.378 s -> 180.533 s` complete JIT comparison is apples-to-apples: both
runs generate the same 1,431,985/1,116,158-word module from the same export.

## Remaining restructure cost

Pass-internal tracing attributes the `53.138 s` restructure interval as
follows. Timers are nested, so rows must not be summed indiscriminately.

| Repeated analysis or transform | Aggregate time |
| --- | ---: |
| `post_restructure_fixed_point` | 32.493 s |
| `drain_selection_exits` | 13.897 s |
| `fixup_construct_exits` | 10.735 s |
| `main_loop_iteration` | 9.421 s |
| `canonicalize_loop_boundary_selection_merges` | 5.788 s |
| `enforce_unique_construct_entries` | 4.331 s |
| final selection-reentry audit | 3.817 s |
| all post-dominator rebuilds | 0.971 s |

The first large post iteration changes the graph, but the final unchanged
iteration still spends `5.122 s` running mutation-oriented whole-graph work.
The next optimization therefore needs CFG-versioned analyses and a worklist
whose invalidation follows mutations. Skipping checks unconditionally would
be semantically wrong, and adding scene-specific cases would only hide the
formal scheduling defect.

The complete local log is
`/var/tmp/psycles-sparse-dom-20260810/vk.log`.
