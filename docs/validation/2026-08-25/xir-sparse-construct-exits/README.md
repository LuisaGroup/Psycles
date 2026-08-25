# Sparse XIR construct-exit enumeration

## Outcome

The native Vulkan XIR path no longer scans every function block while
enumerating the exits of every discovered construct. The restructuring pass
walks the exact sparse region support that it has already constructed, then
retains the existing stable-block sort before any target numbering or mutation.

This is the first compiler-scalability repair exposed by the 564-material
Barbershop shader. It removes the dominant `fixup_construct_exits` whole-CFG
scan, but it is intentionally not presented as a complete Barbershop Vulkan
fix: the next measured bottleneck is repeated global selection-exit relation
reconstruction and remains under investigation.

LuisaCompute `next` commits:

```text
1092a5126 xir: enumerate construct exits over sparse regions
eb9b7cc79 test: keep tensor copy source alive with system STL
```

Psycles records the resulting LuisaCompute submodule revision `eb9b7cc79`.

## Formal equivalence

Let `B` be the stable sequence of function-owned blocks and `R` the set found
by region discovery. The old enumeration was

```text
E_old = concat(successors(b) for b in B if b in R)
```

The new enumeration is

```text
E_new = concat(successors(b) for b in support(R))
```

where `support(R)` contains each member of `R` exactly once. Therefore the two
enumerations select the same boundary-edge set. Discovery order cannot affect
observable output because the selected candidate's exit edges are sorted by
the pre-existing stable block index before selector IDs or CFG rewrites are
created. The transformation, termination measure, and fail-closed checks are
unchanged.

The work bound changes from a full-function membership probe per construct,
`O(sum_c |B|)`, to the actual construct support and outgoing edges,
`O(sum_c (|R_c| + |E(R_c)|))`.

## Regression

`test_xir_pass_restructure_cfg_construct_exits` builds an outer selection with
256 sequential nested diamonds. It asserts exact analytical counters:

```text
region block visits      = 3 * 256
region edge visits       = 4 * 256
region membership probes = region edge visits
```

Unrelated function blocks are therefore never membership-probed by exit
enumeration. The counters are also exported in `PassReport`; the stable null
report schema test was updated to its actual 102 fields. That test had already
drifted by two fields on the incoming `next` revision, and this change restores
the contract rather than weakening it.

## Measured real-input evidence

The cold Barbershop native-XIR profile before the repair is:

```text
/var/tmp/psycles-barbershop-vk-cold-compile.perf.data
```

Its CPU samples attribute 58.50% to `fixup_construct_exits`, 27.31% to
`luisa::hash64`, and 1.05% to `DomTree::contains`. The run spent about 22
minutes in CPU restructuring before failing closed.

With sparse region enumeration, a traced run advanced past the old hotspot and
performed 187 selection-exit rewrites in roughly 41 seconds. The trace then
showed the next structural cost clearly: 184 of those rewrites were local
single-target Switch rewrites, yet each invalidated and rebuilt the complete
selection-exit relation. Full-build timings were about 129--168 ms per relation
rebuild, of which loop-boundary reconstruction alone cost about 94--103 ms.
Three non-local If rewrites correctly forced global invalidation. The run was
stopped after collecting this diagnostic evidence; it is not a render result.

Trace log:

```text
/var/tmp/psycles-primary-dispatch-matrix-20260825/barbershop-fresh/logs/psycles-vk.log
```

## Validation

After rebasing onto the then-current `origin/next`, the complete standalone
LuisaCompute build used 32 host threads and passed all 58 `unit_xir` tests:

```text
cmake --build build/luisa-tests --parallel 32
ctest --test-dir build/luisa-tests -L unit_xir --output-on-failure -j32
```

The incoming tile-to-kernel change also constructed a writable `std::span`
from a temporary `std::vector`, which is ill-formed with system STL and would
leave asynchronous copy input dangling. The separate `eb9b7cc79` fix gives the
buffer a scope that extends through synchronization. `test_tensor` then builds
and its backend-independent checks pass (19 assertions in 3 tests).

