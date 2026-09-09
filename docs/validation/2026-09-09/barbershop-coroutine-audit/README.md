# Barbershop: coroutine frame, cuts and shader sorting

The frame is sizable, but its size alone does not explain Barbershop's
remaining cost. Surface shading is cut before geometry/SVM population and
does use shader sorting. This frozen audit reproduced a scheduling difference:
multiple incoming annotation queues split one logical surface queue for greedy
selection. Its first isolated correction and exact old frame evidence are
retained below; they are not the current integration status.

The [subsequent SDK 98f integration](../next-integration/README.md) now includes
both logical-priority aggregation and explicit compatible Handler batching.
All incoming surface routes join one physical sort/resume queue. Across the
complete fresh 64-spp diagnostic, surface dispatches fall from 1,062 to 798
while entries change only from 332,307,665 to 332,307,666. The 416 B frame
summary is unchanged; no reduction of unnecessary shading or frame traffic
is proved. Fresh 256-spp Barbershop/Monk canaries are finite in all 46 channels,
and host 183/183, HIP 192/192, fallback 194/194, root native Vulkan 12/12 and
SDK native Vulkan 17/17 pass. A new four-scene paired benchmark, residual
DiffInd alignment and Classroom finite-value repair remain outstanding.

The frozen audit uses Psycles `f3f852aedbb4` plus the then-uncommitted native Holdout
changes, with published Luisa `6e58928d8460`. The original source is Cycles
5.2.1, `/home/mike/Projects/blender-cycles-trace-5.2` at `cb168525138f`.
The unchanged production six-canary campaign reports Barbershop median
37.8379 s versus retained Cycles 25.3775 s, not fresh timing pairs. Its
[Holdout checkpoint](../native-holdout/README.md) retains the Classroom
finite-value failure and its historical failed native Vulkan compiler gate;
the later green gate does not overwrite that evidence.

## What the frozen SDK-6e frame actually contains

Two fresh diagnostics use the same frozen renderer: a full 2048x858/64-spp
queue-statistics capture and a separate 1-spp finalized-graph observer. Their
92 physical field names, types, offsets, sizes and alignments match exactly.
Neither diagnostic is promoted to a performance measurement.

| Frame quantity | Frozen value |
| --- | ---: |
| Logical values / physical user fields | 93 / 85 |
| Reserved uint fields | 7 |
| Physical fields in total | 92 |
| Physical user payload | 384 B |
| AoS structure size / alignment | 416 B / 16 B |
| Runtime SoA stride / pool capacity | 412 B / 1,048,576 |
| Main frame pool | 432,013,312 B = 412 MiB |

One user field is a 48-byte `array<uint4,3>`; the other 91 physical fields,
including the reserved fields, are four-byte scalars. AoS has four padding
bytes before the array. The actual staged scheduler uses runtime SoA, which
orders by alignment and has no corresponding per-frame gap. Thus the pool
is `412 * capacity`, not `416 * capacity`. This is an exact reconstruction
from the captured types and `CoroFrameStorageLayout::make_runtime_soa`, not
a GPU allocation-counter measurement.

Main-frame storage excludes the separate direct-light/shadow pools, sorting
scratch, SVM local stack, closure storage and backend spills. It is not
meaningful to compare this number with all Cycles state or with one Cycles
stage's register footprint without matching those scopes.

### Finalized frame-I/O contract, not whole-frame transfers

The observer calls the original `CoroGraph::from_module` and prints its return
value without changing it. `coro_compile.cpp:1115-1122` constructs that graph
after continuation normalization. Its field sets are distinct from earlier
distillation `scope_external` lists; those earlier lists undercount final
input fields in four stages and are retained separately in the evidence.

| Stage | Final input fields / bytes | Token-indexed relocation fields / bytes |
| --- | ---: | ---: |
| entry | 6 / 24 | 0 / 0 |
| intersect_closest | 35 / 140 | 67 / 312 |
| shade_volume | 68 / 316 | 86 / 388 |
| shade_light_forward | 47 / 188 | 82 / 372 |
| shade_background | 25 / 100 | 74 / 340 |
| shade_surface | 56 / 268 | 73 / 336 |

Relocation includes target-token field 6 and dormant pass-through state; it
is not the ordinary resume load set. Stores depend on the outgoing edge:
closest-to-surface has 10 fields / 40 B, volume-to-surface 54 / 260 B, and
surface-to-closest 46 / 228 B, including the token. Normal return stores only
the four-byte token. The scheduler does not unconditionally transfer the
whole 416-byte AoS at every cut. These are exact static transport sets, not
measured memory transactions or spill traffic.

Six floats, `_reg_1167.{0,1,2}` and `_reg_1196.{0,1,2}`, remain in finalized
closest inputs although their distilled edge-store lists are empty. Their
source/XIR definedness and consumer reachability still need proof. This
observation does not authorize removing them or adding clears.

## Cuts: surface placement is appropriate, not complete stage parity

The six main subroutines are entry, closest intersection, volume shading,
forward-light shading, background shading and surface shading. Three separate
auxiliary stages handle surface NEE light shading, shadow intersection and
shadow shading; they are not included in the six main subroutines.

The surface cut in `path_kernel_pipeline.cpp:208-237` occurs before geometry
and SVM closure population. Populated closures, ShaderData and SVM local
storage stay inside the surface stage. Moving a cut into this work would
make large short-lived state persistent and would not reproduce native
SHADE_SURFACE. No such split is proposed.

The principal closest/volume/endpoint edges match the corresponding Cycles
route: only a nonempty volume stack reaches volume shading; attenuation
continues to the selected endpoint; scattering returns to closest. Forward
lamp traversal also returns to closest, superseding the older boundary
report's claim that it reuses the mesh hit without requeueing.

There are still concrete ownership/control-flow gaps:

- Forward-lamp position, normal, UV, PDF and evaluation factor are computed
  before the light cut and can survive an intervening volume cut. Native
  light evaluation reconstructs them inside SHADE_LIGHT_FORWARD. This is
  40 logical bytes, **not a proved 40-byte physical frame saving**.
- Volume NEE still traces its surface/volume shadow inline. Native volume
  NEE forks the independent shadow path, while the current side queue has
  only the surface producer. Three auxiliary names do not certify this fork.
- Camera volume-stack initialization is inline, and not every optional native
  queue/domain is represented. Inlining alone is not treated as a defect.

## Sorting is active; logical queue aggregation differs

The original key is shader identity plus a frame-index locality partition.
Psycles uses the same expression and the same native policy: below 300 shader
keys, partition by approximately 65,536 frames; otherwise one partition.
Barbershop has 490 keys, so partition size 1,048,576 and one partition match
Cycles. This is not missing shader sorting.

The 64-spp capture contains seven scheduler dispatch ranges: 1, 1, 2, 4, 8,
16 and 32 spp. Their generated count sums to **112,459,776**, exactly
`2048 * 858 * 64`. Reading only the last table would omit half the samples.

| Incoming surface route | Handler dispatches | Frame entries |
| --- | ---: | ---: |
| closest -> surface, queue 7 / boundary 2 | 710 | 271,308,737 |
| volume -> surface, queue 8 / boundary 10 | 352 | 60,998,928 |
| Other four registered incoming boundaries | 0 | 0 |
| Total / surface continuation | 1,062 | 332,307,665 |

Equality holds in every batch and in the total: all observed surface resumes
pass through the registered sorting handlers. Handler dispatches are not
individual sort-kernel launch counts. The final graph proves both active
boundaries have identical typed uint key binding, logical value 84 / physical
field 88, and identical complete extension contracts. `use` and `reconstruct`
are only `[88]`; key extraction does not load the whole frame.

Nevertheless the frozen generic scheduler greedily compares each physical
incoming queue separately. Cycles compares one aggregate counter per target
kernel. Minimal dynamic-input counterexample: target-A=4, target-B=4,
rival=7. Native logical selection chooses target=8; the frozen scheduler
chooses rival=7. This scheduling error is reproduced, not inferred solely
from Barbershop timings.

### Historical first correction: logical priority only, isolated HIP validation

The isolated SDK correction sums physical queues by their logical target for
greedy selection and auxiliary admission. It snapshots all selected member
queues before any member resumes, preventing newly enqueued paths from
entering an already selected snapshot. It retains each boundary's handler,
typed binding, permutation, counts and actual resume launch.

| Regression observation | Result |
| --- | --- |
| Original logical-cardinality witness | 28 failed assertions, process exit 255 |
| Original boundary-local binding control | 68 assertions pass, filtered control, exit 0 |
| Expanded complete resume-annotation HIP binary | 7 tests / 6,442 assertions pass |
| Expanded complete auxiliary-admission HIP binary | 2 tests / 171 assertions pass |

The complete HIP controls include self-rescheduling snapshots, mixed bare and
annotated entries, AoS/SoA, queue compaction/count modes, and aggregate side-pool
admission. They use `LUISA_CORO_WAVEFRONT_VERIFY_QUEUES=1` and full-thread builds.
The tested isolated `wavefront.h` SHA is
`58bd439e1f36307d994e1b18c17e5cc49db93417ca9d9e83925a24c9c0dea394`.

This first correction was **not Handler batching/coalescing**: the two surface subsets still
sort and resume separately. A later shared batch needs explicit handler
permission plus compatible typed projections and resident-state certificates;
schema/name equality alone cannot authorize it. At that isolated checkpoint,
batching was design/default-disabled API work. The subsequent integration
linked above completes the explicit batching implementation and fresh backend
gates; it does not turn these older logical-priority-only observations into
joint-batching measurements or establish a queue-only rendering speedup.

## Volume-state excess and a separate correctness finding

The current source VolumeStack stores object/shader plus legacy surface tag,
parameter block, instance index, cached sampling method and count. At three
slots its source storage is 76 B; native object/shader arrays use 24 B.
The 52 B difference is a representation lead, not an immediately removable
frame amount. Legacy tag/block fields have no native SVM consumer, but instance
indices still address the current majorant ranges, sampling methods feed NEE,
and count controls current stack APIs. Those owners must be migrated before
their storage can be removed.

An independent source mismatch was also found: `VolumeStack::_exit` swaps the
last entry into a removed slot, while native Cycles shifts the remaining
entries left. For `[A,B,C]`, exit A yields `[C,B]` versus native `[B,C]`.
The current test even documents swap-last behavior; it needs an original-GPU
regression. Barbershop's three-slot stack permits only two active entries,
so this mismatch is not demonstrated as its performance/DiffInd cause.

## Retained evidence and reproduction

Evidence root: `/var/tmp/psycles-holdout-CXlJR7`.

- `barber-frame-stats-audit.md`, `parse_barber_frame.py`: all 93 logical values,
  92 physical fields, 17 edges and seven complete counter groups.
- `parse_barber_graph.py`, `frame-graph-observer/capture.cpp`: finalized graph,
  exact field transport/binding validation; observer delegates to the original.
- `barber-stage-audit.md`, `barber-sort-audit.md`, `barber-logical-queue-plan.md`,
  `barber-volume-frame-audit.md`: native source mapping, failure proof and limits.
- `barber-queue-cardinality-red-hip.log`, `barber-queue-binding-control-hip.log`,
  `barber-queue-expanded-hip.log`, `barber-queue-aux-expanded-hip.log`: retained
  original failure, anti-overmerging control and isolated first-step results.

```bash
python -B /var/tmp/psycles-holdout-CXlJR7/parse_barber_frame.py \
  /var/tmp/psycles-holdout-CXlJR7/barber-frame-OFalxg/barbershop.log
python -B /var/tmp/psycles-holdout-CXlJR7/parse_barber_graph.py \
  /var/tmp/psycles-holdout-CXlJR7/barber-frame-OFalxg/graph-probe.log \
  /var/tmp/psycles-holdout-CXlJR7/barber-frame-OFalxg/barbershop.log
```

Capture SHA-256:

```text
fbb5473ff7baeee8a1cc9ceee54e4dee970ff4898f0630a8e81b91021d06b4e6  barbershop.log
f0fb45eb76d88cbd4388c14d7232d569f04bb49bd3c8180d63f789fe65f76fc7  graph-probe.log
84b72a6d67148266f3020fed5a6cff53475f041854df128c7c6af6e59773ab6a  barber-queue-expanded-hip.log
a331b5029079c6502f6ff1d4678e5db5ee683dc7df4f5af4d8cb16f0f9dce1a6  barber-queue-aux-expanded-hip.log
```

No part of this audit establishes that frame traffic or sorting accounts for
the whole Barbershop gap. The remaining work is native structural repair with
original regressions and full-scene verification, not a frame-size-only target.
