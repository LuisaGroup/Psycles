# Retained-selection projection experiment

## Outcome

The candidate was rejected and fully reverted. Proving that an entry was
already retained allowed the physical evaluator to omit its repeated
`allocation_weight >= CLOSURE_WEIGHT_CUTOFF` checks, and the generated HIP
program became slightly smaller. However, two normalized Barbershop profiles
made `shade_surface` 0.370% slower than the two-run baseline mean. The change
does not address the structural complex-material cost isolated by the Cycles
5.2 surface probes.

## Formal model

Let `R` be the initialized prefix of `SurfaceClosureSet`. Its append operation
publishes a new count only under

```text
kind != none
and allocation_weight >= closure_weight_cutoff
and count < capacity.
```

Consequently every indexed read `r in R` satisfies the allocation predicate,
so substituting the cutoff for `r.allocation_weight` is semantics-preserving
for categorical selection. The proof does not establish `setup_valid`: Cycles
may allocate a closure slot and let setup change its type to `NONE`. The
candidate therefore retained the setup predicate.

This local equivalence was implemented with a proof-carrying retained-input
type and covered by fallback and HIP physical-closure and collection tests.
The implementation was reverted only because the generated program and
profile showed no performance benefit, not because the equivalence failed.

## Generated HIP structure

The cold one-sample Barbershop dump confirmed that the three physical
selection loops no longer loaded `allocation_weight` or compared it against
the cutoff. All other fields and predicates remained unchanged.

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Final LLVM IR | 53,903 lines / 3,001,514 B | 53,891 lines / 3,000,603 B | -12 lines / -911 B |
| Surface object | 339,800 B | 339,544 B | -256 B |
| Main entry | 318,104 B | 317,872 B | -232 B |
| Private storage | 3,096 B | 3,096 B | unchanged |
| VGPR | 256 | 256 | unchanged |

The baseline dump is
`/var/tmp/psycles-typed-region-b0.vSqMha`; the candidate dump is
`/var/tmp/psycles-retained-selection.obYqKE`.

## Normalized HIP profile

Both builds rendered the official Blender 5.2 Barbershop export at 640x480,
64 fixed samples, staged wavefront scheduling, and a 64-thread block on the
Radeon RX 9070 XT. Time is normalized by the actual launched work rather than
dispatch count.

| Build / run | Calls | Work-items | Total ms | ns/work-item |
| --- | ---: | ---: | ---: | ---: |
| Baseline 1 | 366 | 53,601,728 | 1,154.071 | 21.530472133 |
| Baseline 2 | 357 | 53,557,504 | 1,158.068 | 21.622885000 |
| Candidate 1 | 357 | 53,608,064 | 1,159.107 | 21.621871422 |
| Candidate 2 | 359 | 53,573,184 | 1,162.053 | 21.690954321 |
| Baseline mean | | | | 21.576678567 |
| Candidate mean | | | | 21.656412872 |
| Candidate change | | | | **+0.370%** |

Candidate traces are stored in
`/var/tmp/psycles-retained-selection-profile1.YinsYg` and
`/var/tmp/psycles-retained-selection-profile2.jS1OiB`.

## Consequence

Retention-time facts alone remove only one scalar load and predicate from each
selection pass. Cycles instead stores the post-setup `ClosureType` directly;
its hot categorical loops inspect only `type` and `sample_weight`. The next
surface change must adopt that representation and compact SVM execution shape,
rather than accumulate more projections around Psycles' wider
`kind/lobe/setup_valid/allocation_weight` header.
