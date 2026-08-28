# Surface parameter reuse-distance census

## Outcome

The exact hit-weighted census rejects indiscriminate per-hit parameter
materialization for the compact surface SVM. Barbershop performs 362,482,647
parameter operand reads in the measured 320x240, 16-spp run, but nearly all of
the apparent reusable work is the three-sample pattern introduced by Bump DAG
expansion. Those values remain live for roughly 57--65 bytecode instructions
on average. Retaining all of them would trade cached global reads for long
typed-local intervals, increasing the same VGPR/private-state pressure that is
already limiting `shade_surface`.

Only 8,591,018 reads belong to an exactly-two-use interval. They represent
2.37% of parameter reads, and even an ideal cache can eliminate only one of
the two reads: 4,295,509 reads, or 1.19% of the parameter-read total. This is
too small to explain the measured approximately 2x surface-kernel gap to
Cycles. The next representation experiment must therefore reduce common SVM
fetch, dispatch, and typed-address work instead of preloading every repeated
material value.

## Formal model

For one immutable bytecode segment `s`, parameter address `p`, and exact
surface-population count `H(t)` of topology `t`, define

```text
uses(s, p)  = number of operand occurrences of p in s
first(s, p) = first instruction containing p
last(s, p)  = last instruction containing p
span(s, p)  = last(s, p) - first(s, p) + 1.
```

The automatic-normal prefix and endpoint root are separate segments. Their
typed slots are colored independently and may overlap after the explicit
normal commit, so a parameter used on both sides contributes two intervals.
This is the implementable cache domain; counting it once across the commit
would assume a local value survives slot reuse without reserving storage.

For typed bank `b` and use-count bin `k`, the diagnostic reports

```text
unique_values       = sum H(t)
references          = sum H(t) * uses(s, p)
dynamic_references  = sum H(t) * dynamic_uses(s, p)
instruction_span    = sum H(t) * span(s, p)
```

over all `(t, s, p)` in that bank and bin. Bins 1--7 are exact; bin 8 is the
closed tail `uses >= 8`. A device invocation still records only its topology
tag. The host projects the exact counter through immutable bytecode, checks
every address/range/route, and uses overflow-checked 64-bit weighted sums.
Thus the diagnostic adds no per-instruction device atomic and fails closed by
setting `exact=false` on malformed input or arithmetic overflow.

Materializing a parameter once at its first use replaces `n` parameter loads
with one parameter load, one typed-local definition, and local reads through
the last use. Its optimistic load saving is `n - 1`, while its exact interval
cost is one bank-specific live slot over `span`. This model deliberately does
not equate a saved semantic load with a hardware transaction or claim that a
local array access is free.

## Barbershop evidence

The run used the Blender 5.2 export, compact populate-once surface execution,
HIP on the RX 9070 XT, 320x240, 16 fixed samples, and staged wavefront:

```sh
PSYCLES_COMPACT_SURFACE_VALUES=1 \
PSYCLES_POPULATE_SURFACE_ONCE=1 \
build/bin/psycles_render_blender_scene \
  /home/mike/Projects/psycles-benchmarks/barbershop-480p-64spp/export \
  /var/tmp/psycles-parameter-reuse-20260828.0V0dwJ/barbershop.exr \
  hip 320 240 16 16 - 0 0 0 0 16 - 1 0 wavefront-staged \
  32 32768 32 1 1 0 4 2 4096 0 0 0 1 1048576 \
  - /var/tmp/psycles-parameter-reuse-20260828.0V0dwJ/barbershop-histogram.json
```

The render completed in 0.413529 s. The histogram reported `exact=true`,
3,366,546 surface populations, and the same 380 programs / 8,870 value
instructions / 67 semantic variants as the production image.

| Uses per segment | Hit-weighted values | Reads | Optimistic saved reads |
| ---: | ---: | ---: | ---: |
| 1 | 143,547,563 | 143,547,563 | 0 |
| 2 | 4,295,509 | 8,591,018 | 4,295,509 |
| 3 | 69,825,614 | 209,476,842 | 139,651,228 |
| 4 | 201,066 | 804,264 | 603,198 |
| 5 | 12,592 | 62,960 | 50,368 |
| 6 or more | 0 | 0 | 0 |

The dominant three-use intervals have these bank-specific spans:

| Bank | Hit-weighted values | Weighted span | Mean span |
| --- | ---: | ---: | ---: |
| scalar | 43,673,668 | 2,777,349,414 | 63.59 instructions |
| float3 | 20,516,592 | 1,163,072,549 | 56.69 instructions |
| uint64 | 5,635,354 | 368,194,758 | 65.34 instructions |

This is the exact signature expected from the already-verified Bump expansion:
context-invariant parameters are value-numbered once while the height DAG is
evaluated at center, DX, and DY. The census does not assume that every
three-use value belongs to Bump; the decision needs only the measured fact
that the dominant candidates have long intervals.

The segment-local unique count is 217,882,344, larger than the existing
whole-transaction distinct-address count of 190,876,470 because a parameter
used on both sides of the normal commit must be materialized twice under the
current exact slot-reuse contract. Therefore the feasible segment-local upper
bound is 144,600,303 saved reads, not the earlier whole-transaction bound.

## Regression and decision

`psycles_luisa_sample_dispatch_film_tests` now exercises all four route cells,
a parameter used three times, and split sample dispatch. It requires:

- single and split requests to produce identical complete histograms;
- interval references to partition all parameter operand executions;
- interval dynamic references to equal the dynamic-parameter route cell;
- every nonempty interval bin to have a positive span; and
- the segment-local unique count to be no smaller than the whole-transaction
  distinct-address count.

The regression passed on fallback and HIP after a 32-thread build. It also
passed on Vulkan with `LUISA_VULKAN_USE_XIR=1`,
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and
`LUISA_VULKAN_DISABLE_DXC=1`, which makes native XIR-to-SPIR-V a fail-closed
requirement and forbids the DXC route. The renderer algorithm and output values
are unchanged, so no new numerical or visual-equivalence claim is introduced
by this observational change.

The rejected transformation is an all-parameter or all-repeated-parameter
cache. A later selective materialization experiment remains valid only if it
uses typed interval pressure as a hard constraint and demonstrates an actual
HIP improvement; it cannot cite repeated-reference count alone.
