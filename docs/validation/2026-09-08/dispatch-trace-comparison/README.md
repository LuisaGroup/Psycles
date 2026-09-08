# Scheduler trace: numerical versus structural equality

The longstanding fallback dispatch-test failure is in its **comparison
contract**, not a reason to change the renderer's floating-point path. The
test now distinguishes continuous intermediate values from exact state and
random samples. Film comparison and serial sample-chunking remain unchanged.

## Red reproduction and cause

The unmodified comparator fails in the complete fallback renderer fixture:

```text
wavefront dispatch changed path trace at slot 30, component 2:
expected -0.623230 (0xbf1f8bfd), got -0.623204 (0xbf1f8a50)
```

This is event 0 `light_ng.z`, not a random value or a path-state counter. An
opt-in, lossless trace capture observes all scheduler variants without
modifying their shaders. It records twelve traces: the baseline, ten same-scene
comparisons and the deliberately different zero-NEE fixture. Every exact state,
raw random component and written bit agrees in the ten comparisons. Serial
chunking and all three per-sample variants are entirely bit-identical to the
baseline trace. Coroutine variants have the same continuous differences;
only `light_ng.z` exceeds the old 2e-5 bound.

The source boundary is original Cycles 5.2.1 `kernel/light/point.h`, at the
same authoritative revision documented in the
[image sampler source notes](../bound-image-sampler/source-notes.md). Its
spherical point-light sampling computes the distance by the law of cosines,
then `Ng = normalize(P + D * t - center)` and remaps the point onto the sphere.
Psycles retains that operation structure. The radicand contains cancellation
between distance-squared terms; normalization then magnifies positional error
by approximately the inverse light radius (0.1 in this fixture).

Observed baseline versus wavefront evidence:

| Quantity | Difference |
| --- | --- |
| Surface position, light random sample, light PDF | Bit-identical |
| Sampled direction components | At most 5.96046448e-8 |
| Sampled distance | 3.57627869e-6 |
| Remapped position components | At most 2.50339508e-6 |
| Light normal components | At most 2.55703926e-5 |
| Normal lengths | 0.9999999837 and 1.0000000351 |

This is consistent with legal fast-math roundoff amplified by an ill-conditioned
intermediate, not evidence of lost coroutine state or a different sampled
path. No specific compiler pass is accused without a reduced compiler
counterexample. No renderer operation, precision mode, RNG or inlining policy
is changed to force the intermediate into the old numerical bound.

## Permanent test contract

- Identical serial accumulation across host request splitting remains
  **bit-exact for every film and trace lane**.
- Other schedulers retain the original **2e-5** scale-relative/absolute film
  bound. Sample counts and closure histograms retain their exact checks.
- Continuous intermediate trace values use a separate **1e-4**
  scale-relative/absolute bound. This is not a global renderer or external
  Cycles-oracle tolerance change.
- State, IDs, closure selection, raw RNG, reserved lanes and written markers
  now compare exact bits, even across schedulers. Non-finite values and invalid
  written markers are rejected. Same-renderer traversal identity is stricter
  than the cross-Cycles comparator's diagnostic/topology exceptions.
- The exact-component masks are generated from the existing Python trace
  schema, with explicit same-renderer topology/diagnostic semantics. No SVM
  words or trace ABI change. Regeneration consistency is tested.

The independent host regression performs **11,016 checks** over all 344 x 4
trace lanes. It verifies one-ULP perturbations against each lane's policy,
strict serial equality, out-of-bound continuous errors, NaN/Inf rejection,
written flags, invalid indices and the original captured failing pair. Moving
that same numeric pair into an RNG or state lane must still fail.

## Evidence and reproduction

Raw evidence root: /var/tmp/psycles-native-volume-svm-06XnDX.
`dispatch-trace-red-fallback.log` reproduces the old assertion with lossless
bit capture. `dispatch-trace-red-fallback.json` contains the full typed audit.
The committed [lossless sparse capture](red-trace-bits.json) retains all twelve
traces, including unwritten lanes, with uint32 hexadecimal bits. The
[audit parser](analyze_audit.py) and [archive reducer](archive_audit.py) reproduce
the analysis. The green full-suite capture has **identical bits in all twelve
traces**: only the comparator changes its verdict.
The capture switch is test-only:

```bash
PSYCLES_SAMPLE_DISPATCH_TRACE_AUDIT=1 \
  ./build/bin/psycles_luisa_sample_dispatch_film_tests fallback
```

All six executable/library hashes in the
[sampler canary record](../bound-image-sampler/canary-summary.json) remain
unchanged after this test-only fix and the all-target rebuild. The completed
four-scene canaries therefore still describe the renderer binaries; updating
this assertion cannot explain any render-performance change.

## Final validation

| Gate | Result | Evidence |
| --- | --- | --- |
| All-target build, all 32 threads | Passed | dispatch-semantic-full-build.log |
| Complete standalone dispatch renderer, HIP | Passed | dispatch-semantic-hip.log |
| Full registered HIP suite | **180/180**, 44.47 s | dispatch-semantic-full-hip.log |
| Full fallback suite, including dispatch film | **182/182**, 86.67 s | dispatch-semantic-full-fallback.log |
| Host comparator/schema/decoder gate | **4/4**, 0.29 s | dispatch-semantic-host.log |
| Full host suite | **156/157**, 1.80 s | dispatch-semantic-full-host.log |
| Strict native Vulkan image canary | **5/5**, 0.22 s | dispatch-semantic-native-vk.log |

The remaining host failure is only the existing four source-size violations;
this change neither relaxes nor skips that guard. Native Vulkan emits SPIR-V
under all three strict-XIR/no-DXC guards, with no DXC/DXIL library loaded.
The Python cross-Cycles comparator's numerical bounds remain unchanged.
