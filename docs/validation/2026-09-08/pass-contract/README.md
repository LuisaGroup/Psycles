# Exact common-pass benchmark workload

The runner now requires equal output work, not merely a common subset of
channels to compare. Schema v3 replaces the v2 benchmark protocol; the
renderer itself remains Psycles 3a5cc005 / Luisa 9ea3b720f at this checkpoint.
The first original-scene HIP pair validates the complete new protocol.
The four-scene, three-repeat campaign is still running and is not declared
complete here.

## Cause and correction

The previous Cycles setup enabled its requested passes without disabling
authored extras. Actual original-scene EXRs contained Depth/sample count
plus Mist in Monk, Depth/sample count in Monster, AO/Object Index in
Classroom, and AO in Barbershop. Cycles' AO output feature launches additional
shadow paths; this is not merely a file-format difference and is not the
material Ambient Occlusion node or authored Fast GI lighting.

The headless golden setup now replaces output toggles with exactly 15 common
passes. It clears auxiliary, debug, denoising, custom-AOV and light-group
outputs, without saving the source .blend or enabling disabled view layers.
Non-pass quality options are preserved. The runner reads every EXR subimage
header, requires one layer and 46 unique channels (Combined RGBA, Normal XYZ,
and the remaining RGB passes), and records the actual inventory. Dimensions
must match the requested render. Extra, missing, duplicate or mixed-layer
channels fail before a renderer record can be used for a speed comparison.

The comparison and inventory validator share the existing semantic alias
registry, covering Blender's long labels and Psycles' short labels such as
VolumeDir. The actual names remain in the manifest. Both golden metadata and
the benchmark record declare the pass-contract version. Resume verifies the
real headers again in addition to source/output/log/metadata hashes and timing
records. Schema v1/v2 results cannot be reused as v3.

## Permanent regressions and live validation

- A real Blender 5.2.1 RNA fixture enables inherited AO, Mist, Depth, object
  indices, motion vectors, cryptomatte, debug timing/sample count, denoising
  outputs, an AOV and a light group. The unchanged script fails its exact-work
  assertion; the corrected script passes and preserves the disabled layer
  and unrelated quality option.
- Five header-contract tests cover canonical names, actual Blender 5.2 long
  labels, Psycles aliases, all four archived extra-workload shapes, and
  missing/duplicate/mixed-layer data. The 22 runner tests include rejection
  of v1/v2 resume and missing pass inventory. Real v3 Cycles and Psycles
  render records were independently revalidated through the resume checker.
- Full 32-thread build passed. The host suite was 155/156, with only the four
  existing source-size violations; the final six focused host tests passed
  6/6, including Blender setup, comparison and path-trace compatibility.

An initial actual-image validation attempt exposed that the new validator
recognized only short color-pass labels. It was stopped, not declared a
completed benchmark. Sharing the existing complete comparison alias registry
fixes that naming issue; a permanent real-label test covers it. The original
failed attempt remains at `/var/tmp/psycles-matched-pass-hip-48QkmP`.
The fresh campaign is `/var/tmp/psycles-matched-pass-hip-2c98Q0`.

## First matched pair, not a repeated performance conclusion

Original Barbershop, 2048x858 / 256 fixed samples, HIP RX 9070 XT, animated
seed semantics preserved (effective seed 1267069554). Both EXRs contain
exactly 46 channels. Main Psycles shader caching is disabled; auxiliary/OS
caches retain normal policy, native fast math is enabled, and no profiler
or concurrent build/render overlaps the pair.

| Metric | Cycles HIP | Psycles HIP |
| --- | ---: | ---: |
| Main-loop / render-only wall seconds | 25.3208 | 39.1701 |
| Cycles enclosing render call / Psycles session init seconds | 35.9077 | 25.5929 |

The second row contains different scopes and is not a speed comparison.
Psycles' single render observation is 54.7% slower than Cycles. It does not
meet the efficiency goal. Frame size remains 416 B. The earlier unequal-pass
ratios must not be substituted for this new workload. Repeated measurements
and the other scenes are required before replacing the full campaign table.

Exact commands, raw artifact paths, timing boundaries, actual pass inventories
and hashes are in [first-barbershop.json](first-barbershop.json), with original
build/device/seed metadata in [first-cycles-metadata.json](first-cycles-metadata.json).
Logs for the red/green setup, header tests and real resume checks use the
pass-contract or golden-pass prefixes under
`/var/tmp/psycles-native-volume-svm-06XnDX`.

Technical-report QA keeps protocol correction, first-pair observations,
kernel diagnosis and the still-running campaign separate. In particular,
the [volume-work result](../volume-work/README.md) demonstrates a structural
fix but no measurable Barbershop volume-kernel speedup; surface remains the
larger unresolved kernel difference.
