# Matched-pass report audit notes

Audience: technical. Delivery: the user's requested repository Markdown,
not an additional report app, HTML copy or external site. The build-report
technical structure maps to summary, render findings, compilation/frame
costs, pass quality, scope/method and next questions. Definitions precede
dependent tables; limitations sit beside the affected measurements.

The three neutral tables serve exact row lookup across four fixed scenes:
render time/range, initialization/frame dimensions and pass error. Unlike
units are not combined into a score. Three repeats are insufficient for a
trend/distribution plot. Existing rendered triptychs provide image evidence;
the four first-pair Combined images were viewed at resized resolution.
No decorative chart or color-coded performance verdict is needed.

Evidence root: /var/tmp/psycles-matched-pass-hip-2c98Q0.
The preserved failed initial campaign is /var/tmp/psycles-matched-pass-hip-48QkmP;
it has no completed pair and contributes no summary observation.
run_campaign.py is the exact executed orchestration script, including
its original paths and no-overwrite assertions. For a new run, use a fresh
output path; do not overwrite the archived evidence. analyze_campaign.py
contains the executed validation and aggregation, also retaining source paths.

campaign.json records all 12 manifests, command/log hashes, static frame
layouts and frozen implementation hashes. The per-scene run directories
archive original benchmark.json, cycles-metadata.json and all-pass report.json
contents. The actual EXRs and triptychs remain at the reported local paths;
their hashes and image labels are not rewritten to suggest bundled assets.

After collection, all 24 renderer records passed the schema-v3 resume
validator against actual EXR channel headers, output hashes, stored timings
and original Cycles metadata/log evidence. Each pair had already verified
source blend/export identities and frozen ten executable/library/script
hashes. The script independently read all 46 actual channels and checked
finiteness after every pair. All 12 reports contain exactly 15 passes, zero
actual-invalid pixels and the explicitly recorded original Classroom
non-finite values. No invalid-count tolerance or image-error threshold changed.

summary.json is calculated from these saved records with median/min/max and
all three observations. Relative time is the ratio of medians; individual
paired ratios are also retained, not silently substituted. Visible tables
were checked against those summaries. Renderer order is fixed Cycles first;
only scene order rotates. Single-workstation variation and cache scope are
limitations, not grounds for claiming speedup or completion.
