# Noise code-generation controls: no missing 3D inline explanation

This read-only audit at Psycles `e79644af` (documentation HEAD `5c3746b2`) /
Luisa `da8fff856` does not reproduce Barbershop's roughly twofold surface
cost in isolated 3D Noise. No production source, inline policy, register cap
or math setting changes result from this experiment. It does not establish
that Noise is cost-free or that whole-scene code generation is equivalent.

The sole shader oracle is original Cycles
`cb168525138fecc792cc393f94afc39582b0103c`, included directly in HIP GPU
observers. [Results](results.json) retain final ISA function extents, resource
metadata, output comparisons, commands, logs and source hashes. There is no
CPU shader or host sampling implementation.

## Why these controls

The [typed scene census](../closure-input-guards/README.md) finds 138 authored
Noise nodes in the used Barbershop shaders: all are 3D FBM, with 135 normalized
and three unnormalized. These are static counts, not evaluation frequencies.
Current full-surface machine code already inlines 3D signed noise, fractals,
SVM and microfacet bodies. The remaining ordinary functions are 1D noise,
4D signed noise and OCML tangent. The declaration macro named
`PSYCLES_SVM_OUTLINE_NODE` is a host-only wrapper, not an inline/noinline marker.

The source audit found native Noise input-read order, distortion and color
guards, hash/gradient/interpolation and octave recurrence. A separate tangent
math lowering difference exists, but no Tangent Math node appears in these
used shaders. That is not a demonstrated cause of this scene's surface gap,
and no tangent intervention is made here.

## Bottom-up code and value checks

Both engines use transferred dynamic inputs, 256-thread groups and fast math.
Native observers use HIP O3 and `-parallel-jobs=32`; Luisa retains its ordinary
compiler decisions. All builds use 32 threads. Original saved `.bc` is
pre-optimization; only final ELF ISA is used for final instruction counts.

| Isolated operation | Cycles static instructions / bytes | Luisa static instructions / bytes |
| --- | ---: | ---: |
| Signed Noise 3D wrapper | 608 / 3,116 | 634 / 3,156 |
| FBM 3D, complete wrapper plus callee | 1,349 / 7,036 | 1,293 / 6,776 |

Original FBM has one 44-instruction wrapper and a 1,305-instruction callee;
Luisa has already inlined it. Neither leaf control has ISA scratch load/store
sites or VGPR/SGPR spill metadata. Signed-noise kernel VGPRs are 35 versus 37;
FBM kernel VGPRs are 52 versus 49. Original fixed private bytes are zero and
Luisa eight, without observed scratch instruction sites in these functions.
Code size and resources alone are not timings.

All 32 signed-noise outputs agree exactly. FBM has 31 exact outputs and one
7.4e-9 absolute difference, left alone. The initial complete dynamic Noise
node control checks 128 finite output floats (123 exact; maximum difference
5.9e-8), with Luisa's 12-word payload advance checked exactly. It uses the
actual production node handler, not a separately reimplemented algorithm.

## Synthetic complete-handler throughput

The throughput workload repeats those 32 typed input records across
1,048,576 invocations. Each writes float4 plus a PC-length word. Inputs mix
literal/stack roughness, zero/nonzero distortion and linked/unlinked color;
dimensions/type remain dynamic in the device node payload. Both engines run
18 dispatches at 256 threads/group. The first two are excluded from medians.
Original/Luisa/Luisa/original runs are sequential with no overlapping heavy
task launched by the agent. Only the first 32 output records are observed;
this is not a million-record image comparison or a scene benchmark.

| Run | Median after two warmups, ms | Observed dispatch range, ms |
| --- | ---: | ---: |
| Original | 1.231591 | 1.206248–1.467051 |
| Luisa | 1.310951 | 1.184930–1.466332 |
| Luisa repeat | 1.285271 | 1.184530–1.574893 |
| Original repeat | 1.226067 | 1.208526–1.610165 |

The median of Luisa's two run medians is about 5.6% higher. Dispatch ranges
overlap strongly; these four runs do not establish statistical significance.
They do not reproduce a twofold primitive-cost gap or measure Noise's share
of Barbershop. All observed outputs retain the same finite 2e-5
absolute-relative tolerance without a bit-matching path.

The benchmark's complete emitted function set contains 18,627 instructions /
94,840 bytes in original Cycles (23 functions), versus 24,587 / 124,080 in
Luisa (five functions). Thus Luisa emits about 32% more static code while
using fewer raw kernel VGPRs, 99 versus 162, and fewer fixed private bytes,
144 versus 176. Both have zero spill metadata. The profiler reports the
allocation-granularity VGPR counts 104 / 168; these must not be confused with
the raw ELF metadata counts. Static code size is not executed instruction
count. The earlier 32-invocation handler notes are deliberately not mixed
with these ring-input benchmark metrics.

An initial profile-parser attempt rejected rocprof's demangled original
kernel name. Its log and completed original dispatches remain in the evidence;
the four-run table uses the subsequent successful campaign. The parser now
accepts both original mangled and demangled prefixes. No expected shading
value was changed to address that harness failure.

## Reproduction and next boundary

Evidence directory: `/var/tmp/psycles-noise-codegen-bFjlyl`.
[Probe snapshots](probes/CMakeLists.txt) are byte-identical to the tested
sources, including their explicitly workstation-local checkout/library paths.
The result archiver verifies these snapshots against the frozen inputs.
Copy the eight probe files into a fresh temporary directory before configuring
or running them; do not generate dumps inside the repository. Use the stated
Psycles/Luisa revisions and the root build's system-STL ABI, not an unrelated
Luisa build. Configure/build there with 32 threads, run `noise` and `fbm`
observers from separate dump directories, then `run_throughput.py` from that
temporary probe directory after all builds finish.

```bash
python docs/validation/2026-09-09/noise-codegen/archive_results.py \
  /var/tmp/psycles-noise-codegen-bFjlyl \
  docs/validation/2026-09-09/noise-codegen/results.json
```

This audit prompted the independent [Voronoi octave review](../voronoi-octave/README.md),
which did find duplicated F1 control structure and an original-GPU state
mismatch. The large residual full-surface/DiffInd gaps, resource binding
identities and other code-generation differences remain open.
