# Lone Monk coroutine scheduler equivalence

This checkpoint validates that Psycles' one host-side path program has the
same rendering semantics when lowered as a direct megakernel, a compacting
wavefront coroutine, or a persistent-worker coroutine. It is a scheduler
equivalence check, not a new Cycles differential. All three runs use the same
37-material Lone Monk export, Luisa fallback, 640x480, fixed sample 0, and one
sample per pixel. The follow-up backend and coroutine checks are current through
LuisaCompute `next@f4043aee7`.

The renderer CLI now accepts `-` for the optional sample-chunk JSON argument,
so selecting argument 17's scheduler does not force the render into pixel-probe
mode. The common command shape was:

```text
psycles_render_blender_scene <bundle> <output.ppm> fallback \
    640 480 1 1 - 320 240 0 0 1 - 1 0 \
    <megakernel|wavefront|persistent>
```

## Numerical result

The three PPM files are byte-identical. The three linear Combined, Normal, and
Albedo PFM files are also byte-identical:

| Output | SHA-256 shared by all schedulers |
| --- | --- |
| display PPM | `1e1ddc5f7e8bfe6f8976d411f553e0d9166fa224a9e8df94809e123b6f168e4a` |
| Combined PFM | `50da62d8c86e0092b33f54f684ea26f9aa621c0ee2f34deb64e4c70f8f2f4767` |
| Normal PFM | `3c407f436a9233f404c4a56e72b54bbfb788c89bf46f8ea137642c2d95b5a173` |
| Albedo PFM | `8b573e40bfd95438d756d2594ae7297c7cf792dc6bb02d26242163e3e96f4352` |

The multilayer EXR container hashes differ because OpenEXR records a distinct
`capDate` for each process. `idiff -a -fail 0 -failpercent 0` reports `PASS`
for megakernel versus wavefront and megakernel versus persistent, across all
EXR channels.

## Timing observation

These are single-process wall observations, not a final benchmark. Scene import
is included separately from session creation/JIT, and render time covers exactly
one 640x480 sample.

| Fallback scheduler | Import (s) | Session/JIT (s) | Render (s) | Wall (s) | Peak RSS (KiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| megakernel | 1.602 | 0.091 | 0.160 | 2.20 | 2,409,088 |
| wavefront | 1.592 | 48.178 | 0.435 | 50.55 | 2,804,260 |
| persistent | 1.581 | 14.585 | 0.221 | 16.73 | 2,434,912 |

The wavefront frame capacity is 307,200 at this resolution; the persistent
configuration remains 32,768 workers, block size 128, fetch size 16, shared
SoA, and global-memory extension. The 48.178-second wavefront session time is
not accepted as an inherent image-size cost: the same scene at 16x16 used 256
frames and created its session in 14.506 seconds. The capacity-dependent
generation/cache/compilation path was subsequently removed from Luisa's
wavefront shader identity and is retained above as the pre-fix baseline.

## Runtime-capacity and backend follow-up

Luisa `next@9b3b9ba87` changed the wavefront frame storage to a runtime-capacity
SoA layout. Field offsets now obey one linear ABI formula and the pool capacity
is a dispatch argument rather than shader-AST state. A cold 16x16 wavefront run
with capacity 256 was followed immediately by a 640x480 run with capacity
307,200. The second run reused the same capacity-independent shader cache:

| Backend / scheduler | Resolution | Session/JIT (s) | Render (s) | Wall (s) |
| --- | ---: | ---: | ---: | ---: |
| fallback / wavefront, cold | 16x16 | 48.672 | 0.021 | 50.55 |
| fallback / wavefront, cache reuse | 640x480 | 14.781 | 0.446 | 17.18 |
| fallback / megakernel | 640x480 | 0.089 | 0.156 | 2.19 |
| HIP / wavefront, cold | 640x480 | 42.847 | 0.187 | 48.12 |

The 640x480 fallback wavefront and megakernel outputs remain byte-identical for
the display PPM and all three PFM passes listed above; exact-channel OpenEXR
comparison also passes. HIP completed the same real-scene wavefront schedule
after Luisa `next@d1e0ad6a8` fixed workgroup-memory ordering at HIP block
barriers.

Strict Vulkan (`LUISA_VULKAN_USE_XIR=1` and
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`) originally failed while
restructuring the large surface continuation: a three-phase cycle grew the CFG
from 44 to 423 blocks and exhausted the 64-round safety budget. The root was a
generated selector whose distinct empty proxies both ended in the same
`Break(M)` effect. Luisa `next@cc3a23f44` now compares the exact terminal
effect `(Branch|Break|Continue, target)` after quotienting empty forwarding
chains. The continuation reaches a fixed point after two rounds at 67 blocks
and 7,198 instructions, and native SPIR-V is emitted without loading DXC.

The subsequent RADV pipeline compilation exposed two independent problems.
First, sub-word coroutine-frame stores had lost their base-buffer alignment and
were conservatively emitted as masked compare/exchange loops. Luisa
`next@a56b9f8ca` now composes XIR integer-offset alignment with the captured
ByteBuffer binding offset. Proven four-byte fields use direct word stores;
genuinely one- and two-byte fields retain the masked update. This reduced the
then-largest physical shader from roughly 921 KiB of disassembly with hundreds
of atomic compare/exchanges to 266,488 bytes with five required compare/exchanges.
It did not fix the remaining loop-carried SSA topology: the process still ran
for more than two minutes and reached approximately 111.3 GiB resident memory.

## Vulkan sample-cycle cutpoint

ABI inspection identified that shader as the wavefront generate callable, not
the surface continuation. Its only non-atomic loop was the outer
`sample_offset < samples` loop. The former first suspension was inside
`PathKernelPipeline::emit`, so the generate callable kept the complete path
frame live across a physical sample loop. The sample loop is itself a canonical
scheduler cycle; the coroutine path now suspends at its header, before
`begin_path_sample`. Entry initializes the pixel invocation, each activation
processes one sample, and the loop back-edge targets the sample continuation.
The host-side `is_coro` branch means the direct megakernel receives no suspend
statement or coroutine control flow.

This cut adds one reachable continuation but does not enlarge the frame:

| Property | Before | After |
| --- | ---: | ---: |
| reachable subroutines | 3 | 4 |
| frame fields | 297 | 297 |
| frame bytes | 1,424 | 1,424 |

A strict native Vulkan compile used both
`LUISA_VULKAN_USE_XIR=1` and
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`; DXC was not loaded. The physical
path shaders after the cut were:

| Continuation / utility | Normalized XIR (bytes) | SPIR-V disassembly (bytes) | SPIR-V words |
| --- | ---: | ---: | ---: |
| generate | 77,973 | 94,195 | about 8,625 |
| `path_sample` | 171,669 | 183,434 | about 16,566 |
| `path_bounce` | 358,501 | 436,102 | about 39,354 |
| `surface_shading` | 6,063,587 | 5,643,344 | 491,348 |
| wavefront compact | 150,428 | 345,456 | 30,675 |

The source-dump diagnostic crossed the formerly unbounded RADV stage and then
was stopped after 88.74 seconds while compiling later utility shaders; it was
not an out-of-memory completion. Peak resident memory was 6,687,164 KiB. A
subsequent 16x16, one-sample Lone Monk render completed successfully:

| Stage | Result |
| --- | ---: |
| scene import (348 geometries, 87,534 instances, 37 materials) | 1.922 s |
| session / shader JIT | 31.828 s |
| render | 0.0637 s |
| process wall time | 34.47 s |
| peak resident memory | 1,387,636 KiB |

This is a compile-and-execute smoke test, not a quality comparison: 16x16 at
one sample is too coarse for a Cycles differential. It establishes that strict
XIR-to-SPIR-V wavefront rendering now terminates with bounded memory. The new
dominant problem is also explicit: `surface_shading` begins as 134,071,623
bytes of raw XIR and still emits a 491,348-word module. Material population and
closure evaluate/sample expansion must be decomposed or shared before Vulkan
JIT time is acceptable.

## Optimized-away suspend tokens

Coroutine lowering treats front-end suspend tokens as candidate identities,
not a dense callable index set. If `T_front` is the front-end token set and
`T_live` is the subset remaining after algebraic simplification, CFG
simplification, and dead-code elimination, graph construction and callable
materialization operate only on `T_live`. Therefore:

- a token in `T_front - T_live` has no graph node and no lowered callable;
- surviving tokens retain their original values and may be sparse;
- schedulers map sparse external tokens to dense internal queues explicitly;
- if `T_live` is empty, the coroutine still has exactly the token-zero entry
  callable.

Coroutine ownership is independent of both sets. AST-to-XIR now returns the
exact XIR function associated with the root AST builder, and the lowering
pipeline carries that provenance across optimization. It no longer searches
the translated module for a callable that happens to contain a suspend. This
also defines the stronger degenerate case `T_front = empty`: the root remains
a valid entry-only coroutine even though no suspend marker has ever existed.

The regression places an auto-numbered suspend in constant-dead control flow
before the live sample-loop-header and nested bounce-loop suspends. It verifies
that dead token 1 has no node/callable, live tokens 2 and 3 are not renumbered,
both loop back-edges resume correctly, and four samples produce the exact
accumulation under state-machine, wavefront, and persistent scheduling. The
same matrix also executes the zero-front-end-token case. It reports 61 passing
assertions in 12 tests on fallback, HIP, and strict native Vulkan
XIR-to-SPIR-V.

## Coroutine CFG pass domain

The first strict-Vulkan profile showed a second, independent source of compile
cost before physical shader compilation. AST-to-XIR places the source
coroutine `C`, its ordinary shader-graph dependencies `H`, and later generated
continuations `K` in one module. The old coroutine pipeline applied
destructure, scalar cleanup, reg2mem, and restructure to the whole module. That
made the coroutine front end process `H` even though no helper participates in
the state machine.

The lowering domain is now explicit:

- pre-distill CFG passes operate on `{C}`;
- split/materialization derives `K` from the optimized live scopes of `C`;
- post-materialization CFG passes operate on `{K}`;
- ordinary dependencies `H` retain their structured XIR definitions.

This is also the required relation for an optimized-away suspend. Let
`T_front` be the candidate tokens assigned while recording `C`, and let
`T_live` be the tokens whose suspend instructions remain reachable after CFG
optimization. Lowering obeys

`T_live subset-or-equal T_front`, and
`K = {entry} union {continuation(t) | t in T_live}`.

A token in `T_front - T_live` therefore has neither a graph node nor a lowered
callable. Surviving tokens keep their original values and may be sparse; each
scheduler maps that external token identity onto its own dense queue/storage
index. The degenerate `T_live = empty` case still materializes exactly the
token-zero entry callable. No lookup into `H`, fabricated dead continuation,
or dense token renumbering is involved.

`LUISA_CORO_VERIFY_PASS_DOMAIN=1` enables a structural oracle that snapshots
the body block and ordered block/instruction identities of every function in
`H`, then checks them after each coroutine phase. The regression contains a
structured switch/loop helper and proves that the new path preserves it. A
diagnostic rebuild with the former whole-module normalization fails exactly at
`continuation normalization`, so the test distinguishes the old and new pass
domains rather than merely checking rendered values.

On the unchanged 37-material Lone Monk wavefront compile, the source-coroutine
interval and its formerly module-wide pre-distill pass changed as follows:

| Host compile boundary | Before | After | Change |
| --- | ---: | ---: | ---: |
| pre-distill pass pipeline | 171.85 ms | 5.37 ms | -96.9% |
| source coroutine to four lowered continuations | about 14.60 s | about 7.26 s | about 2.0x |

The remaining roughly seven seconds are no longer hidden inside material
helper CFG normalization. They are in coroutine split/materialization,
continuation normalization, final verification, and XIR-to-AST handoff, which
can now be profiled independently on a much smaller domain.

A cache-disabled strict native Vulkan run completed at 1x1/1 spp. Its physical
path shaders contained 8,625, 16,566, 39,051, and 488,499 SPIR-V words for
generate, `path_sample`, `path_bounce`, and `surface_shading`. Shader JIT was
73.839 s, but 51.547 s of that was a genuinely cold RADV pipeline creation for
`surface_shading`; it is not coroutine-front-end time. Peak RSS was 6,668,204
KiB during that driver compile. The wavefront and separately compiled
megakernel PPM plus Combined/Normal/Albedo PFM files are byte-identical; their
shared hashes are:

| Output | SHA-256 |
| --- | --- |
| display PPM | `3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b` |
| Combined PFM | `4f93bceff43a46a086454e9a50745a497b39cd27e8a694f617a5c934fe3ed3eb` |
| Normal PFM | `acf82e26ab479853be41ff9b3cc348b740cbb77f37671db0d1864efa09052bd0` |
| Albedo PFM | `8e79df2a612628d23badc4d3dcb8ca2e0d8428bcc10ee77edfe44a5cf562f` |

The complete profile is
`/var/tmp/psycles-coro-pass-domain-wavefront-xLxG929W/trace.log` on the
measurement host.

## Value-numbered coroutine dataflow

Luisa `next@35c1ac2fc` removes the two remaining accidental quadratic
analyses at the distill/split boundary. `perf` attributed essentially all of
the former distill time to repeated `unordered_set<Value *>` construction and
instruction transfer while revisiting predecessor states. Each scope now
numbers its values and blocks once, summarizes each block once, and solves the
induced sparse CFG with dense bit-vector worklists. For block `B`, the transfer
functions are

`K_out = K_in union K_B`,
`T_out = T_in union T_B`, and
`E_out = E_in union (E_B - K_in)`.

`K` is a must-definition fact and meets with intersection; `T` and `E` are
may facts and meet with union. The must solution is deliberately initialized
to top for every non-entry block and to the empty boundary at the entry. This
computes the greatest fixed point. An initially faster bottom-initialized
prototype selected the wrong fixed point on loop back-edges, expanded the
Lone Monk frame from 297 fields / 1,424 bytes to 554 fields / 2,528 bytes, and
was rejected. `LUISA_CORO_VERIFY_DENSE_DATAFLOW=1` reruns the old pointer-set
solver as an oracle and compares every scope fact and exit fact. The corrected
solver passes that oracle on the complete Lone Monk coroutine and has a
dedicated loop regression that distinguishes top from bottom initialization.

Split no longer reruns the complete distill fixed point merely to validate its
input. A sealed analysis certificate binds the source definition, ordered
blocks and instructions, instruction kinds/types/names/operands and coroutine
payloads, plus every semantic scope, edge, liveness, and frame result. Split
checks the certificate in linear time and atomically rejects either caller
mutation or a source edit after distillation. A regression changes a store
operand without changing any block or instruction identity and proves that
the stale certificate is rejected. The full canonical recomputation remains
available under `LUISA_XIR_VERIFY_INTERMEDIATE=1`; the same flag enables the
extra intermediate XIR verifier. Normal compilation verifies only the whole
coroutine pass interval at its beginning and end.

The unchanged strict native-XIR Vulkan 1x1/1 spp Lone Monk measurement is:

| Coroutine phase | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| CFG distillation | 2,528.185 ms | 331.869 ms | 7.6x |
| coroutine splitting | 2,543.782 ms | 13.631 ms | 186.6x |
| complete coroutine lowering | 7,250.052 ms | 2,495.317 ms | 2.9x |

The result remains four callables with 297 frame fields / 1,424 bytes. Display,
Combined, Normal, and Albedo hashes are exactly the four hashes recorded above;
there is no shader-state or image change. The production run is under
`/var/tmp/psycles-coro-dense-top-production`, and the pointer-oracle run is
under `/var/tmp/psycles-coro-dense-top-oracle` on the measurement host.

All 53 focused XIR/coroutine CTest executables pass after the provenance API
change. The all-scheduler suite additionally passes 61 assertions in 12 tests
on fallback, HIP, and strict native-XIR Vulkan, including optimized-away
frontend tokens, sparse live tokens, the entry-only empty-live-set case, and a
root coroutine with no front-end suspend at all. This checkpoint uses Luisa
`next@b8c13d424`.

## Multi-sample renderer equivalence after the cut

The actual Lone Monk renderer was also run through the new loop-header cut with
four samples in one dispatch. At 16x16 on fallback, megakernel and wavefront
produced byte-identical display and linear passes:

| Output | SHA-256 shared by megakernel and wavefront |
| --- | --- |
| display PPM | `a0dcfa771396c998feab88b5c6ee5fb5a2248b93ec1bcb5c0cf9d01e48dbf7aa` |
| Combined PFM | `3f59ab4c3ca5fcc957ef228acccc208059b4f34b52fd588e3768042052bc6477` |
| Normal PFM | `f14a2c3fd58474d662060ea30380f8f4c76028e7cb84fc1de26da58e81423d72` |
| Albedo PFM | `17b1a2ba9c5d81e9ae483ef734f39bb0a0f041dd216e4dbbebd56f96b703d0a0` |

Exact-channel `idiff` also reports `PASS` for the multilayer EXR. The display
output was opened after nearest-neighbour enlargement: the coarse Lone Monk
framing, bright sky/window region, warm wall, dark foreground, and green grass
are present, with no scheduler-dependent difference (the source images are
byte-identical). The megakernel rendered in 0.0127 seconds after 0.0818 seconds
of JIT; wavefront rendered in 0.0399 seconds after 48.173 seconds of JIT. This
confirms sample-state equivalence while also showing that continuation
generation, especially the material-heavy surface continuation, remains the
dominant engineering problem.

## Visual inspection

The triptych was opened at its original 1968x525 resolution. Camera framing,
silhouettes, grass, windows, bright roof surfaces, and the sparse one-sample
fireflies coincide in all three panels; no scheduler-dependent structural or
shading difference is visible. This agrees with the byte-exact PPM/PFM result.

![Megakernel, wavefront, and persistent Lone Monk output](lone-monk-fallback-schedulers.png)

The source `.blend` enables Cycles adaptive sampling and denoising, while these
Psycles rows intentionally use fixed one-sample, un-denoised output. Therefore
this image must not be used as a Cycles quality gate; the established high-spp
five-way validation remains the Cycles-alignment evidence.
