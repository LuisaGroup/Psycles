# Lone Monk: Metal / Metal4 scheduler validation

## Published backend repairs

Luisa `8911828eb` repairs the three failures discovered while attempting the
exact upstream Psycles `f690ead6` / Luisa `da8fff856` workload:

- Metal AST emission dropped a `Local` declaration together with its lexical
  lifetime seed.
- Metal4 LLVM switch-to-lookup created constant tables in thread address
  space, causing unresolved symbols when AIR pipelines were linked.
- The shared Metal-cpp ownership wrapper invoked C++ retain/release through
  null pointers. Clang 21 could optimize successful pipeline loading into a
  trap. Nullable operations now have explicit guards.

These are generic Luisa changes, published to `next` before advancing this
gitlink. Psycles renderer, SVM and scheduler implementations are unchanged.
The SDK [regression and proof record](../../../../third_party/LuisaCompute/docs/validation/2026-09-09/metal-world-shader-codegen/README.md)
includes minimal failures, permanent tests, cache invalidation and the
additional historical mutable-swizzle fixture that still fails on Metal4.
Focused validation is not represented as a full-suite pass.

## Original-scene gate (not a performance result)

Both backends completed the original Lone Monk scene at 1920x1080, one sample,
using staged wavefront with an independent direct-light queue. Both expose
four main stages, **55 fields / 220 B**, matching the HIP report's main-frame
boundary. Both EXRs contain precisely 15 common linear passes / 46 channels,
all finite. One-sample times are not extrapolated to the requested 256 spp.

The actual queue capacities differ under the unchanged application policy:
Metal is capped at 131072 pixel-samples per dispatch; Metal4 has no equivalent
name-based watchdog cap. With a requested 1048576-frame capacity this produces
131072 slots for Metal and 1048576 for Metal4. Shadow queue payload/state is
additional storage, not part of the 220-byte main frame.

## Formal campaign configuration

Status: backend validation complete; 1080p/256-sample measurements follow this
published checkpoint. There are no completed Psycles timing pairs yet.

- Host: Apple M1 Max, 10 CPU cores, macOS 26.6.2 / 25G83.
- Release build: Homebrew Clang 21.1.8; Metal4 uses LLVM 22.1.8.
- Blender: official 5.2.1 LTS arm64, build `9e2066aef7ef`.
- Blender DMG SHA-256: `6409e21de80994db5f4c4a34486b6fd43cea21085b912f7491c53e923acb65a3`.
- Fresh export of the authored Lone Monk blend; 1920x1080, 256 fixed samples,
  authored frame/seed. The HIP report used 1440x1080; the requested Mac extent
  is retained, so its absolute timings are not transferred to this machine.
- Cycles Metal explicitly selects Apple M1 Max. Denoising and adaptive
  sampling are disabled. Both engines write the same 46 channels.
- `PSYCLES_COMPACT_SURFACE_VALUES=1`, `PSYCLES_POPULATE_SURFACE_ONCE=1`,
  `PSYCLES_DISABLE_SHADER_CACHE=1`; native fast math remains enabled.
- Main shader cache is disabled; auxiliary, downstream and OS caches retain
  their normal behavior. No build or other render overlaps a timing run.
- Compare Cycles' original main-loop wall time with Psycles render-only wall
  time. Session initialization includes JIT and setup/baking and is separate.

The first profile follows the report's staged scheduler: logical sample batch
64, execution block 32, requested frame capacity 1048576, surface sorting and
independent direct-light queue enabled, counter batch 4 / pipeline depth 2,
tail disabled. Surface block 512 is upstream policy. The graph comparison
uses batch 64, 131072 workers, selective scheduling, automatic tail,
counter batch 1 / depth 1 and inline shadow work. Main-frame bytes therefore
refer to different state boundaries between staged and graph.

The canonical `tools/run_scene_benchmark.py` v3 runner is used through a
build-local adapter that adds only `metal4` to its backend-name whitelist.
Command construction, EXR validation, timing extraction, comparison and
resume logic remain upstream. Implementation hashes are checked before and
after each run; exact commands and source identities are retained.

## Evidence

Local experiment directory:
`build-macos/benchmarks/2026-09-09/lone-monk-metal-schedulers`.

- `staged/run-1` and `staged-metal4/run-1`: original pinned-source failures,
  before either scheduler was constructed. Their Cycles references succeeded;
  those failed matrices are not completed performance comparisons.
- `regressions/`, red/green test logs and AIR dumps: compiler/backend evidence.
- `canary/{metal,metal4}`: complete original-scene 1-sample gates and pass inventories.
- `fixed-published/{staged,graph}/run-N`: formal fresh-reference timing pairs,
  implementation hashes, commands, 15-pass comparisons and original-size images.

The earlier schema-v1/v2 results are not reused. They have superseded timing
boundaries or unequal pass workloads, as documented by the canonical runner.
