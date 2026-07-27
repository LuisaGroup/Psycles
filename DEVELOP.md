# Psycles development status

This is a living implementation plan, last updated 2026-07-27. “Implemented”
does not mean “Cycles compatible”: compatibility requires an official Blender
4.5.10 Cycles probe and a versioned linear-pass baseline.

## Current checkpoint

| Area | Verified state |
|---|---|
| Shader inventory | 96 Cycles-applicable nodes tracked from 105 Blender shader node types |
| Complete coverage | 43/96 complete: 41 `cycles_verified` device nodes and 2 structural output adapters |
| Remaining nodes | 12 partial and 41 pending; no implemented node is waiting for a probe, and 1 Cycles OSL-only node is tracked separately |
| Automated gate | 6/6 CTest groups pass in the current Luisa/fallback build |
| Analytic lights | 11 Point/Spot/Area/Sun baselines, including shapes, spread, finite Sun disk, and light node trees |
| Full-scene geometry/AOV | Negative-scale normal transforms and closure-weighted glossy normals are fixed |
| Full-scene transport | At 640×480/64 spp, Lone Monk Combined RMSE is `0.26116`, Normal is `0.03696`, and DiffCol is `0.01175`; Combined mean energy is 95.75% of Cycles |
| Cold/hot fallback JIT | Frozen-runtime full-scene JIT is `327.574 s` cold and `0.682609 s` hot (479.9×); the 1.20 MB main object loads in 1.87 ms |
| Persistent fallback cache | Native object plus exact metadata implemented, 8/8 isolated assertions pass, and the full-scene cross-process run is bitwise equal across 13 passes |
| Upstream integration | LuisaCompute draft PR [#253](https://github.com/LuisaGroup/LuisaCompute/pull/253); Psycles pins commit `6b6a63d` as its submodule |

The latest glossy-normal probe reduced Normal RMSE from `0.399218` to
`0.00192210` (about 99.5%) and measures Combined relative RMSE `0.5273%`.
The earlier 64×48/256 spp glossy-normal checkpoint measured Normal RMSE
`0.02515`. The current 640×480/64 spp component run measures 95.75% Combined,
97.91% Diffuse Direct, 86.21% Diffuse Indirect, 95.15% Glossy Direct, and
84.33% Glossy Indirect mean energy relative to Cycles. This separates the
remaining indirect-transport deficit from exposure and geometric-normal
errors.

## Binding and JIT policy

Values that can change without changing generated control flow belong in a
kernel argument, resource buffer, or bindless resource:

- render seed, sample start/count, resolution, and exposure;
- camera and transform values;
- integrator thresholds, bounce limits, and clamps;
- material socket values and texture/resource handles;
- light parameters and scene data.

Topology, static node modes, callable signatures, and resource binding types
remain structural. The cache key represents that structural program and its
code-generation environment; it must not absorb frequently changing values.
Each migration to a runtime argument needs two tests: changing the value must
change the result, and it must not create a new shader cache entry.

The existing per-material `GraphSurface` specialization still makes AST size
grow with material topology. Shared callables are an intermediate reduction.
The intended endpoint is a compact, buffer-driven typed-value instruction
stream whose kernel structure stays stable as ordinary material parameters
and most graph data change.

## Sampling parity

Psycles is deterministic for a fixed Psycles seed, but its random stream is not
yet bitwise identical to Cycles. The current integer state/hash generator does
not reproduce the complete Cycles 4.5 Sobol/PMJ sampler, scrambling, or exact
random-dimension advancement at camera, light, BSDF, transparency, and
Russian-roulette events.

No “exact RNG” claim will be made from converged image statistics. The release
gate is a trace probe that records, for fixed pixel/seed/sample indices:

1. pixel-filter and lens samples;
2. light selection and light-shape dimensions;
3. BSDF component and direction dimensions at every bounce;
4. transparency and Russian-roulette dimensions;
5. the final consumed dimension index.

Every value and advancement decision must match the corresponding Cycles trace
before the gate can turn green.

## Roadmap

### P0 — iteration latency and reproducibility

- [x] Validate fallback persistent cache on a full Lone Monk cold/hot
  cross-process run, including 13 bitwise-identical linear passes.
- [x] Submit the cache as LuisaCompute draft PR
  [#253](https://github.com/LuisaGroup/LuisaCompute/pull/253) and pin the
  tested commit through the Psycles submodule.
- [x] Move render window/resolution, sample range/seed, continuous camera
  values, bounce limits, clamps, background, filter width, and pass alpha
  threshold out of the AST and into kernel arguments.
- [x] Record cache hit/miss and compile-stage timings: `183.578 s` main
  AST-to-XIR cold, `327.574 s` total cold JIT, `0.682609 s` hot JIT, and
  `1.8655 ms` main-object load.

Changing the render from 64×48/1 spp to 640×480/64 spp retained the exact same
main cache key, `kernel_bb7a6886f6f75b90`, and completed shader setup in
`0.789711 s`. Structural modes such as projection type and static node modes
remain deliberately outside the runtime argument block.

### P1 — transport parity

- Implement Cycles-compatible environment and emitter importance
  distributions, including PDFs used by MIS.
- Close indirect-transmission and glossy-indirect energy gaps.
- Complete the remaining Principled lobes and event labels.
- Finish Bump/Normal modes and derivative behavior, then re-run full-scene
  AOV and Combined gates.
- Add the bitwise Cycles random-dimension trace and replace the current RNG
  path only when each event is covered.

### P2 — shader graph coverage

- Convert each of the 16 partial nodes into focused mode/socket probes.
- Implement the 41 pending Cycles nodes in dependency order.
- Keep `tools/check_cycles_shader_node_coverage.py --require-complete` red
  until every applicable node is verified or explicitly classified outside
  device scope.
- Add volume and displacement roots after their isolated contracts exist;
  do not silently route them through Surface.

### P3 — performance architecture

- Replace per-material expanded DSL graphs with a shared device instruction
  executor and immutable program buffers.
- Deduplicate texture/attribute services and material parameter layouts.
- Add GPU-backend differential/performance runs after fallback semantics are
  stable.
- Evaluate scheduling or material clustering only after semantic gates remain
  green under the shared device IR.

## Completion criteria

Psycles is not complete merely because a showcase scene renders. A release
candidate requires:

- all 96 applicable shader nodes resolved by versioned policy and all supported
  modes officially probed;
- Surface, Volume, and Displacement behavior explicitly supported or rejected
  at import with no silent approximation;
- integrator, analytic-light, world, BSDF, transparency, pass, and color
  management gates green;
- exact sampling-trace status documented (green if exact, otherwise a named
  compatibility limitation);
- no invalid pixels, deterministic cache behavior, reproducible builds, and
  accepted full-scene linear-pass thresholds;
- cold and warm JIT, render time, and memory baselines recorded for both the
  fallback backend and at least one supported GPU backend.
