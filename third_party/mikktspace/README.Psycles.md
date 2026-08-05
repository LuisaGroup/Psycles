# MikkTSpace dependency

Psycles vendors Blender's current header-only MikkTSpace implementation to
construct the same corner tangent frames that Cycles consumes before and after
true displacement.

- Upstream: https://projects.blender.org/blender/blender/src/commit/29ccd5e2e824128c86fc6174c9c502c02212434a/intern/mikktspace
- Imported revision: `29ccd5e2e824128c86fc6174c9c502c02212434a`
- License: Apache-2.0; see [LICENSE](LICENSE).
- Local integration: `src/luisa/path_tracer_tangent_space.cpp`

The files are kept unmodified so a future Blender/Cycles alignment update can
be reviewed as an upstream diff. This is host-side geometry preprocessing; it
does not bake or evaluate shader closures.
