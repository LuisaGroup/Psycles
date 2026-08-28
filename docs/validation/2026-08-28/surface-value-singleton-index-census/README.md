# Surface-value singleton-index census

## Question

Can an exact surface-value handler replace runtime-indexed typed-bank reads or
writes with statically indexed locals often enough to justify another JIT
specialization dimension?

For an exact evaluator variant `v`, define `R(v)` as the set of local result
indices used by all bytecode instructions assigned to `v`.  A result write is
statically indexable if and only if `|R(v)| = 1`.  For local operand coordinate
`(v, k)`, define `L(v, k)` analogously after the already-proven
parameter/local route partition.  That operand read is statically indexable if
and only if `|L(v, k)| = 1`.  These are finite-set cardinality tests over the
complete scene image; no sampled material values or hash identity enter the
classification.

The inspector independently reconstructs these sets from the final bytecode
and `instruction_variants` streams.  It does not read a compiler-published
singleton flag.  A synthetic regression contains both singleton and
non-singleton fibers for results and local operands:

```text
ctest --test-dir build -R psycles.inspect_blender_material_operand_index_partition --output-on-failure
```

## Complex-scene census

Commands used the Blender 5.2 exports and the scene-wide `*` inspector mode:

```text
build/bin/psycles_inspect_blender_material <export> '*'
```

| Scene | topologies | preparation instructions | static result refs | dynamic result refs | singleton local coordinates | local coordinates | static local refs | dynamic local refs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Barbershop | 190 | 5,159 | 13 | 5,146 | 11 | 58 | 13 | 5,278 |
| Classroom | 59 | 828 | 15 | 813 | 11 | 35 | 26 | 933 |
| Monster Under the Bed | 19 | 234 | 33 | 201 | 8 | 36 | 34 | 205 |
| Lone Monk | 25 | 387 | 24 | 363 | 11 | 37 | 16 | 371 |

Barbershop, the current large material stress case, exposes only 0.252% of
result writes and 0.246% of local operand references to this transformation.
Classroom is similarly low.  Monster and Lone Monk have higher percentages,
but far fewer instructions and therefore do not justify increasing the shared
handler dispatch domain for the production scenes.

## Decision

Retain the exact census and its regression, but do not add static local-index
specialization to the renderer.  The result rules out a tempting source-level
micro-optimization with scene-wide evidence.  The next measurement should be
hit-weighted: count one surface topology per population and project those exact
counts through each topology's immutable preparation program to identify the
dynamically dominant value handlers without instrumenting every bytecode
instruction.
