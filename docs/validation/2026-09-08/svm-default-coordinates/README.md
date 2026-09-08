# Original-scene reduction history, not a default-coordinate fix

This investigation starts with Barbershop's `heater_mat` schedule mismatch
and ends at the [lazy group-input cause](../svm-group-contexts/README.md).
The initial default-coordinate phase hypothesis was not established:
all 32 independent coordinate controls match original Cycles raw-exact.
No default-coordinate source change is retained.

Evidence root `/var/tmp/psycles-default-coordinates-FTAxTb` retains original
blends, Cycles HIP EXRs and dumps, exports, actual production dumps and typed
comparisons for every candidate batch. These tools only transform input
graphs or parse words; they do not evaluate a shader or synthesize expected
streams.

- `tools/create_cycles_default_coordinates_probe.py`: Checker/Gradient,
  explicit/implicit coordinates, bump/surface, shared color, reversed Mix
  input order; 32 controls plus world all raw-exact.
- `tools/create_cycles_material_reduction.py`: load only the original
  material from an immutable blend; twelve variants retain/replace images,
  retain/replace ramps, and select original/normal/shared roots. Replacing
  images preserves the equality of original image/sampler identities with
  matching Checker parameters. The corrected v2 has four equal-word-count
  scheduling witnesses; eight variants plus world are raw-exact.
- `tools/create_cycles_link_reduction_batch.py`: retain a baseline and one
  candidate per deleted authored link, trimming only unreachable nodes. The
  nested mode copies each group instance before editing it. Source material
  renaming avoids collisions with the next batch's `candidate-000`.
- `reduce_links.py`: for every batch, run original Cycles HIP, export the
  exact candidate graphs, run the production compiler, then use the existing
  raw and typed decoders. Preserve the equal-size/equal-node-count witness
  whose first original opcode is MAPPING and whose actual opcode is an image
  or Checker texture. Never normalize resource IDs or expected words.

Top-level deletion ends at 19 nodes / 18 links / 1,374 words.
`nested-reduction-v2/round-05` ends at 24 nodes / 22 links / 232 words,
counting parent and nested trees. No single remaining link deletion retains
the selected witness. This is a greedy, scope-specific reduction, not proof
of global minimality. The live graph has a shared Checker created early by
an unused parent input: deleting that input's internal consumer did not stop
the old importer from eagerly lowering the input.

The first nested batch stopped because Blender renamed a colliding source
material `candidate-000.001`; that failed batch is retained separately as
`nested-reduction`, and must not be confused with the successful v2 history.
The input-reduction v1 also lost the original image-sharing relation and
therefore hid the no-image witness. Both diagnostic mistakes are retained.

Reproduce the greedy nested phase with `reduce_links.py SOURCE.blend MATERIAL
NEW_DIRECTORY --blender ORIGINAL_BLENDER --dump-scene PRODUCTION_DUMPER
--layouts CHECKED_LAYOUTS.json --nested`. The dumper and metadata generator
are in the sibling lamp-routing and Math reports. Each batch records exact
commands and return codes through its logs; `history.json` records the
strictly decreasing selected graph measure. New independent original group
fixtures, not this large reduced scene, provide the permanent regression.
