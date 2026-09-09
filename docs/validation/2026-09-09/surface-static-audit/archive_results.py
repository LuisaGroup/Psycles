"""Freeze the scope-qualified static/format audit, without a speedup inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'docs/validation/2026-09-08/lamp-routing-and-surface'))
from audit_surface import source

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('evidence', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
graphs = {name: json.loads((args.evidence / f'{name}-address-graph.json').read_text())
          for name in ('original', 'actual')}
bindings = json.loads((ROOT / 'docs/validation/2026-09-09/resource-identities/results.json').read_text())
formats = {}
for line in (args.evidence / 'texture-formats.tsv').read_text().splitlines():
    index, width, height, kind, size, name_hex = line.split('\t')
    name = bytes.fromhex(name_hex).decode()
    assert name not in formats
    formats[name] = dict(index=int(index), width=int(width), height=int(height), storage=kind, bytes=int(size))
bound = []
missing = []
assert len({r['name'] for r in bindings['images']}) == len(bindings['images'])
for binding in bindings['images']:
    name, native = binding['name'], binding['cycles']
    if binding['load_failed']:
        assert native['width'] == native['height'] == native['channels'] == 0
        missing.append(name)
        continue
    row = formats[name]
    assert native['data_type'] in (0, 1)
    assert row['storage'] == {0: 'FLOAT4', 1: 'BYTE4'}[native['data_type']]
    assert (row['width'], row['height']) == (native['width'], native['height'])
    bound.append({'name': name, **row})
assert len(bound) == 172 and len(missing) == 2
assert sum(r['bytes'] for r in bound) == 157005312
result = {
    'schema': 'psycles.surface-static-audit.v1',
    'implementation': {'parent': '124cee8c', 'runtime_parent': 'ffb3f7f2', 'child': 'da8fff856'},
    'original_revision': 'cb168525138fecc792cc393f94afc39582b0103c',
    'scope': 'read-only observed function-address closure and production image decode/alpha formats',
    'graphs': graphs,
    'decoded_export_formats': dict(Counter(row['storage'] for row in formats.values())),
    'bound_valid_formats': dict(Counter(row['storage'] for row in bound)),
    'bound_payload_bytes': sum(row['bytes'] for row in bound),
    'bound_images': bound, 'missing_images': missing,
    'formats': formats,
    'sources': [source(args.evidence / name) for name in ('analysis.md', 'address_graph.py',
        'original-address-graph.json', 'actual-address-graph.json', 'texture_formats.cpp',
        'texture-formats.tsv', 'CMakeLists.txt', 'build-formats.log')],
    'binding_evidence': source(ROOT / 'docs/validation/2026-09-09/resource-identities/results.json'),
    'limitations': ['Address materializations are not proven call sites or dynamic counts.',
        'The parser does not validate complete CFG dominance, high carry or all indirect transfers.',
        'Native and actual static semantic coverage differs due to scene JIT pruning.',
        'ELF function extents exclude padding; static scratch sites exclude private-pointer flat accesses.',
        'Decoded pixel payload is not driver allocation, sampling traffic or bandwidth.',
        'No runtime change, performance intervention or causality claim.'],
}
args.output.write_text(json.dumps(result, indent=2) + '\n')
