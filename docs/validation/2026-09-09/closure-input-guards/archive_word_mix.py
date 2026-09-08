"""Count typed surface opcodes and closure producers; no shader execution."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'docs/validation/2026-09-08/svm-math-expansion'))
from audit_typed_words import decode, read_dump
from audit_surface import source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('layouts', 'word_audit', 'cycles', 'psycles', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.layouts.read_text())
    audit = json.loads(args.word_audit.read_text())
    _, originals = read_dump(args.cycles)
    _, actuals = read_dump(args.psycles)
    nodes, closures, shader_counts = Counter(), Counter(), Counter()
    for shader in audit['shaders']:
        original, actual = originals[shader['cycles_index']], actuals[shader['index']]
        original_nodes, _ = decode(original, metadata)
        actual_nodes, _ = decode(actual, metadata)
        assert original_nodes == actual_nodes
        seen = set()
        for pc, name, size in actual_nodes:
            if not actual[1] <= pc < actual[2]:
                continue
            assert pc + size <= actual[2]
            nodes[name] += 1
            if name == 'NODE_CLOSURE_BSDF':
                kind = actual[pc + 1]
                assert kind == original[pc + 1]
                closures[kind] += 1
                seen.add(kind)
        shader_counts.update(seen)
    result = {
        'schema': 'psycles.surface-word-mix.v1',
        'scope': 'used-shader static surface-with-bump word domain; counts are not execution frequencies or timing attribution; exact original and actual node layouts and closure type words checked',
        'shaders': len(audit['shaders']),
        'opcodes': dict(nodes.most_common()),
        'closures': {str(k): {'static_nodes': v, 'shader_count': shader_counts[k],
                             'payload': metadata['closures'][str(k)]}
                     for k, v in sorted(closures.items())},
        'sources': [source(getattr(args, name))
                    for name in ('layouts', 'word_audit', 'cycles', 'psycles')],
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
