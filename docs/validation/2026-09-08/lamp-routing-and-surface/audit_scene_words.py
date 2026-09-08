"""Compare relocated used shader images; do not normalize or invent payloads."""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct

from audit_surface import source


def read_dump(path):
    data = path.read_bytes()
    assert data[:8] == b'PSYSVM52'
    version, count, shader_count = struct.unpack_from('<IQI', data, 8)
    assert version == 1
    cursor = 24
    names = {}
    for _ in range(shader_count):
        index, length = struct.unpack_from('<II', data, cursor)
        cursor += 8
        names[index] = data[cursor:cursor + length].decode()
        cursor += length
    assert len(data) - cursor == count * 4
    words = struct.unpack_from(f'<{count}I', data, cursor)
    images = {}
    for i in range(shader_count):
        jump = words[4 * i:4 * i + 4]
        start = jump[1]
        end = words[4 * (i + 1) + 1] if i + 1 < shader_count else count
        assert jump[0] == 1 and shader_count * 4 <= start < end <= count
        assert all(start <= target < end for target in jump[1:])
        images[i] = (jump[0], *(target - start + 4 for target in jump[1:]), *words[start:end])
    return names, images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cycles', type=Path)
    parser.add_argument('psycles', type=Path)
    args = parser.parse_args()
    cn, ci = read_dump(args.cycles)
    pn, pi = read_dump(args.psycles)
    rows = []
    for index, name in pn.items():
        if not name:
            continue  # Explicitly inert, unused dense-index holes in this dump.
        cycles_index = 0 if name == '__cycles_default_surface__' else index
        assert cycles_index in cn
        # The exporter represents Cycles' default_background as the scene world.
        # The importer creates the original default Principled graph without
        # a source index; production assigns it a new index after authored ones.
        assert (name == cn[cycles_index]
                or (index == 3 and name.startswith('__world__'))
                or (name == '__cycles_default_surface__' and cn[cycles_index] == 'default_surface'))
        original, actual = ci[cycles_index], pi[index]
        # Domain boundaries are original ShaderJump targets, not inferred
        # opcode lengths. Surface includes the fall-through bump prelude.
        # A missing displacement tail must not hide extra surface words.
        domains = {}
        for label, slot in [("surface_with_bump", 1), ("volume", 2), ("displacement", 3)]:
            c_end = original[slot + 1] if slot < 3 else len(original)
            p_end = actual[slot + 1] if slot < 3 else len(actual)
            c_words = original[original[slot]:c_end]
            p_words = actual[actual[slot]:p_end]
            assert c_words and p_words and c_words[-1] == p_words[-1] == 0
            domains[label] = {"cycles_words": len(c_words), "psycles_words": len(p_words),
                              "size_difference": len(p_words) - len(c_words),
                              "raw_equal": c_words == p_words}
        differences = [i for i, (c, p) in enumerate(zip(original, actual)) if c != p]
        rows.append({
            'index': index, 'cycles_index': cycles_index,
            'name': name, 'cycles_name': cn[cycles_index],
            'cycles_words': len(original), 'psycles_words': len(actual),
            'size_difference': len(actual) - len(original),
            'cycles_local_jump': original[:4], 'psycles_local_jump': actual[:4],
            'domains': domains,
            'raw_equal': original == actual,
            'differing_common_words': len(differences),
            'first_differences': [{'local_word': i, 'cycles': f'{original[i]:08x}',
                                   'psycles': f'{actual[i]:08x}'} for i in differences[:12]],
        })
    print(json.dumps({
        'schema': 'psycles.used-svm-word-audit.v1',
        'scope': 'production used-shader domain; only ShaderJump global-to-local relocation; resource and attribute IDs and every literal payload remain unmodified; raw differences are not yet classified as semantic differences',
        'summary': dict(Counter('equal' if r['raw_equal'] else 'different_size' if r['size_difference'] else 'different_payload' for r in rows)),
        'sources': [source(args.cycles), source(args.psycles)],
        'shaders': rows,
    }, indent=2))


if __name__ == '__main__':
    main()
