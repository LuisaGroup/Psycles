"""Freeze observed binding checks and their validation, without timing claims."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'docs/validation/2026-09-08/lamp-routing-and-surface'))
from decode_cycles_bindings import source
from archive_validation import gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    evidence = args.evidence
    comparison = json.loads((evidence / 'binding-comparison.json').read_text())
    assert comparison['summary']['equal_binding_semantics'] == 279
    assert comparison['summary']['reference_mismatches'] == 0
    assert comparison['summary']['nonresource_word_mismatches'] == 0
    # The checked-in replay patch omits only the final blank context line;
    # its hunk counts are adjusted accordingly. The observed additions remain
    # byte-identical, and both patch forms must be applicable to the original.
    observed_patch = (evidence / 'observer-hooks.patch').read_text()
    replay_patch = (ROOT / 'tools/cycles_binding_oracle.patch').read_text()
    assert replay_patch == observed_patch.removesuffix(' \n').replace(
        '@@ -493,6 +494,7 @@', '@@ -493,5 +494,6 @@')
    restored = []
    for line in (evidence / 'original-source-sha256.txt').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = Path(name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        restored.append(source(path))
    assert not Path('/home/mike/Projects/blender-cycles-trace-5.2/intern/cycles/scene/psycles_binding_oracle.h').exists()
    native = (evidence / 'focused-vk.log').read_text()
    for flag in ('LUISA_VULKAN_USE_XIR=1', 'LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1', 'LUISA_VULKAN_DISABLE_DXC=1'):
        assert flag in native
    assert not re.search(r'(?:find library=|calling init:).*lib(?:dxcompiler|dxil|dxc)', native, re.I)
    modules = native.count('SPIR-V compilation successful')
    assert modules == 2
    raw = json.loads((evidence / 'scene-words.json').read_text())
    typed = json.loads((evidence / 'typed-words.json').read_text())
    result = {'schema': 'psycles.resource-identities.v1',
        'implementation': {'parent': '076cf9f5', 'runtime_parent': 'ffb3f7f2', 'child': 'da8fff856',
                           'change': 'diagnostics, captured-fixture regression and documentation only; no rendering/compiler optimization'},
        'original': {'revision': 'cb168525138fecc792cc393f94afc39582b0103c',
            'restored_sources': restored,
            'scope': 'fresh original HIP session at 2048x858/1 spp with metadata-only host observers; SVM and registry captures are from this same session',
            'resources_environment': {'BLENDER_SYSTEM_RESOURCES': '/home/mike/Projects/blender-install-psycles-trace-5.2/5.2'},
            'observer_binary': source(evidence / 'blender-resource-observer'),
            'restored_build_binary': source(Path('/home/mike/Projects/blender-build-psycles-trace-5.2/bin/blender')),
            'startup_failures': 'First two build-tree launches lacked Blender resources and failed before rendering; retained logs. The full child-process resource path fixed startup; no system installation was changed.'},
        'summary': comparison['summary'], 'raw_words_summary': raw['summary'],
        'typed_words_summary_before_binding_resolution': typed['summary'],
        'same_numeric_id_references_checked': sum(r['cycles_id'] == r['psycles_id'] for s in comparison['shaders'] for r in s['references']),
        'attributes': comparison['attributes'],
        'images': comparison['images'],
        'shaders': [{'name': s['name'], 'index': s['index'], 'cycles_index': s['cycles_index'],
            'same_typed_layout': s['same_typed_layout'], 'same_binding_semantics': s['same_binding_semantics'],
            'references': dict(Counter(r['kind'] for r in s['references'])),
            'nonresource_word_mismatches': len(s['nonresource_changes']),
            'reference_mismatches': sum(not r['same_identity'] for r in s['references'])}
            for s in comparison['shaders']],
        'regression': {'source': source(ROOT / 'tests/test_cycles_binding_oracle.py'),
            'fixture': source(ROOT / 'tests/data/cycles_resource_bindings.json'),
            'cases': 8, 'scope': 'complete captured registries and one complete original/actual shader; malformed protocols, metadata mismatches and same-number/wrong-resource negative controls'},
        'gates': {'host': gate(evidence / 'host.log', 173),
            'hip_focused': gate(evidence / 'focused-hip.log', 2),
            'fallback_focused': gate(evidence / 'focused-fallback.log', 2),
            'native_vulkan_focused': {**gate(evidence / 'focused-vk.log', 2), 'modules': modules, 'dxc_dxil_loaded': False}},
        'sources': comparison['sources'] + [source(evidence / name) for name in (
            'analysis.md', 'build-observer.log', 'build-restored.log', 'build-actual.log', 'build-final.log',
            'original-render.log', 'original-render-resources.log', 'original-render-final.log',
            'original-source-sha256.txt', 'observer-sha256.txt', 'observer-hooks.patch',
            'barbershop-original.json', 'actual-compile.log', 'binding-comparison.json', 'scene-words.json',
            'typed-words.json', 'regression.log', 'CMakeLists.txt')],
        'limits': ['Identity checks do not compare decoded texels, filtering results, geometry attribute values or shader-state evolution.',
            'Names are admitted only when unique in both observed registries and the export; unsupported animation, tiled/UDIM or generated-sky identities fail closed.',
            'No benchmark rerun or speedup is attributed to these diagnostic-only changes; the preceding six-scene-run checkpoint remains current.',
            'Barbershop surface cost, residual indirect/shadow path differences, unimplemented opcodes, legacy displacement removal and remaining generic CFG proofs are still open.']}
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
