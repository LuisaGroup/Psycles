"""Freeze original GPU states, full scene controls and validation of Voronoi."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'docs/validation/2026-09-08/lamp-routing-and-surface'))
from archive_validation import campaign, gate
from audit_surface import source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--child', required=True)
    args = parser.parse_args()
    evidence = args.evidence
    profiles = json.loads((evidence / 'profiles.json').read_text())
    assert profiles['schema'] == 'psycles.voronoi-octave-profiles.v1'
    assert set(profiles['runs']) == {'before', 'shared_first', 'shared_repeat', 'restored', 'final'}
    assert '3/8' in (evidence / 'structure-red.log').read_text()
    assert '8/8' in (evidence / 'structure-green.log').read_text()
    assert '168/192' in (evidence / 'focused-hip.log').read_text()
    assert '192/192' in (evidence / 'focused-hip-final.log').read_text()
    fixture = ROOT / 'tests/data/cycles_svm_voronoi_octave_state.txt'
    assert fixture.read_bytes() == (evidence / 'original-oracle.txt').read_bytes()
    assert len(fixture.read_text().splitlines()) == 192
    native = (evidence / 'native-vk.log').read_text()
    assert all(flag in native for flag in (
        'LUISA_VULKAN_USE_XIR=1', 'LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1',
        'LUISA_VULKAN_DISABLE_DXC=1'))
    assert '192/192' in native
    assert not re.search(r'(?:find library=|calling init:).*lib(?:dxcompiler|dxil|dxc)', native, re.I)
    modules = native.count('SPIR-V compilation successful')
    assert modules > 0
    canaries = campaign(evidence / 'canaries/canaries.json', True)
    assert (canaries['data']['implementation_sha256'] ==
            profiles['runs']['final']['command']['implementation_sha256'])
    result = {
        'schema': 'psycles.voronoi-octave.v1',
        'implementation': {'parent': args.parent, 'child': args.child,
            'before_parent': '5c3746b2', 'before_child': 'da8fff856',
            'changes': ['shared original F1 octave fallback', 'native homogeneous W in 3D position'],
            'unchanged': 'typed words, payload addressing, PC, feature masks, inlining policy, fast math, register limits, launch policy and frame'},
        'original_reference': {
            'revision': 'cb168525138fecc792cc393f94afc39582b0103c',
            'sources': [source(Path('/home/mike/Projects/blender-cycles-trace-5.2/intern/cycles') / name)
                        for name in ('kernel/svm/voronoi.h', 'kernel/svm/node_types_template.h',
                                     'util/types_float4.h')],
            'scope': '192 original HIP typed Voronoi node executions, not a CPU shader evaluator'},
        'regressions': {'structural_configurations': 8, 'before_failed_configurations': 5,
            'structural_scope': 'production AST radius-one F1/F2 and radius-two Smooth F1 loop families under both original feature-mask admissions; no instruction ceiling',
            'runtime_states': 192, 'before_failed_runtime_states': 24,
            'runtime_scope': '1D/2D/3D/4D, F1/F2/Smooth F1, zero and positive smoothing, zero and fractional detail, metrics and normalization variations; all nine output lanes and exact 13-word PC; existing independent full-interpreter Voronoi regression also passes',
            'float_comparison': 'finite abs-relative tolerance 2e-5, no bit-matching arithmetic',
            'sources': [source(evidence / name) for name in (
                'formal-analysis.md', 'structure-red.log', 'structure-green.log',
                'focused-hip.log', 'focused-hip-final.log', 'original-oracle.txt',
                'build-original-oracle.log', 'before_cycles_voronoi.cpp', 'after_cycles_voronoi.cpp')],
            'fixture': source(fixture),
            'observer': source(ROOT / 'tools/cycles_svm_voronoi_octave_oracle.hip'),
            'input_marshaller': source(ROOT / 'tests/cycles_svm_voronoi_octave_test_data.h'),
            'final_source': source(ROOT / 'src/luisa/cycles_voronoi.cpp')},
        'profiles': {'source': source(evidence / 'profiles.json'), 'data': profiles},
        'gates': {'host': gate(evidence / 'full-host.log', 172),
            'hip': gate(evidence / 'full-hip.log', 184),
            'focused_hip': gate(evidence / 'focused-hip-final.log', 2),
            'fallback': gate(evidence / 'full-fallback.log', 186),
            'native_vulkan': {**gate(evidence / 'native-vk.log', 5),
                'modules': modules, 'dxc_dxil_loaded': False},
            'build': source(evidence / 'build-final.log')},
        'canaries': canaries,
        'visual_qa': {'inspected': [str(evidence / 'canaries' / (scene + '-fixed-1-triptychs/combined.png'))
            for scene in ('barbershop', 'monk', 'monster', 'classroom')],
            'scope': 'four first-run Combined triptychs viewed; residual pass differences unresolved'},
        'open': ['Barbershop per-surface cost and residual DiffInd/shadow work',
            'remaining SVM input/closure and code-generation differences',
            '164 used-shader resource binding identities',
            'unimplemented SVM nodes and private legacy displacement bridge',
            'existing native Vulkan f16/f64 remainder failures',
            'remaining general restructure_cfg proof obligations'],
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
