"""Freeze native closure-setup dispatch controls and backend verification."""
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
    assert profiles['schema'] == 'psycles.closure-setup-dispatch-profiles.v1'
    assert '60/72' in (evidence / 'structure-red.log').read_text()
    assert '72/72' in (evidence / 'structure-green.log').read_text()
    native = (evidence / 'native-vk.log').read_text()
    assert all(flag in native for flag in (
        'LUISA_VULKAN_USE_XIR=1', 'LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1',
        'LUISA_VULKAN_DISABLE_DXC=1'))
    assert not re.search(r'(?:find library=|calling init:).*lib(?:dxcompiler|dxil|dxc)', native, re.I)
    modules = native.count('SPIR-V compilation successful')
    assert modules > 0
    result = {
        'schema': 'psycles.closure-setup-dispatch.v1',
        'implementation': {'parent': args.parent, 'child': args.child,
            'before_parent': '1d00c12c', 'before_child': 'da8fff856',
            'no_inlining_policy_math_register_limit_or_word_image_change': True},
        'original_reference': {
            'revision': 'cb168525138fecc792cc393f94afc39582b0103c',
            'source': source(Path('/home/mike/Projects/blender-cycles-trace-5.2/intern/cycles/kernel/svm/closure.h')),
            'scope': 'original closure-type switch and shared groups; no CPU shader evaluator'},
        'regressions': {'structural_shapes': 72, 'before_failed_shapes': 12,
            'scope': 'six kernel-feature combinations, four node masks, three shader domains',
            'sources': [source(evidence / name) for name in (
                'formal-analysis.md', 'structure-red.log', 'structure-green.log')]},
        'profiles': {'source': source(evidence / 'profiles.json'), 'data': profiles},
        'gates': {'host': gate(evidence / 'full-host.log', 170),
            'hip': gate(evidence / 'full-hip.log', 182),
            'focused_hip': gate(evidence / 'focused-hip.log', 20),
            'fallback': gate(evidence / 'full-fallback.log', 184),
            'native_vulkan': {**gate(evidence / 'native-vk.log', 3),
                'modules': modules, 'dxc_dxil_loaded': False},
            'build': source(evidence / 'build-final.log')},
        'canaries': campaign(evidence / 'canaries/canaries.json', True),
        'visual_qa': {'inspected': [str(evidence / 'canaries' / (scene + '-fixed-1-triptychs/combined.png'))
            for scene in ('barbershop', 'monk', 'monster', 'classroom')],
            'scope': 'four first-run Combined triptychs viewed; residual pass differences unresolved'},
        'open': ['Barbershop per-surface cost and residual DiffInd/shadow work',
            'eager parameter evaluation relative to native caustic/allocation guards',
            '164 used-shader resource binding identities',
            'unimplemented SVM nodes and remaining legacy displacement path',
            'existing native Vulkan f16/f64 remainder failures',
            'remaining general restructure_cfg proof obligations'],
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
