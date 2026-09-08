"""Freeze original-GPU guard states, A/B/B/A and backend/scene validation."""
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
    assert profiles['schema'] == 'psycles.closure-input-guards-profiles.v1'
    assert '0/8' in (evidence / 'structure-red.log').read_text()
    assert '8/8' in (evidence / 'structure-green.log').read_text()
    assert '72/72' in (evidence / 'dispatch-green.log').read_text()
    assert '128/128' in (evidence / 'guards-hip-final.log').read_text()
    fixture = ROOT / 'tests/data/cycles_svm_closure_guards_state.txt'
    assert fixture.read_bytes() == (evidence / 'original-state-final.txt').read_bytes()
    assert len(fixture.read_text().splitlines()) == 96
    native = (evidence / 'native-vk.log').read_text()
    assert all(flag in native for flag in (
        'LUISA_VULKAN_USE_XIR=1', 'LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1',
        'LUISA_VULKAN_DISABLE_DXC=1'))
    assert not re.search(r'(?:find library=|calling init:).*lib(?:dxcompiler|dxil|dxc)', native, re.I)
    modules = native.count('SPIR-V compilation successful')
    assert modules > 0
    word_mix = json.loads((evidence / 'word-mix.json').read_text())
    assert word_mix['schema'] == 'psycles.surface-word-mix.v1'
    result = {
        'schema': 'psycles.closure-input-guards.v1',
        'implementation': {'parent': args.parent, 'child': args.child,
            'before_parent': 'b5724231', 'before_child': 'da8fff856',
            'no_inlining_policy_math_register_limit_or_word_image_change': True},
        'original_reference': {
            'revision': 'cb168525138fecc792cc393f94afc39582b0103c',
            'sources': [source(Path('/home/mike/Projects/blender-cycles-trace-5.2/intern/cycles') / name)
                        for name in ('kernel/svm/closure.h', 'scene/shader.tables')],
            'scope': '96 executions of the original HIP SVM interpreter, not a CPU shader evaluator'},
        'regressions': {'structural_configurations': 8, 'families_per_configuration': 4,
            'before_failed_configurations': 8, 'runtime_states': 128,
            'runtime_scope': '96 original states plus 32 no-storage controls; both visibilities, four caustic settings, capacities 0/1/4; exact metadata and END/PC; only live fields observed',
            'float_comparison': 'finite abs-relative tolerance 2e-5, no production floating algorithm changes',
            'sources': [source(evidence / name) for name in (
                'formal-analysis.md', 'structure-red.log', 'structure-green.log',
                'dispatch-green.log', 'guards-hip-final.log', 'original-state-final.txt',
                'oracle-build-final.log', 'oracle-run-final.log', 'probe.cpp')],
            'fixture': source(fixture),
            'observer': source(ROOT / 'tools/cycles_svm_closure_guards_oracle.hip')},
        'profiles': {'source': source(evidence / 'profiles.json'), 'data': profiles},
        'static_word_mix': {'source': source(evidence / 'word-mix.json'), 'data': word_mix},
        'gates': {'host': gate(evidence / 'full-host.log', 171),
            'hip': gate(evidence / 'full-hip.log', 183),
            'focused_hip': gate(evidence / 'focused-hip.log', 20),
            'fallback': gate(evidence / 'full-fallback.log', 185),
            'native_vulkan': {**gate(evidence / 'native-vk.log', 4),
                'modules': modules, 'dxc_dxil_loaded': False},
            'build': source(evidence / 'build-final.log')},
        'canaries': campaign(evidence / 'canaries/canaries.json', True),
        'visual_qa': {'inspected': [str(evidence / 'canaries' / (scene + '-fixed-1-triptychs/combined.png'))
            for scene in ('barbershop', 'monk', 'monster', 'classroom')],
            'scope': 'four first-run Combined triptychs viewed; residual pass differences unresolved'},
        'execution_incident': {'source': source(evidence / 'dump-cwd-incident.md'),
            'scope': 'restored A LLVM dump overwrote 45 pre-existing untracked root IR files; new outputs archived; original root contents not recovered; user notified'},
        'open': ['Barbershop per-surface cost and residual DiffInd/shadow work',
            'remaining input/closure access and code-generation differences',
            '164 used-shader resource binding identities',
            'unimplemented SVM nodes and remaining legacy displacement path',
            'existing native Vulkan f16/f64 remainder failures',
            'remaining general restructure_cfg proof obligations'],
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
