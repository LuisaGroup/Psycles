"""Archive original light tables, GPU states, intervention and complete gates."""
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
    e = args.evidence
    profiles = json.loads((e / 'profiles.json').read_text())
    assert profiles['schema'] == 'psycles.scene-light-parameter-profiles.v1'
    assert set(profiles['runs']) == {'before', 'prepared_first', 'prepared_repeat', 'restored'}
    assert '0/7' in (e / 'structure-red.log').read_text()
    assert '7/7' in (e / 'structure-final.log').read_text()
    for fixture, capture in (
        ('cycles_light_parameters_state.txt', 'original-gpu-conditioned-state.txt'),
        ('cycles_light_parameters_state_ftz.txt', 'original-gpu-conditioned-ftz-state.txt')):
        f = ROOT / 'tests/data' / fixture
        assert f.read_bytes() == (e / capture).read_bytes()
        assert len(f.read_text().splitlines()) == 296
    native = (e / 'native-vk-mode.log').read_text()
    assert all(flag in native for flag in (
        'LUISA_VULKAN_USE_XIR=1', 'LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1',
        'LUISA_VULKAN_DISABLE_DXC=1'))
    assert not re.search(r'(?:find library=|calling init:).*lib(?:dxcompiler|dxil|dxc)', native, re.I)
    assert 'failures=36' in native
    # CTest reprints failing-test output. Count the original verbose stream,
    # whose lines carry the numeric test prefix, not the repeated log body.
    modules = len(re.findall(r'^\d+: .*SPIR-V compilation successful', native, re.M))
    assert modules > 0
    def mismatch_rows(text):
        return {(int(m[1]), int(m[2])): {'actual_printed': m[3], 'native_printed': m[4]}
            for m in re.finditer(r'row=(\d+) light=\d+ lane=(\d+) actual=(\S+) native=(\S+)', text)}
    baseline = mismatch_rows((e / 'baseline-vk.log').read_text())
    residual = mismatch_rows(native)
    assert len(baseline) == 183 and len(residual) == 36
    assert all(baseline.get(key) == value for key, value in residual.items())
    canaries = campaign(e / 'canaries/canaries.json', True)
    assert (canaries['data']['implementation_sha256'] ==
            profiles['runs']['prepared_repeat']['command']['implementation_sha256'])
    native_root = Path('/home/mike/Projects/blender-cycles-trace-5.2/intern/cycles')
    for name in ('light', 'svm'):
        assert (native_root / f'scene/{name}.cpp').read_bytes() == (e / f'original-{name}.cpp').read_bytes()
    result = {
        'schema': 'psycles.scene-light-parameters.v1',
        'implementation': {'parent': args.parent, 'child': args.child,
            'before_parent': '124cee8c', 'before_child': 'da8fff856',
            'changes': ['native host-prepared area/spot parameters',
                'device consumers use those fields and original predicates',
                'host-only source IDs preserve light-tree authored-angle access'],
            'unchanged': 'SVM word images, addressing, PC, feature masks, inlining/launch policy, fast math and allocation analysis'},
        'original_reference': {'revision': 'cb168525138fecc792cc393f94afc39582b0103c',
            'sources': [source(native_root / name) for name in (
                'scene/light.cpp', 'kernel/light/area.h', 'kernel/light/spot.h',
                'util/math_base.h', 'util/math_intersect.h', 'util/simd.h', 'bvh/embree.cpp')],
            'scope': '37 observed KernelLight tables (25 boundary, 12 actual Barbershop) and 296 direct original GPU consumer states per denormal mode; no CPU light evaluator',
            'mode': 'preserve plus separately compiled original GPU FTZ fixture; independent device normal/signed/zero/subnormal controls select the corresponding arithmetic environment',
            'coverage_limitations': ['all captured lamps have normalize=true',
                'Blender clamps authored zero spot angle to 0.017453292',
                'ill-conditioned transverse narrow-area cone roots and zero-blend exact-edge NaNs remain diagnostic-only'],
            'observer_artifacts': [source(e / name) for name in (
                'boundary-parameters.json', 'barbershop-parameters.json', 'boundary.blend',
                'observer-hooks.patch', 'build-original-observer.log', 'build-original-restored.log')]},
        'regressions': {'ast_consumers': 7, 'before_failed': 7, 'after_failed': 0,
            'scope': 'absence of scene-owned TAN/COS in seven production-AST consumers, not an instruction-count ceiling',
            'float_comparison': 'host 8e-6 relative plus exact classifications; device 2e-4 abs-relative and exact predicates, interval endpoints normalized to the authored ten-unit segment',
            'sources': [source(ROOT / name) for name in (
                'tests/test_cycles_light_parameter_structure.cpp', 'tests/test_luisa_cycles_light_parameters.cpp',
                'tests/cycles_light_parameters_fixture.h', 'tools/cycles_light_parameters_oracle.hip',
                'tools/cycles_light_parameters_oracle.h', 'tools/cycles_light_parameters_oracle.patch',
                'tools/pack_cycles_light_parameters.py', 'tools/create_cycles_light_parameters_oracle.py',
                'tests/data/cycles_light_parameters.txt', 'tests/data/cycles_light_parameters_state.txt',
                'tests/data/cycles_light_parameters_state_ftz.txt')],
            'diagnostics': [source(e / name) for name in (
                'formal-analysis.md', 'structure-red.log', 'structure-final.log',
                'original-gpu-unfused-state.txt', 'oblique-interval-inputs.h',
                'oblique-interval-state.txt', 'native-vk.log', 'native-vk-mode.log',
                'baseline-vk.log', 'baseline-runtime-test.cpp', 'hip-parallel-aborted.log')]},
        'profiles': {'source': source(e / 'profiles.json'), 'data': profiles},
        'gates': {'build': source(e / 'build-restored-final.log'),
            'host': gate(e / 'full-host-final.log', 174),
            'hip': gate(e / 'full-hip-final.log', 185),
            'fallback': gate(e / 'full-fallback-final.log', 187),
            'mode_controls_hip': gate(e / 'mode-hip.log', 1),
            'mode_controls_fallback': gate(e / 'mode-fallback.log', 1),
            'native_vulkan': {**gate(e / 'native-vk-mode.log', 6, failed=1),
                'native_compilations': modules, 'dxc_dxil_loaded': False,
                'remaining_light_lane_failures': 36,
                'baseline_light_lane_failures': 183,
                'residual_already_in_baseline': [{'row': key[0], 'lane': key[1], **value}
                    for key, value in sorted(residual.items())],
                'printed_comparison_scope': 'six significant digits, not bit equality'}},
        'canaries': canaries,
        'visual_qa': {'inspected': [str(e / 'canaries' / (name + '-fixed-1-triptychs/combined.png'))
                for name in ('barbershop', 'monk', 'monster', 'classroom')],
            'scope': 'four first-run Combined triptychs viewed; all 15 numeric pass comparisons retained'},
        'open': ['Barbershop per-surface cost and residual DiffInd/shadow work',
            'per-ray light transform inverse and remaining scene-owned light fields',
            'native Vulkan area rectangle sampling numeric differences',
            'existing native Vulkan f16/f64 remainder failures',
            'remaining semantic nodes, private legacy displacement bridge and generic CFG proof obligations'],
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
