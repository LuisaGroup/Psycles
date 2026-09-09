"""Archive original GPU film routing, permanent reds and complete scene gates."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'docs/validation/2026-09-08/lamp-routing-and-surface'))
from archive_validation import ORDER, gate
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
    assert profiles['schema'] == 'psycles.film-routing-profiles.v1'
    assert '0/5' in (e / 'structure-red.log').read_text()
    assert '5/5' in (e / 'structure-final.log').read_text()
    assert '351/540; native surface shadow state: 72/180' in (e / 'runtime-red-hip.log').read_text()
    assert '540/540; native surface shadow state: 180/180' in (e / 'runtime-fixed-hip.log').read_text()
    fixture = ROOT / 'tests/data/cycles_film_routing.txt'
    assert fixture.read_bytes() == (e / 'original-gpu-film.txt').read_bytes()
    assert len(fixture.read_text().splitlines()) == 240
    native = (e / 'native-vk.log').read_text()
    assert all(flag in native for flag in (
        'LUISA_VULKAN_USE_XIR=1', 'LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1',
        'LUISA_VULKAN_DISABLE_DXC=1'))
    assert not re.search(r'(?:find library=|calling init:).*lib(?:dxcompiler|dxil|dxc)', native, re.I)
    modules = len(re.findall(r'^\d+: .*SPIR-V compilation successful', native, re.M))
    assert modules > 0
    path = e / 'canaries-complete/canaries.json'
    data = json.loads(path.read_text())
    assert data['schema'] == 'psycles.light-endpoint-canaries.v1'
    assert data['main_shader_cache'] == 'disabled'
    assert data['nonfinite_policy'] == 'record and exit nonzero'
    assert [(r['scene'], r['repeat']) for r in data['records']] == ORDER
    failed_finite = []
    for row in data['records']:
        assert len(row['pass_contract']['channels']) == 46 and len(row['passes']) == 15
        assert row['render']['returncode'] == row['comparison']['returncode'] == 0
        record = row['socket_metadata_control']
        assert record['old_geometry']['sha256'] == record['control_geometry']['sha256']
        assert record['fresh_scene']['sha256'] == record['control_scene']['sha256']
        invalid = {name: p['actual_invalid_pixels'] for name, p in row['passes'].items()
                   if p['actual_invalid_pixels']}
        assert row['actual_all_finite'] == (not invalid)
        if invalid:
            failed_finite.append({'scene': row['scene'], 'repeat': row['repeat'], 'passes': invalid})
    canaries = {'source': source(path), 'data': data,
        'numerical_gate': {'all_actual_channels_finite': not failed_finite,
            'status': 'failed' if failed_finite else 'passed',
            'failed_runs': failed_finite,
            'policy': 'No pixel waiver or numeric tolerance relaxation; runner records all six and exits 2 if any actual invalid value is present.'}}
    tiny = e / 'tiny-weights-replayed/results.json'
    tiny_data = json.loads(tiny.read_text())
    assert tiny_data['schema'] == 'psycles.tiny-weight-fastmath-observation.v1'
    assert len(tiny_data['rows']) == 9
    assert canaries['data']['implementation_sha256'] == profiles['runs']['routed_repeat']['command']['implementation_sha256']
    integration = json.loads((e / 'upstream-integration-binary-verification.json').read_text())
    assert integration['match'] and integration['binaries'] == data['implementation_sha256']
    integration_vk = (e / 'integration-vk.log').read_text()
    assert all(flag in integration_vk for flag in (
        'LUISA_VULKAN_USE_XIR=1', 'LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1',
        'LUISA_VULKAN_DISABLE_DXC=1'))
    assert not re.search(r'(?:find library=|calling init:).*lib(?:dxcompiler|dxil|dxc)', integration_vk, re.I)
    integration_modules = len(re.findall(r'^\d+: .*SPIR-V compilation successful', integration_vk, re.M))
    assert integration_modules > 0
    result = {
        'schema': 'psycles.film-routing.v1',
        'implementation': {'parent': args.parent, 'child': args.child,
            'before_parent': 'f690ead6', 'before_runtime': '4f487ba5',
            'measured_child': 'da8fff856',
            'change': 'native shadow pass initialization and zero-only BSDF component ratio, captured-event pass classification and destination selection; remove six-RGB split-film value/callable',
            'unchanged': 'SVM words/payload/addressing/PC/closure/masks, RNG, inlining/register/launch policy, fastmath and static array analysis'},
        'reference': {'revision': 'cb168525138fecc792cc393f94afc39582b0103c',
            'sources': [source(Path('/home/mike/Projects/blender-cycles-trace-5.2/intern/cycles') / p) for p in (
                'kernel/integrator/shade_surface.h', 'kernel/integrator/shade_volume.h',
                'kernel/film/light_passes.h', 'kernel/film/write.h', 'util/math_float3.h')],
            'scope': 'original-source shadow pass assignment blocks extracted verbatim; original GPU BSDF weights, light clamp, film routing and per-lane atomic counts; 240 boundary states',
            'limitation': 'observer supplies boundary state and invokes extracted blocks plus original film functions, not a complete native light-sampling/traversal integrator',
            'arithmetic': 'native fastmath; finite 2e-6 absolute film/weight tolerance and exact discrete state; no CPU film or shader oracle'},
        'regressions': {'ast': {'configurations': 5, 'red_passed': 0, 'fixed_passed': 5,
                'scope': 'all pass atomic addresses depend on bounce; unreplicated writer sites match original maximum observed writes, including zero values; not actual GPU write counting'},
            'film': {'configurations': 540, 'red_passed': 351, 'fixed_passed': 540,
                'scope': 'actual forward, inline shadow, captured/deferred shadow and volume NEE sinks, serial/atomic; all 52 film lanes'},
            'surface_shadow_state': {'configurations': 180, 'red_passed': 72, 'fixed_passed': 180,
                'scope': 'actual flags/depth/visibility and diffuse/glossy weights; main path deliberately advanced before shadow film completion'},
            'source_files': [source(ROOT / p) for p in (
                'src/luisa/path_kernel_film_routing.cpp', 'src/luisa/path_kernel_shadow_film.cpp',
                'src/luisa/path_tracer_lighting.cpp', 'tests/test_cycles_film_routing_structure.cpp',
                'tests/test_luisa_cycles_film_routing.cpp', 'tests/cycles_film_routing_test_support.h',
                'tests/cycles_film_routing_fixture.h', 'tests/data/cycles_film_routing.txt',
                'tools/cycles_film_routing_oracle.hip', 'tools/capture_cycles_film_routing.py')],
            'evidence': [source(e / p) for p in (
                'formal-analysis.md', 'report-plan.md', 'structure-red.log', 'structure-final.log',
                'runtime-red-hip.log', 'runtime-fixed-hip.log', 'oracle-build.log',
                'film_routing_native.h', 'film_routing_native.json', 'original-gpu-film.txt',
                'unchanged-operation-extraction.patch', 'unchanged-shadow-film.cpp',
                'production-fixed.patch', 'restored-baseline-binary-verification.json',
                'restored-fixed-binary-verification.json', 'barbershop-volume-scope.log',
                'oracle-scene.sha256', 'original-status.log', 'original-user-sources.sha256')]},
        'attribution_limit': 'Current Barbershop has emission-only fog, no world volume and zero volume bounces. Mixed scattering-route defects cannot explain its DiffInd. Profiled film/weight work is a separate intervention.',
        'profiles': {'source': source(e / 'profiles.json'), 'data': profiles},
        'gates': {'build': source(e / 'build-restored-fixed.log'),
            'host': gate(e / 'full-host.log', 176), 'hip': gate(e / 'full-hip.log', 187),
            'fallback': gate(e / 'full-fallback.log', 189),
            'native_vulkan': {**gate(e / 'native-vk.log', 6),
                'native_compilations': modules, 'dxc_dxil_loaded': False}},
        'upstream_integration': {'scope': 'Metal-only/CI changes integrated before publication; Linux all-target build leaves all six measured HIP binaries byte-identical',
            'binary_verification': integration,
            'build': source(e / 'build-upstream-metal-integration.log'),
            'host': gate(e / 'integration-host.log', 176),
            'hip': gate(e / 'integration-hip.log', 6),
            'fallback': gate(e / 'integration-fallback.log', 6),
            'native_vulkan': {**gate(e / 'integration-vk.log', 6),
                'native_compilations': integration_modules, 'dxc_dxil_loaded': False}},
        'canaries': canaries,
        'tiny_weight_diagnostic': {'source': source(tiny), 'data': tiny_data,
            'interpretation': 'Original and actual dynamic GPU inputs reproduce the same non-finite weight classes under native fastmath. Consistent with Classroom separated-pass failures, not a per-path proof or an all-finite canary pass.'},
        'classroom_nonfinite': {'source': source(e / 'classroom-nonfinite.json'),
            'data': json.loads((e / 'classroom-nonfinite.json').read_text())},
        'initial_canary_attempt': {'source': source(e / 'canaries.log'),
            'data': json.loads((e / 'canaries/canaries.json').read_text()),
            'outcome': 'stopped at Classroom all-finite assertion; preserve the three completed preceding records, no timing selection'},
        'visual_qa': {'inspected': [str(e / 'canaries-complete' / (n + '-fixed-1-triptychs/combined.png'))
            for n in ('barbershop', 'monk', 'monster', 'classroom')],
            'scope': 'four first-run Combined triptychs viewed; all 15 numerical comparisons retained'},
        'open': ['remaining Barbershop surface cost and indirect/path divergence',
            'Classroom non-finite separated passes under native fastmath; complete-scene finite gate is not green',
            'scene-owned inverse light transforms and main-surface first-bounce weight work',
            'pre-existing native Vulkan area sampling and f16/f64 remainder failures',
            'remaining semantic opcodes, private displacement bridge and generic CFG proof obligations'],
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
