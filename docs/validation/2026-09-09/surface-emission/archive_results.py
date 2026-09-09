"""Archive native surface-emission control, original GPU film and scene gates."""
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
    assert profiles['schema'] == 'psycles.surface-emission-profiles.v1'
    assert '0/4' in (e / 'structure-red-complete.log').read_text()
    assert '4/4' in (e / 'structure-final.log').read_text()
    fixture = ROOT / 'tests/data/cycles_surface_emission.txt'
    assert fixture.read_bytes() == (e / 'original-gpu-film-camera.txt').read_bytes()
    assert len(fixture.read_text().splitlines()) == 81
    native = (e / 'native-vk.log').read_text()
    assert all(flag in native for flag in (
        'LUISA_VULKAN_USE_XIR=1', 'LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1',
        'LUISA_VULKAN_DISABLE_DXC=1'))
    assert not re.search(r'(?:find library=|calling init:).*lib(?:dxcompiler|dxil|dxc)', native, re.I)
    modules = len(re.findall(r'^\d+: .*SPIR-V compilation successful', native, re.M))
    assert modules > 0
    canaries = campaign(e / 'canaries/canaries.json', True)
    assert canaries['data']['implementation_sha256'] == profiles['runs']['guarded_repeat']['command']['implementation_sha256']
    result = {
        'schema': 'psycles.surface-emission.v1',
        'implementation': {'parent': args.parent, 'child': args.child,
            'before_parent': '6cd48f1f', 'before_runtime': 'ff8385c1',
            'before_child': 'da8fff856',
            'change': 'native emission/exit predicate encloses forward surface MIS and film atomics; retain runtime flags without NEE or trace',
            'unchanged': 'SVM words, typed payloads, addressing, PC, closure allocation, feature masks, RNG, inlining/launch policy, fastmath and direct/indirect split-film representation'},
        'reference': {'revision': 'cb168525138fecc792cc393f94afc39582b0103c',
            'sources': [source(Path('/home/mike/Projects/blender-cycles-trace-5.2/intern/cycles') / p) for p in (
                'kernel/integrator/shade_surface.h', 'kernel/integrator/surface_shader.h',
                'kernel/film/light_passes.h', 'kernel/film/write.h', 'scene/integrator.cpp')],
            'scope': 'original HIP surface_shader_emission and film_write_surface_emission, 80 boundary states plus unity-throughput camera; per-lane native atomic counts are captured',
            'limitation': 'oracle entry eligibility is supplied by the observer from the directly audited native integrate_surface predicate; the observer does not execute that whole integrator function',
            'arithmetic': 'fastmath enabled; 2e-6 absolute per-component film tolerance; no CPU shader/film reference'},
        'regressions': {'ast_configurations': 4, 'red_failed': 4, 'fixed_failed': 0,
            'ast_scope': 'all emission-operation atomics and the forward MIS callable are lexically guarded by the emission and exit state; not an instruction-count limit',
            'operation_gpu_cases': 320, 'camera_renderer_configurations': 4,
            'camera_scope': 'native SVM emission triangle, NEE on/off without trace, megakernel and staged coroutine; original unity-throughput camera film is the oracle',
            'source_files': [source(ROOT / p) for p in (
                'src/luisa/path_kernel_surface_emission.cpp', 'src/luisa/path_kernel_surface_emission.h',
                'src/luisa/path_kernel_surface_shading.cpp', 'tests/test_cycles_surface_emission_structure.cpp',
                'tests/test_luisa_cycles_surface_emission.cpp', 'tests/cycles_surface_emission_test_support.h',
                'tests/cycles_surface_emission_fixture.h', 'tests/data/cycles_surface_emission.txt',
                'tools/cycles_surface_emission_oracle.hip')],
            'evidence': [source(e / p) for p in (
                'formal-analysis.md', 'report-plan.md', 'structure-red-complete.log', 'structure-final.log',
                'film-before-hip.log', 'original-gpu-film-camera.txt', 'oracle-camera-build.log',
                'restored-binary-verification.json', 'restore-fixed-verification.log', 'production-forward.patch',
                'original-user-sources.sha256', 'original-status.log',
                'native-vk-device-lifetime-red.log', 'fallback-early-start-aborted.log')]},
        'profiles': {'source': source(e / 'profiles.json'), 'data': profiles},
        'gates': {'build': source(e / 'build-restored-fixed.log'),
            'host': gate(e / 'full-host.log', 175),
            'hip': gate(e / 'full-hip.log', 186),
            'fallback': gate(e / 'full-fallback.log', 188),
            'final_test_device_lifetime_hip': gate(e / 'final-test-hip.log', 1),
            'final_test_device_lifetime_fallback': gate(e / 'final-test-fallback.log', 1),
            'native_vulkan': {**gate(e / 'native-vk.log', 4),
                'native_compilations': modules, 'dxc_dxil_loaded': False}},
        'canaries': canaries,
        'visual_qa': {'inspected': [str(e / 'canaries' / (n + '-fixed-1-triptychs/combined.png'))
            for n in ('barbershop', 'monk', 'monster', 'classroom')],
            'scope': 'four first-run Combined triptychs viewed, all 15 numerical pass comparisons retained'},
        'open': ['native direct/indirect film address selection instead of two zero-selected contributions',
            'remaining Barbershop per-surface cost and DiffInd/shadow path surplus',
            'per-ray light inverse reconstruction and remaining scene-owned fields',
            'pre-existing native Vulkan area sampling and f16/f64 remainder gates',
            'remaining semantic opcodes, private displacement bridge and generic CFG proof obligations'],
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
