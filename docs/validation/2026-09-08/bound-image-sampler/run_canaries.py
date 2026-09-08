import json
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path('/home/mike/Projects/Psycles-surface-svm')
BASE = Path('/var/tmp/psycles-matched-pass-hip-2c98Q0')
OUT = Path('/var/tmp/psycles-native-volume-svm-06XnDX')
sys.path.insert(0, str(ROOT / 'tools'))
import compare_cycles as compare
import render_pass_contract
import run_scene_benchmark as benchmark

files = ['build/bin/psycles_render_blender_scene', 'build/libpsycles_core.so',
         'build/libpsycles_luisa_runtime.so', 'build/bin/libluisa-coro.so',
         'build/bin/libluisa-xir.so', 'build/bin/libluisa-backend-hip.so']
identity = {name: benchmark._sha256(ROOT / name) for name in files}
environment = os.environ.copy()
environment['PSYCLES_DISABLE_SHADER_CACHE'] = '1'
records = []
for scene, repeat in [('barbershop', 1), ('monk', 1), ('monster', 1),
                      ('classroom', 1), ('barbershop', 2), ('barbershop', 3)]:
    manifest_path = BASE / scene / 'run-1/benchmark.json'
    m = json.loads(manifest_path.read_text())
    assert m['schema'] == 'psycles.scene-benchmark.v3' and m['status'] == 'complete'
    bundle = Path(m['scene']['bundle'])
    assert benchmark._bundle_matches_manifest(m, bundle)
    assert benchmark._sha256(Path(m['scene']['blend'])) == m['scene']['sha256']
    prefix = OUT / f'{scene}-bound-sampler-{repeat}'
    actual = prefix.with_suffix('.exr')
    assert not actual.exists() and not prefix.with_suffix('.ppm').exists()
    command = list(m['commands']['psycles_hip-wavefront-staged']['command'])
    command[2] = str(prefix.with_suffix('.ppm'))
    render = benchmark._run_logged(command, prefix.with_suffix('.log'),
                                   environment=environment, echo_output=False)
    timing = benchmark._parse_psycles_timings(render['output'])
    frame = re.search(r'path coroutine: subroutines=(\d+) frame_fields=(\d+) frame_bytes=(\d+)',
                      render['output'])
    assert frame is not None
    inventory = render_pass_contract.inspect_image(actual)
    pixels, channels = compare._read_image(actual)
    assert len(channels) == 46 and np.isfinite(pixels).all()
    del pixels
    reference = Path(m['renderers']['cycles']['hip']['output'])
    metadata = reference.with_suffix('.json')
    assert benchmark._can_resume_render(m, command_key='cycles_hip',
        renderer_group='cycles', renderer_key='hip',
        expected_command=m['commands']['cycles_hip']['command'],
        expected_output=reference, required_timings=('render_seconds',), metadata_path=metadata)
    report_path = OUT / f'{prefix.name}-report.json'
    comparison = benchmark._run_logged(benchmark._comparison_command(
        Path(sys.executable), ROOT / 'tools/compare_cycles.py', reference, actual,
        report_path, OUT / f'{prefix.name}-triptychs', metadata, bundle / 'scene.json',
        reference_label='Cycles 5.2.1 HIP', actual_label='Psycles bound sampler HIP'),
        OUT / f'{prefix.name}-compare.log', echo_output=False)
    report = json.loads(report_path.read_text())
    assert set(report['passes']) == set(render_pass_contract.PASSES)
    assert all(p['actual_invalid_pixels'] == 0 for p in report['passes'].values())
    records.append({'scene': scene, 'repeat': repeat, 'timings': timing,
                    'render': benchmark._public_process_record(render),
                    'comparison': benchmark._public_process_record(comparison),
                    'source_manifest': str(manifest_path),
                    'source_manifest_sha256': benchmark._sha256(manifest_path),
                    'output_sha256': benchmark._sha256(actual),
                    'reference_sha256': benchmark._sha256(reference),
                    'frame': dict(zip(['stages', 'fields', 'bytes'], map(int, frame.groups()))),
                    'pass_contract': inventory, 'actual_all_finite': True,
                    'passes': report['passes']})
    for name, expected in identity.items():
        assert benchmark._sha256(ROOT / name) == expected, name
    (OUT / 'bound-sampler-canaries.json').write_text(json.dumps({
        'schema': 'psycles.bound-sampler-canaries.v1', 'baseline': 'c5bf9247',
        'main_shader_cache': 'disabled', 'implementation_sha256': identity,
        'reference_scope': 'unchanged original equal-pass Cycles run-1 images, not fresh pairs',
        'records': records}, indent=2) + '\n')
    print(scene, repeat, timing, flush=True)
