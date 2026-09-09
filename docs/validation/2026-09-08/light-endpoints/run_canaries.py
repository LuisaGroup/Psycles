"""Re-run the four original high-resolution scenes with frozen binaries.

Uses reviewed equal-pass benchmark manifests, preserving their commands and
references. This is not a new paired Cycles timing campaign.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
import compare_cycles as compare
import render_pass_contract
import run_scene_benchmark as benchmark

parser = argparse.ArgumentParser()
parser.add_argument('baseline', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--actual-label', default='Psycles light endpoint fix HIP')
parser.add_argument('--socket-control-root', type=Path,
                    help='validated new socket metadata with the baseline geometry/images')
parser.add_argument('--keep-going-on-nonfinite', action='store_true',
                    help='record invalid pixels and finish the campaign; still exit nonzero')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
files = ['build/bin/psycles_render_blender_scene', 'build/libpsycles_core.so',
         'build/libpsycles_luisa_runtime.so', 'build/bin/libluisa-coro.so',
         'build/bin/libluisa-xir.so', 'build/bin/libluisa-backend-hip.so']
identity = {name: benchmark._sha256(ROOT / name) for name in files}
environment = os.environ.copy()
environment['PSYCLES_DISABLE_SHADER_CACHE'] = '1'
environment.pop('LUISA_CORO_WAVEFRONT_STATS', None)
records = []
for scene, repeat in [('barbershop', 1), ('monk', 1), ('monster', 1),
                      ('classroom', 1), ('barbershop', 2), ('barbershop', 3)]:
    manifest_path = args.baseline / scene / 'run-1/benchmark.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['schema'] == 'psycles.scene-benchmark.v3'
    assert manifest['status'] == 'complete'
    bundle = Path(manifest['scene']['bundle'])
    assert benchmark._bundle_matches_manifest(manifest, bundle)
    assert benchmark._sha256(Path(manifest['scene']['blend'])) == manifest['scene']['sha256']
    socket_control = None
    if args.socket_control_root:
        controlled = args.socket_control_root / scene
        socket_control = json.loads((args.socket_control_root / f'{scene}-control.json').read_text())
        assert socket_control['schema'] == 'psycles.socket-metadata-control.v1'
        assert Path(socket_control['old_bundle']).resolve() == bundle.resolve()
        assert Path(socket_control['control_bundle']).resolve() == controlled.resolve()
        assert benchmark._sha256(bundle / 'scene.json') == socket_control['old_scene']['sha256']
        assert benchmark._sha256(controlled / 'scene.json') == socket_control['control_scene']['sha256']
        assert benchmark._sha256(controlled / 'geometry.bin') == socket_control['old_geometry']['sha256']
        bundle = controlled
    prefix = args.output / f'{scene}-fixed-{repeat}'
    actual = prefix.with_suffix('.exr')
    assert not actual.exists() and not prefix.with_suffix('.log').exists()
    command = list(manifest['commands']['psycles_hip-wavefront-staged']['command'])
    command[1] = str(bundle)
    command[2] = str(prefix.with_suffix('.ppm'))
    assert command[6] == command[13] == '256'
    render = benchmark._run_logged(command, prefix.with_suffix('.log'),
                                   environment=environment, echo_output=False)
    frame = re.search(r'path coroutine: subroutines=(\d+) frame_fields=(\d+) frame_bytes=(\d+)',
                      render['output'])
    assert frame is not None
    pixels, channels = compare._read_image(actual)
    assert len(channels) == 46
    all_finite = bool(np.isfinite(pixels).all())
    assert all_finite or args.keep_going_on_nonfinite, actual
    del pixels
    reference = Path(manifest['renderers']['cycles']['hip']['output'])
    metadata = reference.with_suffix('.json')
    assert benchmark._can_resume_render(manifest, command_key='cycles_hip',
        renderer_group='cycles', renderer_key='hip',
        expected_command=manifest['commands']['cycles_hip']['command'],
        expected_output=reference, required_timings=('render_seconds',), metadata_path=metadata)
    report_path = prefix.with_name(prefix.name + '-comparison.json')
    comparison = benchmark._run_logged(benchmark._comparison_command(
        Path(sys.executable), ROOT / 'tools/compare_cycles.py', reference, actual,
        report_path, args.output / f'{prefix.name}-triptychs', metadata, bundle / 'scene.json',
        reference_label='Cycles 5.2.1 HIP', actual_label=args.actual_label),
        args.output / f'{prefix.name}-compare.log', echo_output=False)
    report = json.loads(report_path.read_text())
    assert set(report['passes']) == set(render_pass_contract.PASSES)
    assert all_finite == all(p['actual_invalid_pixels'] == 0 for p in report['passes'].values())
    records.append({'scene': scene, 'repeat': repeat,
                    'socket_metadata_control': socket_control,
                    'timings': benchmark._parse_psycles_timings(render['output']),
                    'render': benchmark._public_process_record(render),
                    'comparison': benchmark._public_process_record(comparison),
                    'source_manifest': str(manifest_path),
                    'source_manifest_sha256': benchmark._sha256(manifest_path),
                    'output_sha256': benchmark._sha256(actual),
                    'reference_sha256': benchmark._sha256(reference),
                    'frame': dict(zip(['stages', 'fields', 'bytes'], map(int, frame.groups()))),
                    'pass_contract': render_pass_contract.inspect_image(actual),
                    'actual_all_finite': all_finite, 'passes': report['passes']})
    for name, expected in identity.items():
        assert benchmark._sha256(ROOT / name) == expected, name
    (args.output / 'canaries.json').write_text(json.dumps({
        'schema': 'psycles.light-endpoint-canaries.v1',
        'actual_label': args.actual_label,
        'main_shader_cache': 'disabled', 'implementation_sha256': identity,
        'nonfinite_policy': 'record and exit nonzero' if args.keep_going_on_nonfinite else 'stop',
        'reference_scope': 'retained equal-pass Cycles run-1 images; not fresh timing pairs',
        'records': records}, indent=2) + '\n')
    print(scene, repeat, records[-1]['timings'], flush=True)
if not all(row['actual_all_finite'] for row in records):
    print('Campaign finished with non-finite actual pixels; numerical gate is NOT green.', flush=True)
    raise SystemExit(2)
