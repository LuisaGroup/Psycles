import json
import os
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path('/home/mike/Projects/Psycles-surface-svm')
BASE = pathlib.Path('/var/tmp/psycles-four-scene-hip-0PRq34')
OUT = pathlib.Path('/var/tmp/psycles-matched-pass-hip-2c98Q0')
sys.path.insert(0, str(ROOT / 'tools'))
import compare_cycles as compare
import run_scene_benchmark as benchmark

files = [
    'build/bin/psycles_render_blender_scene', 'build/libpsycles_core.so',
    'build/libpsycles_luisa_runtime.so', 'build/bin/libluisa-coro.so',
    'build/bin/libluisa-xir.so', 'build/bin/libluisa-backend-hip.so',
    'tools/run_scene_benchmark.py', 'tools/render_cycles_golden.py',
    'tools/render_pass_contract.py', 'tools/compare_cycles.py',
]
identity = {name: benchmark._sha256(ROOT / name) for name in files}
environment = os.environ.copy()
environment['PSYCLES_DISABLE_SHADER_CACHE'] = '1'
records = []
orders = [
    ['barbershop', 'monk', 'monster', 'classroom'],
    ['classroom', 'monster', 'monk', 'barbershop'],
    ['monster', 'barbershop', 'classroom', 'monk'],
]
for repeat, scenes in enumerate(orders, 1):
    for scene in scenes:
        original = json.loads((BASE / scene / 'run-1/benchmark.json').read_text())
        bundle = BASE / scene / 'export'
        assert benchmark._bundle_matches_manifest(original, bundle), scene
        blend = pathlib.Path(original['scene']['blend'])
        assert benchmark._sha256(blend) == original['scene']['sha256'], scene
        target = OUT / scene / f'run-{repeat}'
        assert not target.exists(), f'will not overwrite {target}'
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(ROOT / 'tools/run_scene_benchmark.py'),
            '--blender', '/home/mike/Projects/blender-install-5.2-hiprt/blender',
            '--psycles-render', str(ROOT / 'build/bin/psycles_render_blender_scene'),
            '--blend', str(blend), '--output-dir', str(target),
            '--bundle', str(bundle), '--reuse-export', '--skip-cycles-cpu',
            '--cycles-gpu-device', 'HIP', '--cycles-gpu-device-name', 'Radeon RX 9070 XT',
            '--psycles-backends', 'hip', '--psycles-schedulers', 'wavefront-staged',
            '--width', str(original['settings']['width']),
            '--height', str(original['settings']['height']),
            '--samples', '256', '--max-samples-per-dispatch', '64',
            '--wavefront-execution-block-size', '32', '--persistent-worker-count', '32768',
            '--persistent-block-size', '32', '--persistent-fetch-size', '1',
            '--staged-surface-sorting', '--staged-direct-light-queue',
            '--wavefront-counter-readback-batch-size', '4',
            '--wavefront-counter-readback-pipeline-depth', '2',
            '--wavefront-tail-megakernel-threshold', '0', '--wavefront-graph-worker-count', '0',
            '--no-wavefront-graph-selective', '--wavefront-graph-refill-threshold', '0',
            '--fast-math', '--wavefront-frame-capacity', '1048576',
        ]
        process = benchmark._run_logged(command, OUT / f'{scene}-{repeat}-runner.log',
                                        environment=environment, echo_output=False)
        manifest_path = target / 'benchmark.json'
        manifest = json.loads(manifest_path.read_text())
        assert manifest['schema'] == 'psycles.scene-benchmark.v3'
        assert manifest['status'] == 'complete', manifest_path
        actual_record = manifest['renderers']['psycles']['hip-wavefront-staged']
        actual, channels = compare._read_image(pathlib.Path(actual_record['output']))
        assert len(channels) == 46 and np.isfinite(actual).all(), scene
        del actual
        log = pathlib.Path(manifest['commands']['psycles_hip-wavefront-staged']['log']).read_text()
        frame = re.search(r'path coroutine: subroutines=(\d+) frame_fields=(\d+) frame_bytes=(\d+)', log)
        assert frame is not None, scene
        records.append({
            'scene': scene, 'repeat': repeat, 'manifest': str(manifest_path),
            'manifest_sha256': benchmark._sha256(manifest_path),
            'process': benchmark._public_process_record(process),
            'frame': dict(zip(['stages', 'fields', 'bytes'], map(int, frame.groups()))),
            'cycles_render_seconds': manifest['renderers']['cycles']['hip']['render_seconds'],
            'psycles_timings': {name: actual_record[name] for name in
                               ['scene_compile_seconds', 'shader_jit_seconds', 'render_seconds']},
            'actual_channels': len(channels), 'actual_all_finite': True,
        })
        for name, expected in identity.items():
            assert benchmark._sha256(ROOT / name) == expected, f'implementation changed: {name}'
        (OUT / 'campaign.json').write_text(json.dumps({
            'schema': 'psycles.matched-pass-campaign.v1',
            'pass_contract': benchmark.render_pass_contract.SCHEMA,
            'orders': orders, 'renderer_order': ['Cycles HIP', 'Psycles HIP'],
            'implementation_sha256': identity,
            'main_shader_cache': 'disabled', 'records': records,
        }, indent=2) + '\n')
        print(json.dumps(records[-1]), flush=True)
