"""Audit exact queue cardinalities and original-image differences.

GPU timings are deliberately not inferred from these host logging runs.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
import compare_cycles as compare
import render_pass_contract
import run_scene_benchmark as benchmark


def source(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest}


def cycles_counts(path):
    counts = defaultdict(lambda: {'dispatches': 0, 'submitted': 0})
    rows = re.findall(r'GPU queue launch (\w+), work_size (\d+)', path.read_text())
    assert rows, 'Cycles TRACE queue logging is required'
    for name, work in rows:
        counts[name]['dispatches'] += 1
        counts[name]['submitted'] += int(work)
    return dict(sorted(counts.items()))


def psycles_counts(path):
    text = path.read_text()
    blocks = text.split('Wavefront stats: ')[1:]
    assert blocks, 'LUISA_CORO_WAVEFRONT_STATS=1 is required'
    totals = defaultdict(lambda: {'dispatches': 0, 'submitted': 0})
    records = []
    for block in blocks:
        summary = dict((name, int(value)) for name, value in re.findall(
            r'(iterations|generated|resumed|extensions)=(\d+)', block.split('\n')[0]))
        nodes = re.findall(r"Wavefront continuation: index=(\d+) token=(\d+) name='([^']*)' "
                           r'dispatches=(\d+) executed=(\d+) peak_queued=(\d+)', block)
        aux = re.findall(r"Wavefront auxiliary: name='([^']*)' dispatches=(\d+) "
                         r'executed=(\d+) peak_queued=(\d+)', block)
        extensions = re.findall(r"Wavefront Extension: queue=(\d+) boundary=(\d+) extension=(\d+) "
                                r"schema='([^']*)' handler='([^']*)' dispatches=(\d+) "
                                r'executed=(\d+) peak_queued=(\d+)', block)
        assert sum(int(n[4]) for n in nodes if int(n[0]) != 0) == summary['resumed']
        assert sum(int(n[4]) for n in nodes if int(n[0]) == 0) == summary['generated']
        assert sum(int(n[6]) for n in extensions) == summary['extensions']
        for name, dispatches, executed, peak in [*(n[2:] for n in nodes), *aux]:
            totals[name]['dispatches'] += int(dispatches)
            totals[name]['submitted'] += int(executed)
        records.append({'summary': summary, 'continuations': nodes,
                        'auxiliary': aux, 'extensions_not_added_again': extensions})
    frame = re.search(r'path coroutine: subroutines=(\d+) frame_fields=(\d+) frame_bytes=(\d+)', text)
    assert frame is not None
    return {'stages': dict(sorted(totals.items())), 'dispatches': records,
             'generated': sum(r['summary']['generated'] for r in records),
             'frame': dict(zip(['stages', 'fields', 'bytes'], map(int, frame.groups())))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-images', action='store_true')
    args = parser.parse_args()
    directory = args.evidence
    metadata = json.loads((directory / 'cycles.json').read_text())
    assert metadata['blender_build']['build_hash'] == '9e2066aef7ef'
    assert metadata['cycles_compute_device_type'] == 'HIP'
    assert metadata['samples'] == 64 and not metadata['adaptive_sampling']
    expected_paths = metadata['width'] * metadata['height'] * metadata['samples']
    c = cycles_counts(directory / 'cycles.log')
    # This scene runs the camera volume-stack stage once per primary path.
    # INIT_FROM_CAMERA work_size instead includes per-tile padding, and must
    # not be substituted for the active generator cardinality.
    assert c['integrator_intersect_volume_stack']['submitted'] == expected_paths
    result = {'schema': 'psycles.light-endpoint-work.v1',
              'scope': 'exact host queue counts, not GPU kernel time or occupied lanes',
              'extent': [metadata['width'], metadata['height']],
              'samples': metadata['samples'], 'primary_paths': expected_paths,
              'cycles_stages': c, 'psycles': {}, 'comparison': {}, 'sources': {}}
    for filename in ['cycles.log', 'cycles.json', 'cycles.exr', 'psycles.log',
                     'psycles.exr', 'psycles-fixed.log', 'psycles-fixed.exr']:
        result['sources'][filename] = source(directory / filename)
    for label, name in [('before', 'psycles'), ('after', 'psycles-fixed')]:
        work = psycles_counts(directory / f'{name}.log')
        assert work['generated'] == expected_paths
        result['psycles'][label] = work
        result['comparison'][label] = {}
        for stage, value in work['stages'].items():
            if stage == '<entry>':
                continue
            reference = c['integrator_' + stage]['submitted']
            count = value['submitted']
            result['comparison'][label][stage] = {
                'cycles': reference, 'psycles': count,
                'difference': count - reference,
                'relative_difference': count / reference - 1.0}
        # Current Psycles caches the mesh hit while visiting lamp endpoints.
        # Report this known routing difference; do not silently relabel its
        # continuation count as Cycles' post-lamp geometry traversal count.
        adjusted = (work['stages']['intersect_closest']['submitted'] +
                    work['stages']['shade_light_forward']['submitted'])
        result['comparison'][label]['closest_plus_lamp_visits'] = {
            'cycles': c['integrator_intersect_closest']['submitted'],
            'psycles_counterfactual': adjusted,
            'difference': adjusted - c['integrator_intersect_closest']['submitted']}
        if not args.skip_images:
            actual = directory / f'{name}.exr'
            pixels, channels = compare._read_image(actual)
            assert len(channels) == 46 and np.isfinite(pixels).all()
            del pixels
            work['actual_all_channels_finite'] = True
            work['pass_contract'] = render_pass_contract.inspect_image(actual)
            comparison_path = directory / f'{name}-comparison.json'
            command = benchmark._comparison_command(
                Path(sys.executable), ROOT / 'tools/compare_cycles.py',
                directory / 'cycles.exr', actual, comparison_path,
                directory / f'{name}-triptychs', directory / 'cycles.json',
                Path('/var/tmp/psycles-four-scene-hip-0PRq34/barbershop/export/scene.json'),
                reference_label='Cycles 5.2.1 HIP', actual_label=f'Psycles {label} endpoint fix')
            if comparison_path.exists():
                raise FileExistsError(f'Preserve the prior comparison: {comparison_path}')
            process = benchmark._run_logged(command, directory / f'{name}-compare.log', echo_output=False)
            report = json.loads(comparison_path.read_text())
            assert set(report['passes']) == set(render_pass_contract.PASSES)
            work['passes'] = report['passes']
            work['comparison_command'] = benchmark._public_process_record(process)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['comparison'], indent=2))


if __name__ == '__main__':
    main()
