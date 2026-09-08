"""Validate the separately recorded Monster link-time repeat.

This does not rerender, discard the first run, or claim an isolated compiler
benchmark. The render command completed with exit status zero before this audit.
"""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

import analyze_work as work


def links(text):
    return [{'code_object_bytes': int(size), 'seconds': float(ms) / 1000}
            for size, ms in re.findall(
                r'Linked HIP LLVM bitcode to AMDGPU code object \((\d+) bytes\) in ([\d.]+) ms',
                text)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve previous evidence'
    directory = args.evidence
    campaign = json.loads((directory / 'canaries.json').read_text())
    first = next(r for r in campaign['records'] if r['scene'] == 'monster')
    for name, digest in campaign['implementation_sha256'].items():
        assert work.source(work.ROOT / name)['sha256'] == digest
    prefix = directory / 'monster-warm'
    actual = prefix.with_suffix('.exr')
    log_path = prefix.with_suffix('.log')
    text = log_path.read_text()
    assert 'Wavefront stats:' not in text
    assert 'Rendered 1080x1080 at 256 spp from absolute sample range [0, 256) of 256' in text
    command = list(first['render']['command'])
    command[2] = str(prefix.with_suffix('.ppm'))
    pixels, channels = work.compare._read_image(actual)
    assert len(channels) == 46 and np.isfinite(pixels).all()
    del pixels
    reference = Path(first['comparison']['command'][2])
    assert work.source(reference)['sha256'] == first['reference_sha256']
    manifest = json.loads(Path(first['source_manifest']).read_text())
    bundle = Path(manifest['scene']['bundle'])
    assert work.benchmark._bundle_matches_manifest(manifest, bundle)
    report_path = directory / 'monster-warm-comparison.json'
    assert not report_path.exists(), 'Preserve previous comparison'
    comparison = work.benchmark._run_logged(work.benchmark._comparison_command(
        Path(sys.executable), work.ROOT / 'tools/compare_cycles.py', reference, actual,
        report_path, directory / 'monster-warm-triptychs', reference.with_suffix('.json'),
        bundle / 'scene.json', reference_label='Cycles 5.2.1 HIP',
        actual_label='Psycles Monster warm link control'),
        directory / 'monster-warm-compare.log', echo_output=False)
    report = json.loads(report_path.read_text())
    assert set(report['passes']) == set(work.render_pass_contract.PASSES)
    assert all(p['actual_invalid_pixels'] == 0 for p in report['passes'].values())
    frame = re.search(r'path coroutine: subroutines=(\d+) frame_fields=(\d+) frame_bytes=(\d+)', text)
    assert frame is not None
    result = {
        'schema': 'psycles.light-endpoint-warm-control.v1',
        'scope': 'extra Monster repeat, not part of the six-canary campaign',
        'render_command': command,
        'render_environment_overrides': {'PSYCLES_DISABLE_SHADER_CACHE': '1'},
        'render_completion': 'exit status zero observed by the invoking terminal; complete image audited here',
        'log': work.source(log_path), 'output': work.source(actual),
        'reference': work.source(reference),
        'implementation_sha256': campaign['implementation_sha256'],
        'timings': work.benchmark._parse_psycles_timings(text),
        'frame': dict(zip(['stages', 'fields', 'bytes'], map(int, frame.groups()))),
        'first_run_links': links((directory / 'monster-fixed-1.log').read_text()),
        'repeat_links': links(text),
        'actual_all_channels_finite': True,
        'comparison': work.benchmark._public_process_record(comparison),
        'passes': report['passes']}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['timings', 'frame', 'repeat_links']}, indent=2))


if __name__ == '__main__':
    main()
