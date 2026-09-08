"""Create the bounded checked-in record from retained live evidence."""
import argparse
import json
from pathlib import Path
import re
import statistics

import analyze_work as work


def verify(record):
    assert work.source(Path(record['path'])) == record, record['path']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence', type=Path)
    args = parser.parse_args()
    directory = args.evidence
    count_path = directory / 'work-summary.json'
    counts = json.loads(count_path.read_text())
    assert counts['schema'] == 'psycles.light-endpoint-work.v1'
    for record in counts['sources'].values():
        verify(record)
    assert work.cycles_counts(directory / 'cycles.log') == counts['cycles_stages']
    for label, prefix in [('before', 'psycles'), ('after', 'psycles-fixed')]:
        actual = work.psycles_counts(directory / f'{prefix}.log')
        for name, value in actual.items():
            assert json.loads(json.dumps(value)) == counts['psycles'][label][name], (label, name)
    canary_path = directory / 'canaries.json'
    canaries = json.loads(canary_path.read_text())
    assert canaries['schema'] == 'psycles.light-endpoint-canaries.v1'
    assert len(canaries['records']) == 6
    for name, digest in canaries['implementation_sha256'].items():
        assert work.source(work.ROOT / name)['sha256'] == digest, name
    records = []
    for row in canaries['records']:
        assert row['render']['returncode'] == row['comparison']['returncode'] == 0
        assert row['actual_all_finite']
        assert len(row['passes']) == 15
        assert all(p['actual_invalid_pixels'] == 0 for p in row['passes'].values())
        actual_image = Path(row['render']['command'][2]).with_suffix('.exr')
        assert work.source(actual_image)['sha256'] == row['output_sha256']
        verify({'path': row['source_manifest'], 'sha256': row['source_manifest_sha256']})
        reference = Path(row['comparison']['command'][2])
        assert work.source(reference)['sha256'] == row['reference_sha256']
        records.append({k: row[k] for k in ['scene', 'repeat', 'timings', 'frame',
                                          'source_manifest', 'source_manifest_sha256',
                                          'output_sha256', 'reference_sha256',
                                          'actual_all_finite']})
        records[-1]['passes'] = {name: {key: p[key] for key in
            ['relative_rmse', 'rmse', 'maximum_absolute_error',
             'actual_invalid_pixels', 'reference_invalid_pixels', 'valid_pixels']}
            for name, p in row['passes'].items()}
    gates = {}
    for name, expected in [('hip', (181, 181)), ('fallback', (183, 183)),
                           ('host', (156, 157))]:
        path = directory / f'ctest-{name}.log'
        text = path.read_text()
        match = re.search(r'(\d+)% tests passed(?:, (\d+) tests failed)? (?:out )?of (\d+)', text)
        assert match, name
        total = int(match[3])
        failed = int(match[2] or 0)
        assert (total - failed, total) == expected
        gates[name] = {'passed': total - failed, 'total': total, 'log': work.source(path)}
    native_path = directory / 'endpoint-native-vulkan.log'
    native = native_path.read_text()
    assert '240 Cycles GPU endpoint checks, 2048 visibility predicates; failures=0' in native
    assert len(re.findall('SPIR-V compilation successful', native)) == 2
    assert not re.search(r'(?:calling init:|find library=).*?(?:dxcompiler|dxil|libdxc)', native, re.I)
    gates['strict_native_vulkan'] = {'endpoint_components': 240, 'visibility_predicates': 2048,
                                    'spirv_modules': 2, 'dxc_dxil_loaded': False,
                                    'log': work.source(native_path)}
    red_path = directory / 'endpoint-red-hip.log'
    red = red_path.read_text()
    assert len(re.findall(r'^case=\d+ lane=\d+ expected=.* actual=.*$', red, re.M)) == 46
    gates['reduced_red'] = {'failed_endpoint_components': 46, 'total': 240,
                            'log': work.source(red_path)}
    warm_path = directory / 'monster-warm-summary.json'
    warm = json.loads(warm_path.read_text())
    assert warm['schema'] == 'psycles.light-endpoint-warm-control.v1'
    assert warm['actual_all_channels_finite']
    assert warm['implementation_sha256'] == canaries['implementation_sha256']
    for key in ['log', 'output', 'reference']:
        verify(warm[key])
    result = {'schema': 'psycles.light-endpoint-evidence.v1',
              'psycles_code_revision': '7419301e', 'luisa_revision': '9ea3b720f',
              'work_counts': counts, 'canaries': records,
              'barbershop_render_seconds_median': statistics.median(
                  r['timings']['render_seconds'] for r in records if r['scene'] == 'barbershop'),
              'implementation_sha256': canaries['implementation_sha256'], 'gates': gates,
              'raw_work_record': work.source(count_path), 'raw_canary_record': work.source(canary_path),
              'monster_warm_control': warm, 'raw_warm_record': work.source(warm_path),
              'oracle': {str(p.relative_to(work.ROOT)): work.source(p)['sha256'] for p in
                         [work.ROOT / 'tools/cycles_light_endpoint_oracle.hip',
                          work.ROOT / 'tests/cycles_light_endpoint_fixture.h',
                          work.ROOT / 'tests/data/cycles_light_endpoints.txt']}}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
