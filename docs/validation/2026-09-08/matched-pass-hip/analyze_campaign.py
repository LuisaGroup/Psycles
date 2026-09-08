"""Validate archived render evidence and summarize equal-pass observations."""
import json
from pathlib import Path
import statistics
import sys

ROOT = Path('/home/mike/Projects/Psycles-surface-svm')
CAMPAIGN = Path('/var/tmp/psycles-matched-pass-hip-2c98Q0/campaign.json')
sys.path.insert(0, str(ROOT / 'tools'))
import run_scene_benchmark as benchmark
import render_pass_contract as contract

campaign = json.loads(CAMPAIGN.read_text())
assert len(campaign['records']) == 12
for name, expected in campaign['implementation_sha256'].items():
    assert benchmark._sha256(ROOT / name) == expected, name

def stats(values):
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values),
            'observations': values}

summary = {'schema': 'psycles.matched-pass-summary.v1', 'campaign': str(CAMPAIGN),
           'source_revision': 'c5bf9247', 'luisa_revision': '9ea3b720f',
           'passes': list(contract.PASSES), 'samples': 256, 'scenes': {}}
for scene in ['monk', 'monster', 'classroom', 'barbershop']:
    records = sorted([r for r in campaign['records'] if r['scene'] == scene],
                     key=lambda r: r['repeat'])
    assert [r['repeat'] for r in records] == [1, 2, 3]
    timings = {key: [] for key in ['cycles_main_loop', 'cycles_render_call',
                                  'psycles_render', 'psycles_session_init', 'psycles_scene_compile']}
    quality = {key: [] for key in contract.PASSES}
    sources = []
    for r in records:
        path = Path(r['manifest'])
        assert benchmark._sha256(path) == r['manifest_sha256']
        m = json.loads(path.read_text())
        assert m['schema'] == 'psycles.scene-benchmark.v3' and m['status'] == 'complete'
        assert r['actual_all_finite'] and r['actual_channels'] == 46
        c, p = m['renderers']['cycles']['hip'], m['renderers']['psycles']['hip-wavefront-staged']
        for group, key, command in [('cycles', 'hip', 'cycles_hip'),
                                    ('psycles', 'hip-wavefront-staged', 'psycles_hip-wavefront-staged')]:
            record = m['renderers'][group][key]
            assert benchmark._can_resume_render(
                m, command_key=command, renderer_group=group, renderer_key=key,
                expected_command=m['commands'][command]['command'],
                expected_output=Path(record['output']), required_timings=('render_seconds',),
                metadata_path=Path(c['metadata']) if group == 'cycles' else None)
        metadata = json.loads(Path(c['metadata']).read_text())
        assert metadata['passes'] == list(contract.PASSES) and metadata['samples'] == 256
        comparison_path = Path(m['comparisons']['psycles-hip-wavefront-staged-vs-cycles-hip']['report'])
        comparison = json.loads(comparison_path.read_text())
        assert set(comparison['passes']) == set(contract.PASSES)
        for name, item in comparison['passes'].items():
            assert item['actual_invalid_pixels'] == 0
            quality[name].append({key: item[key] for key in
                                 ['relative_rmse', 'rmse', 'cycles_rms', 'mean_absolute_error',
                                  'maximum_absolute_error', 'actual_invalid_pixels',
                                  'reference_invalid_pixels', 'invalid_pixels']})
        timings['cycles_main_loop'].append(c['render_seconds'])
        timings['cycles_render_call'].append(c['render_call_seconds'])
        timings['psycles_render'].append(p['render_seconds'])
        timings['psycles_session_init'].append(p['shader_jit_seconds'])
        timings['psycles_scene_compile'].append(p['scene_compile_seconds'])
        sources.append({'manifest': str(path), 'manifest_sha256': r['manifest_sha256'],
                        'comparison': str(comparison_path),
                        'comparison_sha256': benchmark._sha256(comparison_path)})
        assert r['frame'] == records[0]['frame']
    medians = {k: stats(v) for k, v in timings.items()}
    summary['scenes'][scene] = {
        'extent': [m['settings']['width'], m['settings']['height']],
        'seed': metadata['seed'], 'frame_number': metadata['frame'],
        'use_animated_seed': metadata['use_animated_seed'], 'effective_seed': metadata['effective_seed'],
        'frame': records[0]['frame'], 'timings_seconds': medians,
        'psycles_over_cycles': medians['psycles_render']['median'] / medians['cycles_main_loop']['median'],
        'relative_time_observations': [p/c for p,c in zip(timings['psycles_render'], timings['cycles_main_loop'])],
        'quality': quality, 'sources': sources}
print(json.dumps(summary, indent=2))
