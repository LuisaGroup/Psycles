"""Archive exact canary commands and metrics without repeated pass metadata."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

source = Path(sys.argv[1])
document = json.loads(source.read_text())
assert len(document['records']) == 6
records = []
metric_keys = ('relative_rmse', 'rmse', 'mae', 'max_abs_error',
               'actual_invalid_pixels', 'reference_invalid_pixels',
               'comparison_invalid_pixels')
for record in document['records']:
    assert record['actual_all_finite'] and len(record['passes']) == 15
    records.append({
        **{k: v for k, v in record.items() if k not in ('passes', 'pass_contract')},
        'extent': [record['pass_contract']['width'], record['pass_contract']['height']],
        'channel_count': len(record['pass_contract']['channels']),
        'passes': {name: {k: value[k] for k in metric_keys if k in value}
                   for name, value in record['passes'].items()}})
old_seconds = {'monk': 13.6826, 'monster': 15.1328,
               'classroom': 18.7626, 'barbershop': 39.2753}
summary = {}
for scene, old in old_seconds.items():
    times = [r['timings']['render_seconds'] for r in records if r['scene'] == scene]
    median = statistics.median(times)
    summary[scene] = {'old_equal_pass_psycles_median_seconds': old,
                      'new_psycles_seconds': times, 'new_median_seconds': median,
                      'change_fraction': median / old - 1.}
print(json.dumps({
    **{k: v for k, v in document.items() if k != 'records'},
    'archive_schema': 'psycles.bound-sampler-canary-summary.v1',
    'raw_source': str(source),
    'raw_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'summary': summary, 'records': records}, indent=2))
