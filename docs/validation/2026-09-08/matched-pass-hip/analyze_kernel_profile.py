"""Separate actual stage durations from setup kernels and padded work counts."""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path('/home/mike/Projects/Psycles-surface-svm')
EVIDENCE = Path('/var/tmp/psycles-native-volume-svm-06XnDX')
sys.path.insert(0, str(ROOT / 'tools'))
import render_pass_contract

def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

sources = {}
for engine, suffix in [('cycles', 'cycles'), ('psycles', 'psycles-valid')]:
    base = EVIDENCE / f'barbershop-matched-kernel-profile-{suffix}'
    paths = {'stats': base / f'{engine}_kernel_stats.csv',
             'trace': base / f'{engine}_kernel_trace.csv',
             'log': base.with_suffix('.log'), 'image': base.with_suffix('.exr')}
    log = paths['log'].read_text()
    stages = dict(re.findall(r"stage='[^/]+/([^']+)' structural_hash=([0-9a-f]+)", log))
    names = {stage: ('kernel_gpu_integrator_' + stage if engine == 'cycles'
                     else 'kernel_' + stages[stage])
             for stage in ['shade_surface', 'shade_volume', 'intersect_closest']}
    stats = {row['Name']: row for row in csv.DictReader(paths['stats'].open())}
    traces = {name: [] for name in names.values()}
    for row in csv.DictReader(paths['trace'].open()):
        if row['Kernel_Name'] in traces:
            traces[row['Kernel_Name']].append(row)
    stages = {}
    for stage, name in names.items():
        rows = traces[name]
        count = len(rows)
        duration = sum(int(r['End_Timestamp']) - int(r['Start_Timestamp']) for r in rows)
        assert count == int(stats[name]['Calls']) and duration == int(stats[name]['TotalDurationNs'])
        stages[stage] = {'kernel': name, 'calls': count, 'duration_ns': duration,
                         'duration_seconds': duration / 1e9,
                         'vgpr_counts': sorted({int(r['VGPR_Count']) for r in rows}),
                         'scratch_bytes': sorted({int(r['Scratch_Size']) for r in rows})}
    sources[engine] = {'pass_contract': render_pass_contract.inspect_image(paths['image']),
                       'stages': stages,
                       'sources': {k: {'path': str(v), 'sha256': sha(v)} for k, v in paths.items()}}
print(json.dumps({'schema': 'psycles.matched-stage-profile.v1',
                  'extent': [2048, 858], 'samples': 64, 'engines': sources,
                  'scope': 'single-run instrumented kernel durations; not active-path counts or render-wall time'},
                 indent=2))
