"""Sequential original/Luisa/Luisa/original isolated Noise handler controls."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

EVIDENCE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/Projects/Psycles-surface-svm')


def source(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest}


runs = []
for engine, label in (('cycles', 'original'), ('luisa', 'actual'),
                      ('luisa', 'actual-repeat'), ('cycles', 'original-repeat')):
    path = Path(tempfile.mkdtemp(prefix=label + '-', dir=EVIDENCE))
    binary = (EVIDENCE / 'build/cycles-node/original-node-probe' if engine == 'cycles'
              else EVIDENCE / 'build/node-codegen-probe')
    files = [binary, *[EVIDENCE / name for name in ('node_inputs.h', 'inputs.h', 'node_probe.cpp', 'node_oracle.hip')]]
    if engine == 'luisa':
        files += [ROOT / name for name in ('build/libpsycles_luisa_runtime.so',
            'build/bin/libluisa-backend-hip.so', 'build/bin/libluisa-xir.so')]
    sources = [source(f) for f in files]
    command = ['/opt/rocm/bin/rocprofv3', '--kernel-trace', '--stats',
        '--output-format', 'csv', '--output-directory', str(path),
        '--output-file', 'run', '--', str(binary), 'bench']
    overrides = {} if engine == 'cycles' else {
        'LUISA_DUMP_LLVM_IR': '1', 'LUISA_DUMP_HIP_ISA': str(path)}
    record = {'engine': engine, 'label': label, 'directory': str(path),
        'command': command, 'cwd': str(path), 'environment_overrides': overrides,
        'sources': sources}
    (path / 'command.json').write_text(json.dumps(record, indent=2) + '\n')
    with (path / 'run.log').open('x') as log:
        subprocess.run(command, cwd=path, env={**os.environ, **overrides},
            stdout=log, stderr=subprocess.STDOUT, check=True)
    assert sources == [source(f) for f in files]
    pattern = ('_Z10node_probe', 'node_probe(') if engine == 'cycles' else ('kernel_',)
    with (path / 'run_kernel_trace.csv').open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['Kernel_Name'].startswith(pattern)]
    assert len(rows) == 18 and len({r['Kernel_Name'] for r in rows}) == 1
    assert all(r['Workgroup_Size_X'] == '256' and r['Grid_Size_X'] == '1048576' for r in rows)
    assert all(r['Workgroup_Size_Y'] == r['Workgroup_Size_Z'] == '1' for r in rows)
    seconds = [(int(r['End_Timestamp']) - int(r['Start_Timestamp'])) / 1e9 for r in rows]
    record.update({'kernel': rows[0]['Kernel_Name'], 'dispatch_seconds': seconds,
        'after_two_warmups_median_seconds': statistics.median(seconds[2:]),
        'vgprs': sorted({int(r['VGPR_Count']) for r in rows}),
        'scratch_bytes': sorted({int(r['Scratch_Size']) for r in rows})})
    outputs = {int(i): list(map(float, values.split())) for i, values in re.findall(
        r'^result (\d+) (.+)$', (path / 'run.log').read_text(), re.M)}
    assert len(outputs) == 32 and all(len(r) == 4 for r in outputs.values())
    record['outputs'] = outputs
    runs.append(record)
    print(label, path.name, record['after_two_warmups_median_seconds'], flush=True)

(EVIDENCE / 'throughput.json').write_text(json.dumps({
    'schema': 'psycles.noise-handler-throughput.v1',
    'scope': 'synthetic dynamic SVM Noise handler, not a scene timing; 32 input records repeated across 1048576 invocations; 18 sequential 256-thread dispatches per run, first two excluded from reported median; original/Luisa/Luisa/original order; no overlapping heavy task launched by the agent',
    'runs': runs}, indent=2) + '\n')
