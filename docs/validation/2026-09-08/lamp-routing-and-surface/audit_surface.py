"""Attribute a captured shader to its stage and inspect IR/ISA, without timing inference."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re


def source(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def llvm_functions(path):
    result = {}
    current = None
    for line in path.read_text().splitlines():
        if match := re.match(r'define .*?@([^ (]+)\(', line):
            current = match[1]
            result[current] = {'body_lines': 0, 'calls': Counter(), 'allocas': []}
        elif line == '}':
            current = None
        elif current is not None:
            row = result[current]
            row['body_lines'] += 1
            if match := re.search(r'\bcall\b.*?@([^ (]+)\(', line):
                row['calls'][match[1]] += 1
            if re.match(r'\s+%[^=]+ = alloca ', line):
                row['allocas'].append(line.strip())
    return result


def isa_functions(path):
    result = {}
    current = None
    for line in path.read_text().splitlines():
        if match := re.match(r'([0-9a-f]+) <(.+)>:$', line):
            current = match[2]
            result[current] = {'address': int(match[1], 16), 'opcodes': Counter()}
        elif current is not None:
            if match := re.match(r'\s+(\w+).*?// [0-9A-Fa-f]+:', line):
                result[current]['opcodes'][match[1]] += 1
    for row in result.values():
        ops = row.pop('opcodes')
        row.update({
            'static_instructions': sum(ops.values()),
            'scratch_load_instructions': sum(v for k, v in ops.items() if k.startswith('scratch_load')),
            'scratch_store_instructions': sum(v for k, v in ops.items() if k.startswith('scratch_store')),
            'call_sites': ops['s_swappc_b64'],
            'image_sample_instructions': sum(v for k, v in ops.items() if k.startswith('image_sample')),
        })
    return result


def profile(stats_path, trace_path, names):
    stats = {r['Name']: r for r in csv.DictReader(stats_path.open())}
    traces = {name: [] for name in names.values()}
    for row in csv.DictReader(trace_path.open()):
        if row['Kernel_Name'] in traces:
            traces[row['Kernel_Name']].append(row)
    result = {}
    for stage, name in names.items():
        rows = traces[name]
        assert rows, name
        duration = sum(int(r['End_Timestamp']) - int(r['Start_Timestamp']) for r in rows)
        assert duration == int(stats[name]['TotalDurationNs'])
        assert len(rows) == int(stats[name]['Calls'])
        result[stage] = {
            'kernel': name, 'launches': len(rows), 'gpu_seconds': duration / 1e9,
            'vgpr_counts': sorted({int(r['VGPR_Count']) for r in rows}),
            'scratch_bytes': sorted({int(r['Scratch_Size']) for r in rows}),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('llvm_directory', type=Path)
    parser.add_argument('isa_directory', type=Path)
    args = parser.parse_args()
    ir_dir, isa_dir = args.llvm_directory, args.isa_directory
    # The IR diagnostic's log contains raw bitcode from LUISA_DUMP_AMDGPU.
    # Decode defensively and match only complete timestamped mapping lines.
    log = (ir_dir / 'render.log').read_bytes().decode('utf-8', errors='replace')
    stages = dict(re.findall(
        r"^\[.*?Wavefront shader map: stage='[^/]+/([^']+)' structural_hash=([0-9a-f]+)\.$",
        log, re.MULTILINE))
    surface = 'kernel_' + stages['shade_surface']
    final_candidates = []
    for path in ir_dir.glob('hip_kernel_final_*.ll'):
        if re.search(r'^define amdgpu_kernel void @' + surface + r'\(', path.read_text(), re.MULTILINE):
            final_candidates.append(path)
    assert len(final_candidates) == 1, final_candidates
    final = final_candidates[0]
    before = final.with_name(final.name.replace('final', 'before_opt'))
    after = final.with_name(final.name.replace('final', 'after_opt'))
    psycles_isa = isa_functions(isa_dir / 'surface-isa.txt')
    cycles_isa = isa_functions(isa_dir / 'cycles-compute-isa.txt')
    assert surface in psycles_isa
    selected_cycles = {
        name: row for name, row in cycles_isa.items()
        if name == 'kernel_gpu_integrator_shade_surface'
        or 'integrate_surfaceILj1979' in name
        or name.startswith('_Z18svm_node_tex_noise')
        or ('svm_node_closure_bsdfILj1979' in name)
    }
    names = {stage: 'kernel_' + stages[stage]
             for stage in ['shade_surface', 'intersect_closest', 'shade_volume']}
    stats_path = isa_dir / 'psycles_kernel_stats.csv'
    trace_path = isa_dir / 'psycles_kernel_trace.csv'
    files = [before, after, final, ir_dir / 'render.log', isa_dir / 'render.log',
             isa_dir / 'surface-isa.txt', isa_dir / 'cycles-compute-isa.txt',
             isa_dir / 'surface-code-object.txt', isa_dir / 'cycles-compute-object.txt',
             stats_path, trace_path]
    report = {
        'schema': 'psycles.surface-codegen-inspection.v1',
        'surface_kernel': surface,
        'scope': {
            'profile': 'one instrumented Psycles 2048x858/64-spp run; not an unprofiled benchmark or a fresh Cycles pair',
            'isa': 'static emitted instruction sites, not dynamic spill traffic; Cycles rows are selected functions, not the complete transitive call graph',
            'llvm': 'textual body lines are not an instruction-count or performance metric',
        },
        'stages': profile(stats_path, trace_path, names),
        'llvm': {label: {n: row for n, row in llvm_functions(path).items()
                         if not n.startswith('__ocml_')}
                 for label, path in [('before', before), ('after', after), ('final', final)]},
        'psycles_isa': psycles_isa,
        'cycles_selected_isa': selected_cycles,
        'sources': [source(path) for path in files],
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
