"""Audit captured original GPU and production Luisa Noise, not a host shader."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'docs/validation/2026-09-08/lamp-routing-and-surface'))
from audit_surface import function_symbols, isa_functions, source


def code(notes_path, isa_path):
    symbols = function_symbols(notes_path)
    instructions = isa_functions(isa_path, symbols)
    assert set(symbols) == set(instructions)
    resources = {}
    for block in re.split(r'(?m)^  - \.args:', notes_path.read_text())[1:]:
        name = re.search(r'(?m)^    \.name:\s*(\S+)', block)[1]
        resources[name] = {key: int(re.search(r'\.' + key + r':\s*(\d+)', block)[1])
                          for key in ('private_segment_fixed_size', 'sgpr_count', 'vgpr_count',
                                      'sgpr_spill_count', 'vgpr_spill_count')}
    return {'functions': {n: {**s, **instructions[n]} for n, s in symbols.items()},
            'kernel_resources': resources, 'sources': [source(notes_path), source(isa_path)]}


def output(log_path, width):
    rows = {int(i): list(map(float, values.split())) for i, values in re.findall(
        r'^result (\d+) (.+)$', log_path.read_text(), re.M)}
    assert set(rows) == set(range(32))
    assert all(len(v) == width and all(map(math.isfinite, v)) for v in rows.values())
    return [rows[i] for i in range(32)]


def agreement(reference, actual):
    assert len(reference) == len(actual) == 32
    errors = [abs(a - b) for x, y in zip(reference, actual) for a, b in zip(x, y)]
    assert all(abs(a - b) <= 2e-5 * max(1, abs(a))
               for x, y in zip(reference, actual) for a, b in zip(x, y))
    return {'observed_floats': len(errors), 'exact_floats': errors.count(0),
            'maximum_absolute_error': max(errors), 'all_finite': True,
            'comparison': 'finite absolute-relative tolerance 2e-5; no math changes for last bits'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    base = args.evidence
    leaf = {}
    for name in ('noise', 'fbm'):
        original = base / ('original-' + name + '.log')
        actual = base / name / 'render.log'
        leaf[name] = {'agreement': agreement(output(original, 1), output(actual, 1)),
                      'logs': [source(original), source(actual)],
                      'actual_code': code(base / name / 'notes.txt', base / name / 'isa.txt')}
    throughput = json.loads((base / 'throughput.json').read_text())
    assert throughput['schema'] == 'psycles.noise-handler-throughput.v1'
    runs = throughput['runs']
    assert [r['engine'] for r in runs] == ['cycles', 'luisa', 'luisa', 'cycles']
    reference = output(Path(runs[0]['directory']) / 'run.log', 4)
    for row in runs:
        directory = Path(row['directory'])
        assert row['cwd'] == str(directory) and directory != ROOT
        assert len(row['dispatch_seconds']) == 18
        assert row['after_two_warmups_median_seconds'] == statistics.median(row['dispatch_seconds'][2:])
        row['after_two_warmups_range_seconds'] = [min(row['dispatch_seconds'][2:]), max(row['dispatch_seconds'][2:])]
        row['agreement'] = agreement(reference, output(directory / 'run.log', 4))
        # Source snapshots must still match the files frozen before each run.
        for frozen in row['sources']:
            path = Path(frozen['path'])
            if path.parent == base:
                assert source(path)['sha256'] == frozen['sha256']
        row['artifacts'] = [source(directory / f) for f in (
            'command.json', 'run.log', 'run_kernel_trace.csv', 'run_kernel_stats.csv')]
    snapshot_names = ('CMakeLists.txt', 'inputs.h', 'node_inputs.h', 'probe.cpp',
                      'oracle.hip', 'node_probe.cpp', 'node_oracle.hip', 'run_throughput.py')
    for name in snapshot_names:
        assert (Path(__file__).parent / 'probes' / name).read_bytes() == (base / name).read_bytes()
    result = {
        'schema': 'psycles.noise-codegen.v1',
        'implementation': {'parent': 'e79644af', 'documentation_head': '5c3746b2', 'child': 'da8fff856',
                           'no_production_source_or_inline_policy_changes': True},
        'original': {'revision': 'cb168525138fecc792cc393f94afc39582b0103c',
            'sources': [source(Path('/home/mike/Projects/blender-cycles-trace-5.2/intern/cycles') / f)
                        for f in ('kernel/svm/noise.h', 'kernel/svm/fractal_noise.h', 'kernel/svm/noisetex.h')]},
        'scope': 'bottom-up 3D signed noise, FBM and dynamic full Noise handler controls; original GPU routines included directly; no CPU shader evaluator',
        'compilation': 'fast math, native original HIP O3, 256 threads/group, 32-thread builds and hipcc -parallel-jobs=32; ordinary Luisa inlining decisions',
        'isa_scope': 'ELF STT_FUNC extents, not dynamic instructions; original saved .bc is pre-optimization and is not presented as final IR',
        'leaf_code_original': code(base / 'original-notes.txt', base / 'original-isa.txt'),
        'leaf_controls': leaf,
        'initial_handler_control': {
            'agreement': agreement(output(base / 'original-node.log', 4), output(base / 'node/render.log', 4)),
            'scope': '32 invocations before the ring-input benchmark edit; live outputs agree and Luisa checks 12-word PC; initial notes are not mixed with benchmark code metrics',
            'logs': [source(base / 'original-node.log'), source(base / 'node/render.log')]},
        'handler_throughput': throughput,
        'handler_throughput_code': {
            'original': code(base / 'throughput-original-notes.txt', base / 'throughput-original-isa.txt'),
            'actual': code(base / 'throughput-actual-notes.txt', base / 'throughput-actual-isa.txt')},
        'throughput_limitations': '32 input records repeated over 1048576 invocations; first 32 output records observed, not a million-record image comparison; four runs and overlapping per-dispatch ranges do not establish statistical significance or Noise share of Barbershop',
        'decision': 'does not reproduce the roughly twofold full-surface gap; retain no inline-policy or math intervention from this audit',
        'retained_harness_failure': {'source': source(base / 'throughput-run.log'),
            'cause': 'rocprof demangled original kernel name; parser now accepts original mangled/demangled prefixes; failed parsing run retained, not included in four-run table'},
        'evidence': [source(base / f) for f in (*snapshot_names, 'formal-analysis.md',
            'build.log', 'build-node.log', 'build-node-throughput.log', 'throughput.json', 'throughput-run-final.log')]}
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
