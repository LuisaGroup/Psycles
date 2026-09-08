import json
import math
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
import cycles_path_trace_schema as schema

def number(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]

traces = []
for line in Path(sys.argv[1]).read_text().splitlines():
    line = re.sub(r'^\d+: ', '', line)  # CTest verbose output prefixes.
    if line.startswith('dispatch_trace_begin '):
        record = {'context': [int(v) for v in line.split()[1:]], 'slots': []}
    elif line.startswith('dispatch_trace_slot '):
        _, slot, *bits = line.split()
        assert int(slot) == len(record['slots'])
        record['slots'].append([int(v, 16) for v in bits])
    elif line == 'dispatch_trace_end':
        assert len(record['slots']) == schema.AOV_COUNT
        traces.append(record)
assert traces and traces[0]['context'][0] == 0
reference = traces[0]['slots']
results = []
for trace in traces[1:]:
    if trace['context'][-1]:
        continue  # Deliberately different zero-NEE scene, not a scheduler pair.
    differences = []
    for slot in schema.SLOTS:
        policies = (*schema.comparison_policies(slot), 'written_exact')
        for component, (expected_bits, actual_bits, policy) in enumerate(zip(
                reference[slot.index], trace['slots'][slot.index], policies)):
            expected, actual = number(expected_bits), number(actual_bits)
            assert math.isfinite(expected) and math.isfinite(actual)
            if expected_bits != actual_bits:
                differences.append({'slot': slot.index, 'name': slot.name,
                    'component': component, 'policy': policy,
                    'expected': expected, 'actual': actual,
                    'expected_bits': f'{expected_bits:08x}',
                    'actual_bits': f'{actual_bits:08x}',
                    'scaled_error': abs(expected - actual) / max(1., abs(expected), abs(actual))})
    exact = [d for d in differences if d['policy'] in
             (schema.COMPARE_EXACT, schema.COMPARE_RANDOM_EXACT, 'written_exact')]
    results.append({'context': trace['context'], 'exact_differences': exact,
                    'max_scaled_error': max((d['scaled_error'] for d in differences), default=0.),
                    'above_old_bound': [d for d in differences if d['scaled_error'] > 2e-5],
                    'differences': differences})
output = {'schema': 'psycles.dispatch-trace-audit.v1', 'trace_count': len(traces),
          'results': results, 'raw_traces': traces}
Path(sys.argv[2]).write_text(json.dumps(output, indent=2) + '\n')
for r in results:
    print(r['context'], 'exact errors', len(r['exact_differences']),
          'max scaled', r['max_scaled_error'], 'above old', r['above_old_bound'])
