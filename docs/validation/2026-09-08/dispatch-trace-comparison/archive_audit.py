"""Losslessly compact captured trace bits, not reference rendering results."""
import hashlib
import json
from pathlib import Path
import sys

source = Path(sys.argv[1])
document = json.loads(source.read_text())
traces = document['raw_traces']
reference = traces[0]['slots']
zero = [0, 0, 0, 0]
baseline = {str(i): [f'{x:08x}' for x in v]
            for i, v in enumerate(reference) if v != zero}
captures = []
for trace in traces:
    changes = {str(i): [f'{x:08x}' for x in v]
               for i, v in enumerate(trace['slots']) if v != reference[i]}
    reconstructed = [list(v) for v in reference]
    for slot, bits in changes.items():
        reconstructed[int(slot)] = [int(x, 16) for x in bits]
    assert reconstructed == trace['slots']
    captures.append({'context': trace['context'], 'changed_slots': changes})
print(json.dumps({
    'schema': 'psycles.dispatch-trace-bits.v1',
    'source': str(source),
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'slot_count': len(reference),
    'context_fields': ['scheduler_enum', 'samples_per_dispatch', 'split_request',
                       'surface_sorting', 'tail_threshold', 'zero_nee_fixture'],
    'encoding': 'uint32 hexadecimal float32 bits; absent reference slots are four zero bits; captures override reference slots',
    'reference_slots': baseline,
    'captures': captures,
    'audit': [{k: v for k, v in r.items() if k != 'differences'}
               for r in document['results']]}, indent=2))
