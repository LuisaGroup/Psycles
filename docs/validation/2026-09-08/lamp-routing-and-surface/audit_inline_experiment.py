"""Audit a full-scene A/B/A policy intervention, not a CPU rendering oracle."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

import numpy as np

from audit_surface import function_symbols, isa_functions, profile, source

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
import compare_cycles as compare
import render_pass_contract


def identify_stage_artifacts(directory, kernel):
    # LLVM and code-object dump counters are independent and can change with
    # auxiliary shader/cache admission. Resolve the actual entry symbol,
    # never assume that shade_surface is dump 12 in both namespaces.
    def unique(paths, predicate):
        matches = [path for path in paths if predicate(path.read_bytes())]
        assert len(matches) == 1, (kernel, list(map(str, matches)))
        return matches[0]

    definition = re.compile(rb'^define\b[^\n]*@' + re.escape(kernel.encode()) + rb'\(', re.MULTILINE)
    return {
        'before': unique(directory.glob('hip_kernel_before_opt_*.ll'), definition.search),
        'final': unique(directory.glob('hip_kernel_final_*.ll'), definition.search),
        'code_object': unique(directory.glob('hip_isa_*.co'),
                              lambda data: kernel.encode() + b'\0' in data),
    }


def inspect(directory):
    log = (directory / 'render.log').read_text()
    stages = dict(re.findall(
        r"^\[.*?Wavefront shader map: stage='[^/]+/([^']+)' structural_hash=([0-9a-f]+)\.$",
        log, re.MULTILINE))
    names = {s: 'kernel_' + stages[s]
             for s in ['shade_surface', 'intersect_closest', 'shade_volume']}
    work = defaultdict(int)
    for name, count in re.findall(
            r"Wavefront (?:continuation: index=\d+ token=\d+|auxiliary:) name='([^']*)' "
            r'dispatches=\d+ executed=(\d+)', log):
        work[name] += int(count)
    assert work['<entry>'] == 2048 * 858 * 64
    assert work['shade_surface'] > 0
    frame = re.search(r'path coroutine: subroutines=(\d+) frame_fields=(\d+) frame_bytes=(\d+)', log)
    render = re.search(r'Rendered 2048x858 at 64 spp .*? in ([\d.]+) s:', log)
    assert render and frame
    notes = (directory / 'surface-code-object.txt').read_text()
    resources = {key: int(re.search(r'\.' + key + r':\s*(\d+)', notes)[1])
                 for key in ['private_segment_fixed_size', 'sgpr_count', 'vgpr_count',
                             'sgpr_spill_count', 'vgpr_spill_count']}
    symbols = function_symbols(directory / 'surface-code-object.txt')
    isa = isa_functions(directory / 'surface-isa.txt', symbols)
    assert set(isa) == set(symbols), (set(isa), set(symbols))
    assert names['shade_surface'] in symbols
    assert re.search(r'\.name:\s*' + re.escape(names['shade_surface']) + r'\s*$', notes, re.MULTILINE)
    artifacts = identify_stage_artifacts(directory, names['shade_surface'])
    files = ['render.log', 'psycles.exr', 'psycles_kernel_stats.csv', 'psycles_kernel_trace.csv',
             'surface-code-object.txt', 'surface-isa.txt', *[path.name for path in artifacts.values()]]
    return {
        'directory': str(directory),
        'render_wall_seconds': float(render[1]),
        'frame': dict(zip(['stages', 'fields', 'bytes'], map(int, frame.groups()))),
        'work': dict(work),
        'profile': profile(directory / 'psycles_kernel_stats.csv',
                           directory / 'psycles_kernel_trace.csv', names),
        'resources': resources,
        'functions': {name: {**symbols[name], **row} for name, row in isa.items()},
        'code_object_sha256': source(artifacts['code_object'])['sha256'],
        'stage_artifacts': {key: str(path) for key, path in artifacts.items()},
        'sources': [source(directory / name) for name in files],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('keep_boundaries', type=Path)
    parser.add_argument('restored', type=Path)
    args = parser.parse_args()
    directories = vars(args)
    rows = {label: inspect(directory) for label, directory in directories.items()}
    reference, ref_channels = compare._read_image(args.baseline / 'psycles.exr')
    render_pass_contract.validate_channels(ref_channels)
    assert np.isfinite(reference).all()
    for label, directory in directories.items():
        if label == 'baseline':
            pixels, channels = reference, ref_channels
        else:
            pixels, channels = compare._read_image(directory / 'psycles.exr')
        render_pass_contract.validate_channels(channels)
        assert pixels.shape == reference.shape and np.isfinite(pixels).all()
        rows[label]['image'] = {'shape': list(pixels.shape), 'all_channels_finite': True,
                               'passes_vs_baseline_control': {}}
        for name in render_pass_contract.PASSES:
            ref = reference[:, :, compare._find_cycles_channels(ref_channels, name)]
            actual = pixels[:, :, compare._find_cycles_channels(channels, name)]
            delta = actual.astype(np.float64) - ref
            mse = np.mean(delta * delta)
            rms = np.sqrt(np.mean(ref.astype(np.float64) ** 2))
            rows[label]['image']['passes_vs_baseline_control'][name] = {
                'relative_rmse': float(np.sqrt(mse) / max(rms, 1e-20)),
                'maximum_absolute_error': float(np.max(np.abs(delta))),
            }
        if label != 'baseline':
            del pixels
    assert rows['baseline']['code_object_sha256'] == rows['restored']['code_object_sha256']
    print(json.dumps({
        'schema': 'psycles.surface-inline-experiment.v1',
        'scope': 'same full Barbershop 2048x858/64-spp scene and fixed seed; A/B/A profiler runs, not a fresh Cycles pair; image comparisons are intervention controls, not renderer oracles',
        'intervention': 'B skips only inline_unique_oversized_generated_callables after IPO; ABI projection/demotion and downstream HIPRTC stay enabled; no noinline marker, arithmetic or register cap added',
        'decision': 'reject B and restore the existing policy; this is not a compiler repair',
        'runs': rows,
    }, indent=2))


if __name__ == '__main__':
    main()
