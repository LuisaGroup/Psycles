"""Freeze full Barbershop native film routing A/B/B/A evidence."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'docs/validation/2026-09-08/lamp-routing-and-surface'))
from audit_inline_experiment import inspect
from audit_surface import source
sys.path.insert(0, str(ROOT / 'tools'))
import compare_cycles as compare
import render_pass_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'routed_first', 'routed_repeat', 'restored', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    paths = {k: v for k, v in vars(args).items() if k != 'output'}
    rows = {name: inspect(path) for name, path in paths.items()}
    baseline, channels = compare._read_image(args.before / 'psycles.exr')
    render_pass_contract.validate_channels(channels)
    assert baseline.shape == (858, 2048, 46) and np.isfinite(baseline).all()
    for name, path in paths.items():
        row = rows[name]
        row['surface_text'] = source(path / 'surface.text')
        row['command'] = json.loads((path / 'command.json').read_text())
        assert Path(row['command']['cwd']).resolve() == path.resolve() != ROOT
        row['session_init_seconds'] = float(re.search(
            r'Luisa shader JIT completed in ([\d.]+) s', (path / 'render.log').read_text())[1])
        row['total_static_instructions'] = sum(x['static_instructions'] for x in row['functions'].values())
        kernel = row['profile']['shade_surface']['kernel']
        with (path / 'psycles_kernel_trace.csv').open() as stream:
            blocks = {tuple(int(r['Workgroup_Size_' + a]) for a in 'XYZ')
                      for r in csv.DictReader(stream) if r['Kernel_Name'] == kernel}
        assert blocks == {(512, 1, 1)}
        row['surface_workgroup_sizes'] = sorted(blocks)
        actual, actual_channels = ((baseline, channels) if name == 'before'
                                   else compare._read_image(path / 'psycles.exr'))
        render_pass_contract.validate_channels(actual_channels)
        assert actual.shape == baseline.shape and np.isfinite(actual).all()
        passes = {}
        for render_pass in render_pass_contract.PASSES:
            reference = baseline[:, :, compare._find_cycles_channels(channels, render_pass)]
            pixels = actual[:, :, compare._find_cycles_channels(actual_channels, render_pass)]
            delta = pixels.astype(np.float64) - reference
            rms = np.sqrt(np.mean(reference.astype(np.float64) ** 2))
            passes[render_pass] = {
                'relative_rmse': float(np.sqrt(np.mean(delta * delta)) / max(rms, 1e-20)),
                'maximum_absolute_error': float(np.max(np.abs(delta))),
                'different_pixels': int(np.count_nonzero(np.any(delta != 0, axis=2))),
            }
        row['image_control'] = {'shape': list(actual.shape), 'all_channels_finite': True,
                                'passes_vs_before': passes}
        if name != 'before':
            del actual
    for a, b in (('before', 'restored'), ('routed_first', 'routed_repeat')):
        assert rows[a]['surface_text']['sha256'] == rows[b]['surface_text']['sha256']
        assert rows[a]['command']['implementation_sha256'] == rows[b]['command']['implementation_sha256']
    assert len({r['command']['executor_sha256'] for r in rows.values()}) == 1

    def median(names):
        return {'render_seconds': statistics.median(rows[n]['render_wall_seconds'] for n in names),
                'surface_seconds': statistics.median(rows[n]['profile']['shade_surface']['gpu_seconds'] for n in names)}

    result = {
        'schema': 'psycles.film-routing-profiles.v1',
        'scope': 'sequential full Barbershop 2048x858/64 spp/seed 0 A/B/B/A; no overlapping heavy work launched by the agent; not fresh Cycles pairs',
        'intervention': 'native shadow-pass state, zero-only component ratios and direct/indirect destination routing',
        'unchanged': 'inlining policy, fast math, register limits, launch geometry, RNG, SVM word images and static allocation analysis',
        'isa_scope': 'ELF STT_FUNC extents, excluding padding; static sites, not executed instruction/spill counts',
        'image_scope': '15-pass/46-channel intervention controls, not a CPU shader oracle or original-Cycles parity proof',
        'jit_scope': 'main shader cache disabled, downstream caches not cleared; initialization includes setup/baking, not matched cold-JIT trials',
        'runs': rows,
        'medians': {'before': median(('before', 'restored')),
                    'routed': median(('routed_first', 'routed_repeat'))},
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
