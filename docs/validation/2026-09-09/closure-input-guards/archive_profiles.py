"""Freeze the closure-input guards A/B/B/A; no CPU shader oracle."""
import argparse
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
    for name in ('before', 'guarded_first', 'guarded_repeat', 'restored', 'output'):
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
        row['session_init_seconds'] = float(re.search(
            r'Luisa shader JIT completed in ([\d.]+) s', (path / 'render.log').read_text())[1])
        row['total_static_instructions'] = sum(x['static_instructions'] for x in row['functions'].values())
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
    assert rows['before']['surface_text']['sha256'] == rows['restored']['surface_text']['sha256']
    assert rows['guarded_first']['surface_text']['sha256'] == rows['guarded_repeat']['surface_text']['sha256']
    def median(names):
        return {'render_seconds': statistics.median(rows[n]['render_wall_seconds'] for n in names),
                'surface_seconds': statistics.median(rows[n]['profile']['shade_surface']['gpu_seconds'] for n in names)}
    result = {
        'schema': 'psycles.closure-input-guards-profiles.v1',
        'scope': 'sequential full Barbershop 2048x858/64 spp/seed 0 A/B/B/A; no overlapping heavy work launched by the agent; not fresh Cycles pairs',
        'intervention': 'typed payload address snapshot with original caustic/allocation/anisotropy/color input guards for Glossy, Refraction, Glass and Metallic; one PC advance even when skipped',
        'unchanged': 'inlining policy, fast math, register limits, SVM word images and static allocation analysis',
        'isa_scope': 'ELF STT_FUNC extents, excluding alignment padding; static sites, not executed instruction/spill counts',
        'image_scope': '15-pass/46-channel intervention controls, not a CPU shader oracle or original-Cycles parity proof',
        'jit_scope': 'main shader cache disabled, downstream caches not cleared; initialization includes setup/baking, not matched cold-JIT trials',
        'runs': rows,
        'medians': {'before': median(('before', 'restored')), 'guarded': median(('guarded_first', 'guarded_repeat'))},
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
