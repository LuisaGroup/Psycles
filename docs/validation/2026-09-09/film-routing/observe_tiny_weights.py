"""Compare original GPU and production DSL weight behavior, without changing math.

This is an arithmetic-domain diagnostic, not a finite-pixel waiver or a
software division implementation. Native and actual use fast math.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import struct
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cycles', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--arch', default='gfx1201')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    commands = []

    def run(command, name, cwd, **kwargs):
        commands.append({'command': command, 'cwd': str(cwd), **kwargs})
        with (out / name).open('w') as log:
            subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                           check=True, **kwargs)

    run(['/opt/rocm/bin/hipcc', '-parallel-jobs=32', '--offload-arch=' + args.arch,
         '-DHIPCC', '-std=c++20', '-O3', '-ffast-math', '-I', str(args.cycles),
         str(HERE / 'tiny_weights_original.hip'), '-o', str(out / 'original')],
        'original-build.log', out)
    native = subprocess.check_output(['ninja', '-C', 'build', '-t', 'commands',
        'psycles_luisa_cycles_film_routing_tests'], cwd=ROOT, text=True).splitlines()
    entry, = [c for c in native if c.endswith('/test_luisa_cycles_film_routing.cpp')]
    compile = shlex.split(entry)
    compile[compile.index('-o') + 1] = str(out / 'actual.o')
    compile[compile.index('-c') + 1] = str(HERE / 'tiny_weights_actual.cpp')
    for flag, target in (('-MF', 'actual.o.d'), ('-MT', 'actual.o')):
        if flag in compile:
            compile[compile.index(flag) + 1] = str(out / target)
    run(compile, 'actual-build.log', ROOT / 'build')
    link = [s for s in shlex.split(native[-1]) if s not in (':', '&&')
            and not s.startswith('-Wl,--dependency-file=')]
    link = [str(out / 'actual.o') if s.endswith('test_luisa_cycles_film_routing.cpp.o')
            else s for s in link]
    link[link.index('-o') + 1] = str(out / 'actual')
    run(link, 'actual-link.log', ROOT / 'build')
    run([str(out / 'original')], 'original.log', out)
    run([str(ROOT / 'build/bin/psycles-ratio-diagnostic'), 'hip'], 'actual.log', out,
        executable=str(out / 'actual'))

    def rows(path):
        return [[int(v, 16) for v in line.split()] for line in path.read_text().splitlines()
                if re.fullmatch(r'[0-9a-f]{8}(?: [0-9a-f]{8}){7}', line)]

    original, actual = rows(out / 'original.log'), rows(out / 'actual.log')
    assert len(original) == len(actual) == 9

    def classification(bits):
        if bits & 0x7f800000 == 0x7f800000:
            return 'nan' if bits & 0x7fffff else ('-inf' if bits >> 31 else '+inf')
        return 'finite'

    comparisons = []
    for a, b in zip(original, actual):
        assert a[:2] == b[:2]
        kinds = [classification(v) for v in a[2:]]
        assert kinds == [classification(v) for v in b[2:]]
        for kind, x, y in zip(kinds, a[2:], b[2:]):
            if kind == 'finite':
                fx = struct.unpack('<f', struct.pack('<I', x))[0]
                fy = struct.unpack('<f', struct.pack('<I', y))[0]
                assert abs(fx - fy) <= 2e-6
        comparisons.append({'input_bits': a[:2], 'original_bits': a[2:],
                            'actual_bits': b[2:], 'classifications': kinds})
    sources = [HERE / name for name in ('tiny_weight_inputs.h',
               'tiny_weights_original.hip', 'tiny_weights_actual.cpp')]
    sources += [args.cycles / name for name in ('kernel/film/light_passes.h',
                                               'util/math_float3.h')]
    result = {'schema': 'psycles.tiny-weight-fastmath-observation.v1',
        'scope': 'original GPU and actual production ratio, dynamic inputs; no CPU reference math',
        'limitation': 'confirms shared arithmetic boundary; not a trace of every invalid Classroom path',
        'commands': commands, 'rows': comparisons,
        'sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in sources]}
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Original and actual GPU: 9/9 finite/classification controls match.')


if __name__ == '__main__':
    main()
