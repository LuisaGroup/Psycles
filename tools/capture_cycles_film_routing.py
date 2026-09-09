"""Extract original shadow-pass state blocks for a GPU-only film observer."""
import argparse
import hashlib
import json
from pathlib import Path


def block(text, start):
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cycles', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    observed = []
    source = ['// Verbatim original GPU state-assignment blocks; generated.\n']
    for domain in ('surface', 'volume'):
        path = args.cycles / f'kernel/integrator/shade_{domain}.h'
        text = path.read_text()
        anchor = '  uint32_t shadow_flag = INTEGRATOR_STATE(state, path, flag);'
        assert text.count(anchor) == 1
        start = text.index(anchor)
        state_block = block(text, text.index(
            '  if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_PASSES)', start))
        visibility_block = ''
        if domain == 'volume':
            visibility_block = block(text, text.index('  if (bounce == 0)', start))
        observed.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'state_block': state_block, 'visibility_block': visibility_block})
        # The harness supplies the boundary state and BsdfEval. All native
        # branching, weight evaluation and bit changes below are unedited.
        source.append(f'''__device__ void original_{domain}_shadow_pass(
    IntegratorState state, IntegratorShadowState shadow_state, BsdfEval bsdf_eval) {{
  const auto bounce = INTEGRATOR_STATE(state, path, bounce);
  auto path_visibility = INTEGRATOR_STATE(state, path, visibility);
{anchor}
{state_block}
{visibility_block}
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, flag) = shadow_flag;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, visibility) = path_visibility;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, bounce) = bounce;
}}
''')
    args.output.write_text('\n'.join(source))
    args.output.with_suffix('.json').write_text(json.dumps(observed, indent=2) + '\n')


if __name__ == '__main__':
    main()
