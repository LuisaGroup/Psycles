"""Resolve typed SVM resource references from observed registries, not ID guesses.

This checks compilation and binding identity, not image pixels or shading values.
The narrowly admitted image domain is static, non-UDIM, non-generated-sky images.
Unknown/ambiguous identities and metadata mismatches fail closed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'docs/validation/2026-09-08/svm-math-expansion'))
from decode_cycles_bindings import attribute_bindings, image_bindings, psycles_bindings, source
from audit_typed_words import RESOURCE_FIELDS, decode, read_dump


def unique_map(rows, key):
    result = {}
    for row in rows:
        if row[key] in result:
            raise ValueError('ambiguous resource identity: ' + str(row[key]))
        result[row[key]] = row
    return result


def match_images(original, actual, exported):
    if any(row is not None for row in original['udims']):
        raise ValueError('UDIM identity comparison is not implemented by this audit')
    by_name = unique_map([row for row in original['images'] if row], 'loader_name')
    exports = unique_map(exported, 'name')
    unique_map(actual, 'id')
    matches = {}
    rows = []
    colors = {'sRGB': 2, 'Linear': 1, 'Linear Rec.709': 1, 'Non-Color': 0, 'Raw': 0, '': 0}
    alphas = {'STRAIGHT': 0, 'PREMUL': 1, 'CHANNEL_PACKED': 2, 'NONE': 3}
    for image in actual:
        if 'nishita' in image or image.get('name') not in by_name:
            raise ValueError('unresolved actual image identity: ' + str(image))
        name = image['name']
        native = by_name[name]
        exported_image = exports[name]
        if native['state_flags'] & 6 or native['animated'] or native['frame'] != 0 or native['miplevel_offset'] != 0 or native['tile_number'] != 0:
            raise ValueError('unadmitted native image state: ' + name)
        if native['metadata_flags'] & 48 or native['tile_size'] != 0:
            raise ValueError('unadmitted native tiled image: ' + name)
        if exported_image['source'] not in ('FILE', 'GENERATED'):
            raise ValueError('unadmitted exported image source: ' + name)
        for key in ('interpolation', 'extension', 'alpha_type', 'width', 'height'):
            if native[key] != image[key]:
                raise ValueError('native image ' + key + ' mismatch: ' + name)
        for key in ('width', 'height'):
            if exported_image[key] != image[key]:
                raise ValueError('export image ' + key + ' mismatch: ' + name)
        if colors[exported_image['colorspace']] != image['color_space'] or alphas[exported_image['alpha_mode']] != image['alpha_type']:
            raise ValueError('imported image color/alpha mismatch: ' + name)
        if image['load_failed'] != exported_image.get('load_failed', False):
            raise ValueError('imported image load state mismatch: ' + name)
        # A missing image has no decoded pixels and bypasses conversion. Its
        # native zero-sized metadata is retained, not replaced with a texture.
        if image['load_failed']:
            if native['width'] != 0 or native['height'] != 0 or native['channels'] != 0:
                raise ValueError('native missing-image metadata mismatch: ' + name)
        else:
            if native['channels'] != 4 or min(native['width'], native['height']) <= 0:
                raise ValueError('unadmitted native image dimensions/channels: ' + name)
            spaces = {0: 'data', 1: 'scene_linear', 2: 'scene_linear_srgb'}
            if spaces[image['color_space']] != native['final_colorspace']:
                raise ValueError('native final colorspace mismatch: ' + name)
            if bool(native['metadata_flags'] & 1) != (image['color_space'] == 2):
                raise ValueError('native sRGB metadata mismatch: ' + name)
        if native['requested_colorspace'] and colors[native['requested_colorspace']] != image['color_space']:
            raise ValueError('native requested colorspace mismatch: ' + name)
        matches[image['id']] = native['id']
        rows.append({'psycles_id': image['id'], 'cycles_id': native['id'],
                     'name': name, 'load_failed': image['load_failed'],
                     'cycles': native, 'psycles': image,
                     'export': exported_image})
    if len(set(matches.values())) != len(matches):
        raise ValueError('image mapping is not one-to-one')
    return matches, rows


def compare_words(original, actual, layouts, image_map, original_attrs, actual_attrs):
    c_nodes, c_fields = decode(original, layouts)
    p_nodes, p_fields = decode(actual, layouts)
    if c_nodes != p_nodes or c_fields != p_fields:
        raise ValueError('different typed instruction layout')
    # Inspect every reference, including numerically equal IDs: equality of
    # two allocator counters does not imply equality of their bound resources.
    references = []
    nonresource_changes = []
    for offset, (c, p) in enumerate(zip(original, actual)):
        field = c_fields.get(offset * 4)
        if field in RESOURCE_FIELDS:
            if any(c_fields.get(offset * 4 + byte) != field for byte in range(4)):
                raise ValueError('resource field is not a complete uint32 word')
            if field in {'SVMNodeTexImage.id', 'SVMNodeTexImageBox.id', 'SVMNodeTexEnvironment.id'}:
                # KERNEL_IMAGE_NONE is a semantic sentinel, not an allocated ID.
                same = c == p == 0x7fffffff or image_map.get(p) == c
                identity = 'image'
            elif field == 'SVMNodeTexSkyNishitaData.texture_id':
                raise ValueError('generated sky image identity is not admitted here')
            else:
                # ATTR_STD_NUM = 35 in the pinned original and host ABI tests.
                # Named IDs must resolve even when their numeric values agree.
                if c < 35 or p < 35:
                    same = c == p
                else:
                    same = c in original_attrs and p in actual_attrs and original_attrs[c] == actual_attrs[p]
                identity = 'attribute'
            references.append({'word': offset, 'field': field, 'kind': identity,
                               'cycles_id': c, 'psycles_id': p, 'same_identity': same})
        elif c != p:
            nonresource_changes.append({'word': offset, 'cycles': c, 'psycles': p, 'field': field})
    return {'same_typed_layout': True, 'references': references,
            'nonresource_changes': nonresource_changes,
            'same_binding_semantics': not nonresource_changes and all(r['same_identity'] for r in references)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('layouts', 'word_audit', 'cycles', 'psycles', 'original_images',
                     'original_attributes', 'actual_bindings', 'scene'):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    original_images = image_bindings(args.original_images.read_bytes())
    original_attrs = attribute_bindings(args.original_attributes.read_bytes())
    actual = psycles_bindings(args.actual_bindings.read_bytes())
    exported = json.loads(args.scene.read_text())
    image_map, images = match_images(original_images, actual['images'], exported['images'])
    ca = {row['id']: row['name'] for row in original_attrs}
    pa = {row['id']: row['name'] for row in actual['attributes']}
    if set(ca.values()) != set(pa.values()):
        raise ValueError('different named attribute sets')
    metadata = json.loads(args.layouts.read_text())
    _, originals = read_dump(args.cycles)
    _, actuals = read_dump(args.psycles)
    audit = json.loads(args.word_audit.read_text())
    shaders = []
    for shader in audit['shaders']:
        result = compare_words(originals[shader['cycles_index']], actuals[shader['index']],
                               metadata, image_map, ca, pa)
        shaders.append({'name': shader['name'], 'index': shader['index'],
                        'cycles_index': shader['cycles_index'], **result})
    counts = Counter(r['kind'] for shader in shaders for r in shader['references'])
    summary = {'shaders': len(shaders), 'images': len(images), 'named_attributes': len(pa),
               'references': dict(counts),
               'equal_binding_semantics': sum(s['same_binding_semantics'] for s in shaders),
               'reference_mismatches': sum(not r['same_identity'] for s in shaders for r in s['references']),
               'nonresource_word_mismatches': sum(len(s['nonresource_changes']) for s in shaders)}
    print(json.dumps({'schema': 'psycles.observed-svm-bindings.v1',
        'scope': 'typed words, stack addresses, PCs and observed resource identity; no word rewriting, pixel equivalence or render-performance claim',
        'summary': summary, 'images': images, 'attributes': {'cycles': original_attrs, 'psycles': actual['attributes']},
        'shaders': shaders, 'sources': [source(path) for path in vars(args).values()]}, indent=2))
    return int(summary['equal_binding_semantics'] != len(shaders))


if __name__ == '__main__':
    sys.exit(main())
