"""Decode actual original registry captures and reject false identity matches.

No CPU shader evaluator: the positive word images and resource tables are
captured original/production compiler output, with provenance in the fixture.
"""
import base64
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'docs/validation/2026-09-09/resource-identities'))
import decode_cycles_bindings as decoder
import audit_bindings as audit


class BindingOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((ROOT / 'tests/data/cycles_resource_bindings.json').read_text())
        cls.data = {}
        for name, capture in cls.fixture['captures'].items():
            data = base64.b64decode(capture['base64'], validate=True)
            if len(data) != capture['bytes'] or hashlib.sha256(data).hexdigest() != capture['sha256']:
                raise AssertionError('original capture integrity failure')
            cls.data[name] = data
        cls.original_images = decoder.image_bindings(cls.data['original_images'])
        cls.original_attrs = decoder.attribute_bindings(cls.data['original_attributes'])
        cls.actual = decoder.psycles_bindings(cls.data['actual'])

    def match(self, originals=None, actuals=None, exported=None):
        return audit.match_images(
            self.original_images if originals is None else originals,
            self.actual['images'] if actuals is None else actuals,
            self.fixture['exported_images'] if exported is None else exported)

    def words(self, actual=None, mapping=None):
        shader = self.fixture['shader']
        return audit.compare_words(
            shader['cycles'], shader['psycles'] if actual is None else actual,
            self.fixture['layouts'], self.match()[0] if mapping is None else mapping,
            {r['id']: r['name'] for r in self.original_attrs},
            {r['id']: r['name'] for r in self.actual['attributes']})

    def test_complete_observed_registries(self):
        self.assertEqual(len(self.original_images['images']), 174)
        self.assertEqual(len(self.actual['images']), 174)
        self.assertEqual(self.original_images['udims'], [])
        self.assertEqual(len(self.original_attrs), 21)
        self.assertEqual(len(self.actual['attributes']), 21)
        self.assertEqual({r['name'] for r in self.original_attrs},
                         {r['name'] for r in self.actual['attributes']})
        matches, rows = self.match()
        self.assertEqual(len(matches), 174)
        self.assertEqual(sum(row['load_failed'] for row in rows), 2)
        self.assertEqual(self.original_images['images'][0]['loader_name'], 'Razor_Blade_Col')
        self.assertEqual(self.actual['images'][0]['name'], 'spiderwebs.png')

    def test_original_complete_shader_uses_resolved_identities(self):
        self.assertNotEqual(self.fixture['shader']['cycles'], self.fixture['shader']['psycles'])
        result = self.words()
        self.assertTrue(result['same_binding_semantics'])
        self.assertEqual(len(result['references']), 4)
        self.assertEqual(result['nonresource_changes'], [])

    def test_same_integer_is_not_resource_equivalence(self):
        for field in ('SVMNodeAttr.attr', 'SVMNodeTexImage.id'):
            reference = next(r for r in self.words()['references']
                             if r['field'] == field and r['cycles_id'] != r['psycles_id'])
            words = list(self.fixture['shader']['psycles'])
            words[reference['word']] = reference['cycles_id']
            with self.subTest(field=field):
                self.assertFalse(self.words(actual=words)['same_binding_semantics'])
        mapping, _ = self.match()
        reference = next(r for r in self.words()['references']
                         if r['kind'] == 'image' and r['cycles_id'] == r['psycles_id'])
        mapping[reference['psycles_id']] += 1
        self.assertFalse(self.words(mapping=mapping)['same_binding_semantics'])

    def test_unknown_resource_and_nonresource_words_are_not_normalized(self):
        reference = next(r for r in self.words()['references'] if r['kind'] == 'image')
        words = list(self.fixture['shader']['psycles'])
        words[reference['word']] = 123456
        self.assertFalse(self.words(actual=words)['same_binding_semantics'])
        _, fields = audit.decode(self.fixture['shader']['cycles'], self.fixture['layouts'])
        # Flip a table float, not an opcode or control target. Even one bit in
        # literal stream data is not excused by a resource identity comparison.
        offset = next(byte // 4 for byte, field in fields.items() if field.endswith('.table_float'))
        words = list(self.fixture['shader']['psycles'])
        words[offset] ^= 1
        result = self.words(actual=words)
        self.assertFalse(result['same_binding_semantics'])
        self.assertEqual(len(result['nonresource_changes']), 1)

    def test_sampler_metadata_mismatch_rejected(self):
        for field in ('interpolation', 'extension', 'alpha_type', 'width', 'height', 'color_space', 'load_failed'):
            actual = copy.deepcopy(self.actual['images'])
            actual[0][field] = 1 if field == 'load_failed' else actual[0][field] ^ 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.match(actuals=actual)

    def test_ambiguous_names_and_unadmitted_state_rejected(self):
        exported = copy.deepcopy(self.fixture['exported_images'])
        exported.append(exported[0])
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.match(exported=exported)
        for field, value in (('animated', True), ('frame', 1), ('miplevel_offset', 1),
                             ('state_flags', 3), ('tile_number', 1001),
                             ('metadata_flags', 35), ('final_colorspace', 'data')):
            originals = copy.deepcopy(self.original_images)
            originals['images'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.match(originals=originals)

    def test_protocol_bounds_version_and_trailing_data(self):
        for key, read in (('original_images', decoder.image_bindings),
                          ('original_attributes', decoder.attribute_bindings),
                          ('actual', decoder.psycles_bindings)):
            data = self.data[key]
            for cut in (0, 7, 8, 11, 15, 16, 20, 31, len(data) - 1):
                with self.subTest(key=key, cut=cut), self.assertRaises(ValueError):
                    read(data[:cut])
            for bad in (b'INVALID!' + data[8:], data[:8] + struct.pack('<I', 2) + data[12:], data + b'x'):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    read(bad)

    def test_dense_slot_presence_and_finite_frame_validated(self):
        for offset, value in ((16, 1), (20, 2), (24, 7), (28, 8), (48, 0x7f800000)):
            data = bytearray(self.data['original_images'])
            struct.pack_into('<I', data, offset, value)
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                decoder.image_bindings(bytes(data))


if __name__ == '__main__':
    unittest.main()
