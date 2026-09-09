"""Check original GPU capture provenance and controls, not host BSDF math."""

import hashlib
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
NAMES = [
    "diffuse_identity_frequency",
    "diffuse_positive_frequency",
    "translucent_identity_frequency",
    "translucent_positive_frequency",
    "label_none_positive_frequency",
    "diffuse_bump_identity_frequency",
]


class ShadingTerminatorFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provenance = json.loads(
            (ROOT / "tests/data/cycles_shading_terminator.json").read_text())
        lines = (ROOT / "tests/data/cycles_shading_terminator.txt").read_text().splitlines()
        if len(lines) != 7 or lines[0] != "cycles-shading-terminator 1 6 9 3":
            raise ValueError("Incomplete or incompatible original GPU capture")
        cls.rows = []
        for index, line in enumerate(lines[1:]):
            fields = line.split()
            if len(fields) != 50 or fields[:2] != [str(index), NAMES[index]]:
                raise ValueError(f"Invalid original row identity/shape: {index}")
            values = list(map(float, fields[2:38]))
            integers = list(map(int, fields[38:50]))
            cls.rows.append(([values[i:i + 4] for i in range(0, 36, 4)], integers))

    def test_original_identity_and_hashes(self):
        p = self.provenance
        self.assertEqual(p["schema"], "psycles.cycles-shading-terminator.v1")
        self.assertEqual(p["source_revision"], "cb168525138fecc792cc393f94afc39582b0103c")
        self.assertTrue(p["native_fast_math"])
        self.assertEqual(p["device_arch"], "gfx1201")
        self.assertEqual(p["compiler_jobs"], 32)
        self.assertEqual(p["cases"], NAMES)
        self.assertEqual((p["rows"], p["float_lanes"], p["integer_lanes"]), (6, 36, 12))
        self.assertTrue(p["original_kernel_files_match_head"])
        self.assertEqual((p["compile_exit"], p["capture_exit"]), (0, 0))
        self.assertEqual(set(p["fixture_sha256"]), {
            "tests/cycles_shading_terminator_fixture.h",
            "tools/cycles_shading_terminator_oracle.hip",
            "tests/data/cycles_shading_terminator.txt",
        })
        for flag in ["-parallel-jobs=32", "--offload-arch=gfx1201", "-O3", "-ffast-math"]:
            self.assertIn(flag, p["command"])
        for relative, expected in p["fixture_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected, relative)

    def test_all_rows_finite_defined_and_state_preserving(self):
        for i, (values, integers) in enumerate(self.rows):
            with self.subTest(case=NAMES[i]):
                self.assertTrue(all(math.isfinite(x) for row in values for x in row))
                self.assertTrue(all(0 <= x <= 0xffffffff for x in integers))
                self.assertEqual(integers[:4], integers[4:8])
                self.assertEqual(integers[1:3], [1, 0])
                self.assertEqual(integers[9], i)  # Valid native object index.
                self.assertEqual(integers[10], 1)  # PRIMITIVE_TRIANGLE.
                self.assertEqual(integers[11], [0, 0, 1, 1, 2, 0][i])
                self.assertEqual(values[0][3], [1, 2, 1, 2, 2, 1][i])

    def test_reflection_frequency_is_a_real_discriminator(self):
        identity, positive = self.rows[0][0], self.rows[1][0]
        self.assertEqual(self.rows[0][1][8], self.rows[1][1][8])
        self.assertTrue(self.rows[0][1][8] & 2)  # LABEL_REFLECT.
        self.assertFalse(self.rows[0][1][8] & 1)  # Not LABEL_TRANSMIT.
        for output in (3, 6):  # Sampled and evaluated RGB, unchanged PDF.
            self.assertGreater(identity[output][0], 0)
            self.assertGreaterEqual(positive[output][0], 0)
            self.assertLess(positive[output][0], identity[output][0])
            self.assertEqual(identity[output][3], positive[output][3])
        self.assertEqual(identity[4:6], positive[4:6])

    def test_transmission_frequency_control(self):
        self.assertEqual(self.rows[2][0][1:], self.rows[3][0][1:])
        self.assertEqual(self.rows[2][1][8], self.rows[3][1][8])
        self.assertTrue(self.rows[2][1][8] & 1)
        self.assertFalse(self.rows[2][1][8] & 2)

    def test_label_none_control(self):
        values, integers = self.rows[4]
        self.assertEqual(integers[8], 0)
        self.assertEqual(values[3], [0, 0, 0, 0])
        self.assertEqual(values[6], [0, 0, 0, 0])

    def test_bump_correction_remains_active_at_identity_frequency(self):
        values, integers = self.rows[5]
        self.assertTrue(integers[0] & (1 << 15))
        self.assertEqual(values[0][3], 1)
        self.assertTrue(all(0 <= value < 1 for value in values[8][:2]))
        self.assertGreater(values[6][0], 0)


if __name__ == "__main__":
    unittest.main()
