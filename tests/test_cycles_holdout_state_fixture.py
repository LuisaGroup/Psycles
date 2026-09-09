"""Validate original GPU capture identity/shape; never evaluate holdout shading."""
import hashlib
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class HoldoutStateFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provenance = json.loads((ROOT / "tests/data/cycles_holdout_state.json").read_text())

    def test_original_identity(self):
        self.assertEqual(self.provenance["schema"], "psycles.cycles-holdout-state.v1")
        self.assertEqual(self.provenance["source_revision"], "cb168525138fecc792cc393f94afc39582b0103c")
        self.assertTrue(self.provenance["native_fast_math"])
        self.assertEqual(self.provenance["device_arch"], "gfx1201")
        self.assertEqual(self.provenance["snapshots"], ["before_node", "after_node", "after_holdout"])
        self.assertTrue(self.provenance["preinitialized_slot_sentinels"])
        for relative, expected in self.provenance["fixture_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected, relative)

    def test_complete_finite_state_records(self):
        rows = (ROOT / "tests/data/cycles_holdout_state.txt").read_text().splitlines()
        self.assertEqual(len(rows), 36)
        self.assertEqual(self.provenance["rows"], len(rows))
        self.assertEqual(self.provenance["float_lanes"], 132)
        self.assertEqual(self.provenance["integer_lanes"], 24)
        self.assertEqual(len(self.provenance["cases"]), len(rows))
        self.assertEqual(len(set(self.provenance["cases"])), len(rows))
        for i, row in enumerate(rows):
            values = row.split()
            self.assertEqual(len(values), 157)
            self.assertEqual(int(values[0]), i)
            self.assertTrue(all(map(math.isfinite, map(float, values[1:133]))))
            self.assertTrue(all(0 <= int(v) <= 0xffffffff for v in values[133:]))


if __name__ == "__main__":
    unittest.main()
