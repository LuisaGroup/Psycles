"""Check original capture integrity, not a CPU transport implementation."""
import hashlib
import json
import math
from pathlib import Path
import unittest

DATA = Path(__file__).resolve().parent / "data/cycles_path_lifetime"


class PathLifetimeFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((DATA / "manifest.json").read_text())

    def test_original_input_bytes(self):
        self.assertEqual(self.manifest["schema"], "psycles.cycles-path-lifetime-fixture.v1")
        for name, expected in self.manifest["fixture_sha256"].items():
            self.assertEqual(hashlib.sha256((DATA / name).read_bytes()).hexdigest(), expected)
        for row in self.manifest["records"]:
            name = row["name"]
            scene = (DATA / (name + "-scene.json")).read_bytes()
            self.assertEqual(hashlib.sha256(scene).hexdigest(), row["scene"]["sha256"])
            geometry = bytes.fromhex((DATA / (name + "-geometry.txt")).read_text())
            self.assertEqual(len(geometry), row["geometry_bytes"])
            self.assertEqual(hashlib.sha256(geometry).hexdigest(), row["geometry"]["sha256"])

    def test_original_render_domain(self):
        self.assertEqual([row["name"] for row in self.manifest["records"]],
                         ["boundary-1", "boundary-2", "boundary-2-budget-control"])
        for row in self.manifest["records"]:
            settings = row["original_settings"]
            self.assertEqual(settings["cycles_compute_device_type"], "HIP")
            self.assertEqual(settings["cycles_device"], "GPU")
            self.assertEqual(settings["blender_build"]["build_hash"], "9e2066aef7ef")
            self.assertEqual(settings["blender_build"]["version_tuple"], [5, 2, 1])
            self.assertEqual((settings["width"], settings["height"], settings["samples"]), (16, 16, 1))
            self.assertEqual(settings["effective_seed"], 0)

    def test_complete_original_images_and_invariant_control(self):
        images = {}
        for capture in self.manifest["records"]:
            name = capture["name"]
            rows = (DATA / (name + "-film.txt")).read_text().splitlines()
            self.assertEqual(len(rows), 256)
            values = []
            for index, row in enumerate(rows):
                fields = row.split()
                self.assertEqual(len(fields), 5)
                self.assertEqual(tuple(map(int, fields[:2])), (index % 16, index // 16))
                rgb = list(map(float, fields[2:]))
                self.assertTrue(all(map(math.isfinite, rgb)))
                values.extend(rgb)
            images[name] = values
        tolerance = self.manifest["tolerance"]
        # Compare the two observed original-renderer controls, not a local
        # Beer-Lambert formula or actual renderer output.
        for a, b in zip(images["boundary-2"], images["boundary-2-budget-control"]):
            self.assertLessEqual(abs(a - b), tolerance["absolute"] + tolerance["relative"] * abs(a))


if __name__ == "__main__":
    unittest.main()
