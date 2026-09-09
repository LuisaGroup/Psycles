"""Validate immutable original captures, never evaluate portal transport."""
import hashlib
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/cycles_ray_portal_render"
NAMES = ("transparent-control", "ray-portal", "portal-depth-world", "portal-depth-surface",
         "portal-depth-chain", "portal-signed-weight", "portal-transparent-mix", "portal-limit-zero",
         "portal-depth-nee", "portal-depth-shadow", "portal-default-position")


class PortalFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((DATA / "manifest.json").read_text())

    def test_original_input_bytes(self):
        self.assertEqual(self.manifest["schema"], "psycles.cycles-ray-portal-render-fixture.v1")
        self.assertEqual(tuple(row["name"] for row in self.manifest["records"]), NAMES)
        for name, digest in self.manifest["fixture_sha256"].items():
            self.assertEqual(hashlib.sha256((DATA / name).read_bytes()).hexdigest(), digest, name)
        for row in self.manifest["records"]:
            name = row["name"]
            scene = (DATA / (name + "-scene.json")).read_bytes()
            self.assertEqual(hashlib.sha256(scene).hexdigest(), row["scene"]["sha256"])
            geometry = bytes.fromhex((DATA / (name + "-geometry.txt")).read_text())
            self.assertEqual(len(geometry), row["geometry_bytes"])
            self.assertEqual(hashlib.sha256(geometry).hexdigest(), row["geometry"]["sha256"])

    def test_original_render_domain(self):
        for row in self.manifest["records"]:
            settings = row["original_settings"]
            self.assertEqual(settings["cycles_compute_device_type"], "HIP")
            self.assertEqual(settings["cycles_device"], "GPU")
            self.assertEqual(settings["blender_build"]["build_hash"], "9e2066aef7ef")
            self.assertEqual(settings["blender_build"]["version_tuple"], [5, 2, 1])
            self.assertEqual((settings["width"], settings["height"], settings["samples"]), (16, 16, 1))
            self.assertEqual(settings["effective_seed"], 0)

    def test_complete_finite_films(self):
        for name in NAMES:
            rows = (DATA / (name + "-film.txt")).read_text().splitlines()
            self.assertEqual(len(rows), 256)
            for i, row in enumerate(rows):
                values = row.split()
                self.assertEqual(len(values), 5)
                self.assertEqual(tuple(map(int, values[:2])), (i % 16, i // 16))
                self.assertTrue(all(map(math.isfinite, map(float, values[2:]))))

    def test_position_is_a_real_link_for_relocation_cases(self):
        for name in NAMES:
            scene = json.loads((DATA / (name + "-scene.json")).read_text())
            for material in scene["materials"]:
                tree = material["node_tree"]
                for node in tree["nodes"]:
                    if node["bl_idname"] != "ShaderNodeBsdfRayPortal":
                        continue
                    linked = any(link["to_node"] == node["name"] and
                                 link["to_socket"] == "Position" for link in tree["links"])
                    self.assertEqual(linked, name not in ("ray-portal", "portal-default-position"))
                    position = next(socket for socket in node["inputs"] if socket["name"] == "Position")
                    self.assertEqual(position["linked"], linked)

    def test_gpu_state_provenance_and_complete_records(self):
        provenance = json.loads((ROOT / "tests/data/cycles_ray_portal_state.json").read_text())
        for relative, expected in provenance["fixture_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)
        self.assertEqual(provenance["source_revision"], "cb168525138fecc792cc393f94afc39582b0103c")
        self.assertTrue(provenance["native_fast_math"])
        rows = (ROOT / "tests/data/cycles_ray_portal_state.txt").read_text().splitlines()
        self.assertEqual(len(rows), 26)
        for i, row in enumerate(rows):
            values = row.split()
            self.assertEqual(len(values), 45)
            self.assertEqual(int(values[0]), i)
            self.assertTrue(all(map(math.isfinite, map(float, values[1:29]))))
            self.assertTrue(all(0 <= int(v) <= 0xffffffff for v in values[29:]))


if __name__ == "__main__":
    unittest.main()
