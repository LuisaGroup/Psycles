"""Verify original Holdout capture integrity, not a host shader or transport oracle."""
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/cycles_holdout"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "tools"))
from render_pass_contract import PASSES, validate_channels

NAMES = (
    "ordinary-emission", "object-emission", "object-transparent",
    "object-mixed-emission", "object-mixed-diffuse", "object-colored-transparent",
    "object-mixed-opaque", "object-secondary-emission", "node-holdout",
    "node-mixed-emission", "node-mixed-transparent", "node-dynamic-holdout",
)
REVISION = "cb168525138fecc792cc393f94afc39582b0103c"


class HoldoutFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((DATA / "manifest.json").read_text())

    def identity(self, value):
        self.assertTrue(Path(value["path"]).is_absolute())
        self.assertRegex(value["sha256"], r"^[0-9a-f]{64}$")

    def test_complete_fixture_hashes_and_original_input_bytes(self):
        self.assertEqual(self.manifest["schema"], "psycles.cycles-holdout-fixture.v1")
        self.assertEqual(sorted(row["name"] for row in self.manifest["records"]), sorted(NAMES))
        expected = {name + suffix for name in NAMES for suffix in
                    ("-scene.json", "-geometry.txt", "-film.txt", "-words.txt")}
        self.assertEqual(set(self.manifest["fixture_sha256"]), expected)
        for name, digest in self.manifest["fixture_sha256"].items():
            self.assertEqual(hashlib.sha256((DATA / name).read_bytes()).hexdigest(), digest, name)
        for row in self.manifest["records"]:
            name = row["name"]
            scene = (DATA / (name + "-scene.json")).read_bytes()
            self.assertEqual(hashlib.sha256(scene).hexdigest(), row["scene"]["sha256"])
            header, *lines = (DATA / (name + "-geometry.txt")).read_text().splitlines()
            self.assertRegex(header, r"^[0-9]+$")
            tokens = " ".join(lines).split()
            self.assertTrue(all(re.fullmatch(r"[0-9a-f]{2}", token) for token in tokens))
            geometry = bytes.fromhex(" ".join(tokens))
            self.assertEqual(len(geometry), int(header))
            self.assertEqual(len(geometry), row["geometry_bytes"])
            self.assertEqual(geometry[:8], b"PSYGEO2\0")
            self.assertEqual(hashlib.sha256(geometry).hexdigest(), row["geometry"]["sha256"])

    def test_word_observer_provenance(self):
        source = self.manifest["word_source"]
        self.assertEqual(source["revision"], REVISION)
        self.assertEqual(set(source["source_sha256"]), {
            "intern/cycles/scene/svm.cpp", "intern/cycles/scene/shader_graph.cpp",
            "intern/cycles/scene/shader_nodes.cpp", "intern/cycles/scene/shader_nodes.h",
            "intern/cycles/kernel/svm/svm.h", "intern/cycles/kernel/svm/closure.h",
            "intern/cycles/kernel/integrator/surface_shader.h",
            "intern/cycles/kernel/integrator/shade_surface.h", "intern/cycles/scene/light.cpp"})
        for digest in source["source_sha256"].values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(source["dirty_status"], [
            " M intern/cycles/scene/light.cpp", " M intern/cycles/scene/svm.cpp"])
        # Dirty observer source must never be described as the clean revision.
        self.assertEqual(source["source_sha256"]["intern/cycles/scene/svm.cpp"],
                         "b0c7db6de10d5e77a26159015f16b633783032749e687ed9a7d3cd3658492e9e")
        self.assertEqual(source["source_sha256"]["intern/cycles/scene/light.cpp"],
                         "6075fa6cb941548af4ce9dffc8ab34233860709095bc71517ad4bc7742747fdb")
        for row in self.manifest["records"]:
            self.assertEqual(row["word_observer_build"]["build_hash"], REVISION[:12])
            self.assertEqual(row["word_observer_build"]["version_tuple"], [5, 2, 1])
            for key in ("blend", "original", "metadata", "words", "scene", "geometry",
                        "word_observer_metadata"):
                self.identity(row[key])

    def test_capture_process_identities(self):
        capture = self.manifest["capture"]
        self.identity(capture["record"])
        self.assertEqual(len(capture["producers"]), 2)
        for producer in capture["producers"]:
            self.identity(producer)
        production, observer = (value["path"] for value in capture["producers"])
        self.assertNotEqual(production, observer)
        processes = capture["processes"]
        self.assertEqual(len(processes), 36)
        self.assertEqual(len({p["log"]["path"] for p in processes}), 36)
        for row in self.manifest["records"]:
            matching = [p for p in processes if row["blend"]["path"] in p["command"]]
            self.assertEqual(len(matching), 3)
            self.assertEqual(sum(p["svm_dump"] is not None for p in matching), 1)
            for process in matching:
                self.identity(process["log"])
                self.assertEqual(process["status"], 0)
                command = process["command"]
                self.assertEqual(command[command.index("--threads") + 1], "32")
                self.assertEqual(command[command.index("--python-exit-code") + 1], "1")
                if process["svm_dump"] is not None:
                    self.assertEqual(process["svm_dump"], row["words"]["path"])
                    self.assertEqual(command[0], observer)
                else:
                    self.assertEqual(command[0], production)
                if "--cycles-device" in command:
                    self.assertEqual(command[command.index("--cycles-device") + 1], "HIP")
                    self.assertEqual(command[command.index("--device-name") + 1], "Radeon RX 9070 XT")
                else:
                    self.assertIn(row["bundle"], command)

    def test_original_render_domain(self):
        for row in self.manifest["records"]:
            settings = row["original_settings"]
            self.assertEqual(settings["cycles_device"], "GPU")
            self.assertEqual(settings["cycles_compute_device_type"], "HIP")
            self.assertEqual(settings["blender_build"]["build_hash"], "9e2066aef7ef")
            self.assertEqual(settings["blender_build"]["version_tuple"], [5, 2, 1])
            self.assertEqual((settings["width"], settings["height"], settings["samples"]), (16, 16, 1))
            self.assertEqual(settings["effective_seed"], 0)

    def test_complete_finite_film_channels(self):
        self.assertEqual(self.manifest["passes"], list(PASSES))
        self.assertEqual(len(PASSES), 15)
        channels = ["ViewLayer." + name + "." + lane for name in PASSES
                    for lane in ("RGBA" if name == "Combined" else "XYZ" if name == "Normal" else "RGB")]
        self.assertEqual(self.manifest["channels"], channels)
        self.assertEqual(len(channels), 46)
        validate_channels(channels)
        for name in NAMES:
            rows = (DATA / (name + "-film.txt")).read_text().splitlines()
            self.assertEqual(len(rows), 256)
            for i, row in enumerate(rows):
                fields = row.split()
                self.assertEqual(len(fields), 48)
                self.assertEqual(tuple(map(int, fields[:2])), (i % 16, i // 16))
                self.assertTrue(all(map(math.isfinite, map(float, fields[2:]))))

    def test_local_word_image_structure(self):
        self.assertEqual(self.manifest["word_scope"],
                         "Only ShaderJump relocation; all typed payload words copied from original Cycles.")
        for row in self.manifest["records"]:
            header, *payload = (DATA / (row["name"] + "-words.txt")).read_text().split()
            self.assertRegex(header, r"^[0-9]+$")
            self.assertTrue(all(re.fullmatch(r"[0-9a-f]{8}", word) for word in payload))
            words = [int(word, 16) for word in payload]
            self.assertEqual(len(words), int(header))
            self.assertEqual(len(words), row["word_count"])
            self.assertGreaterEqual(len(words), 7)
            self.assertEqual(words[0], 1)  # Native NODE_SHADER_JUMP ABI tag.
            self.assertEqual(words[1], 4)
            self.assertTrue(all(4 <= target < len(words) for target in words[1:4]))
            # These authored fixtures have no volume or displacement roots.
            self.assertEqual(words[words[2]], 0)
            self.assertEqual(words[words[3]], 0)

    def test_authored_holdout_and_domain_controls(self):
        for name in NAMES:
            scene = json.loads((DATA / (name + "-scene.json")).read_text())
            render, cycles = scene["render"], scene["render"]["cycles"]
            self.assertEqual(render["transparent"], name != "object-mixed-opaque")
            self.assertEqual((render["width"], render["height"], render["percentage"]), (16, 16, 100))
            self.assertEqual((render["pixel_filter_type"], render["filter_width"]), ("BOX", 1.0))
            self.assertEqual((cycles["samples"], cycles["effective_seed"]), (1, 0))
            self.assertEqual(cycles["max_bounces"], int(name == "object-secondary-emission"))
            self.assertEqual(cycles["transparent_max_bounces"], 8)
            self.assertEqual(cycles["volume_bounces"], 0)
            for key in ("use_adaptive_sampling", "use_denoising", "use_light_tree"):
                self.assertFalse(cycles[key])
            primary, = [instance for instance in scene["instances"] if instance["name"] == name]
            self.assertEqual(primary["use_holdout"], name.startswith("object-"))
            self.assertEqual(primary["transform"][14], 6 if name == "object-secondary-emission" else 3)
            self.assertEqual(len(scene["instances"]), 2 if name == "object-secondary-emission" else 1)
            material, = [material for material in scene["materials"] if material["name"] == name]
            tree = material["node_tree"]
            holdout, = [node for node in tree["nodes"] if node["type"] == "HOLDOUT"]
            outgoing = [link for link in tree["links"] if link["from_node"] == holdout["name"]]
            self.assertEqual(bool(outgoing), name.startswith("node-"))
            outputs = {node["name"] for node in tree["nodes"] if node["type"] == "OUTPUT_MATERIAL"}
            roots = [link for link in tree["links"] if link["to_node"] in outputs]
            self.assertEqual(len(roots), 1)
            self.assertEqual(roots[0]["to_socket"], "Surface")
            dynamic = [link for link in tree["links"] if link["to_socket"] == "Fac"]
            self.assertEqual(bool(dynamic), name == "node-dynamic-holdout")


if __name__ == "__main__":
    unittest.main()
