"""Validate original Holdout domain inputs and words without evaluating a shader."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import unittest

DATA = Path(__file__).resolve().parent / "data/cycles_holdout_volume"
NAMES = ("holdout-only-volume", "holdout-emission-volume", "holdout-shared-domains")
REVISION = "cb168525138fecc792cc393f94afc39582b0103c"


class HoldoutVolumeFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((DATA / "manifest.json").read_text())

    def identity(self, value):
        self.assertTrue(Path(value["path"]).is_absolute())
        self.assertRegex(value["sha256"], r"^[0-9a-f]{64}$")

    def test_fixture_hashes_and_original_input_bytes(self):
        self.assertEqual(self.manifest["schema"], "psycles.cycles-holdout-volume-fixture.v1")
        self.assertEqual(tuple(row["name"] for row in self.manifest["records"]), NAMES)
        self.assertEqual(set(self.manifest["fixture_sha256"]), {"scene.json", "geometry.txt", "words.txt"})
        for name, digest in self.manifest["fixture_sha256"].items():
            self.assertEqual(hashlib.sha256((DATA / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(hashlib.sha256((DATA / "scene.json").read_bytes()).hexdigest(),
                         self.manifest["scene"]["sha256"])
        header, *lines = (DATA / "geometry.txt").read_text().splitlines()
        self.assertRegex(header, r"^[0-9]+$")
        tokens = " ".join(lines).split()
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{2}", token) for token in tokens))
        geometry = bytes.fromhex(" ".join(tokens))
        self.assertEqual(len(geometry), int(header))
        self.assertEqual(geometry[:8], b"PSYGEO2\0")
        self.assertEqual(hashlib.sha256(geometry).hexdigest(), self.manifest["geometry"]["sha256"])

    def test_original_source_and_process_identities(self):
        self.assertEqual(self.manifest["source"]["revision"], REVISION)
        self.assertEqual(self.manifest["word_observer_build"]["build_hash"], REVISION[:12])
        self.assertEqual(self.manifest["word_observer_build"]["version_tuple"], [5, 2, 1])
        source = self.manifest["source"]
        self.assertEqual(set(source["file_sha256"]), {
            "intern/cycles/scene/svm.cpp", "intern/cycles/scene/shader_graph.cpp",
            "intern/cycles/scene/shader_nodes.cpp", "intern/cycles/scene/shader_nodes.h",
            "intern/cycles/kernel/svm/svm.h", "intern/cycles/kernel/svm/closure.h"})
        for digest in source["file_sha256"].values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(source["dirty_status"], [
            " M intern/cycles/scene/light.cpp", " M intern/cycles/scene/svm.cpp"])
        self.assertEqual(source["file_sha256"]["intern/cycles/scene/svm.cpp"],
                         "b0c7db6de10d5e77a26159015f16b633783032749e687ed9a7d3cd3658492e9e")
        for key in ("blend", "scene", "geometry", "words", "word_observer_metadata"):
            self.identity(self.manifest[key])
        producers = self.manifest["producers"]
        self.assertEqual(len(producers), 2)
        for producer in producers:
            self.identity(producer)
        self.assertNotEqual(producers[0]["path"], producers[1]["path"])
        processes = self.manifest["processes"]
        self.assertEqual([process["label"] for process in processes], ["author", "export", "word-observer"])
        for process in processes:
            self.assertEqual(process["status"], 0)
            self.identity(process["log"])
            command = process["command"]
            self.assertEqual(command[command.index("--threads") + 1], "32")
            self.assertEqual(command[command.index("--python-exit-code") + 1], "1")
            self.assertIn(self.manifest["blend"]["path"], command)
            observer = process["label"] == "word-observer"
            self.assertEqual(command[0], producers[int(observer)]["path"])
            if observer:
                self.assertEqual(command[command.index("--cycles-device") + 1], "HIP")
                self.assertEqual(command[command.index("--device-name") + 1], "Radeon RX 9070 XT")

    def test_exact_local_image_domains(self):
        self.assertEqual(self.manifest["scope"],
                         "Compiler word oracle only: three ShaderJump targets relocated; every other word unchanged.")
        tokens = iter(shlex.split((DATA / "words.txt").read_text()))
        self.assertEqual(int(next(tokens)), 3)
        for row in self.manifest["records"]:
            self.assertEqual(next(tokens), row["name"])
            self.assertEqual(int(next(tokens)), row["shader_index"])
            self.assertEqual(int(next(tokens)), row["word_count"])
            words = []
            for _ in range(row["word_count"]):
                token = next(tokens)
                self.assertRegex(token, r"^[0-9a-f]{8}$")
                words.append(int(token, 16))
            self.assertGreaterEqual(len(words), 7)
            self.assertEqual(words[0:2], [1, 4])
            self.assertTrue(all(4 <= target < len(words) for target in words[1:4]))
            self.assertEqual(words[words[1]] != 0, row["name"] == "holdout-shared-domains")
            self.assertEqual(words[words[2]] != 0, row["name"] != "holdout-only-volume")
            self.assertEqual(words[words[3]], 0)
        self.assertIsNone(next(tokens, None))

    def test_single_native_holdout_and_authored_domain_links(self):
        scene = json.loads((DATA / "scene.json").read_text())
        self.assertEqual(scene["blender_build"]["build_hash"], "9e2066aef7ef")
        self.assertEqual((scene["render"]["width"], scene["render"]["height"]), (16, 16))
        for row in self.manifest["records"]:
            name = row["name"]
            material, = [material for material in scene["materials"] if material["name"] == name]
            self.assertEqual(material["cycles_sync"]["shader_index"], row["shader_index"])
            tree = material["node_tree"]
            holdout, = [node for node in tree["nodes"] if node["type"] == "HOLDOUT"]
            emission, = [node for node in tree["nodes"] if node["type"] == "EMISSION"]
            output, = [node for node in tree["nodes"] if node["type"] == "OUTPUT_MATERIAL"]
            roots = [link for link in tree["links"] if link["to_node"] == output["name"]]
            self.assertEqual({link["to_socket"] for link in roots},
                             {"Surface", "Volume"} if name == "holdout-shared-domains" else {"Volume"})
            outgoing = [link for link in tree["links"] if link["from_node"] == holdout["name"]]
            self.assertEqual(len(outgoing), 2 if name == "holdout-shared-domains" else 1)
            emitted = [link for link in tree["links"] if link["from_node"] == emission["name"]]
            if name == "holdout-only-volume":
                self.assertEqual(outgoing[0]["to_node"], output["name"])
                self.assertFalse(emitted)
            else:
                mix, = [node for node in tree["nodes"] if node["type"] == "MIX_SHADER"]
                self.assertEqual(len(emitted), 1)
                self.assertEqual(emitted[0]["to_node"], mix["name"])
                self.assertTrue(any(link["to_node"] == mix["name"] for link in outgoing))
                if name == "holdout-shared-domains":
                    self.assertTrue(any(link["to_node"] == output["name"] and
                                        link["to_socket"] == "Surface" for link in outgoing))


if __name__ == "__main__":
    unittest.main()
