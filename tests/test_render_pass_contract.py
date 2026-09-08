"""Header-only validation rejects unequal benchmark output workloads."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import render_pass_contract as contract


class PassContract(unittest.TestCase):
    def channels(self, layer="ViewLayer"):
        prefix = f"{layer}." if layer else ""
        return [f"{prefix}{name}.{component}" for name in contract.PASSES
                for component in ("RGBA" if name == "Combined" else
                                  "XYZ" if name == "Normal" else "RGB")]

    def test_exact_common_channels(self):
        for layer in ("ViewLayer", "Layer.with.dots", ""):
            with self.subTest(layer=layer):
                channels = self.channels(layer)
                self.assertEqual(len(channels), 46)
                contract.validate_channels(channels)

    def test_rejects_archived_extra_workloads(self):
        for extras in (("Depth.Z", "Debug Sample Count.X", "Mist.Z"),
                       ("Depth.Z", "Debug Sample Count.X"),
                       ("Depth.Z", "Debug Sample Count.X", "AO.R", "AO.G", "AO.B", "IndexOB.X"),
                       ("Depth.Z", "Debug Sample Count.X", "AO.R", "AO.G", "AO.B")):
            with self.subTest(extras=extras):
                with self.assertRaisesRegex(RuntimeError, "unequal benchmark pass workload"):
                    contract.validate_channels(self.channels() + [f"ViewLayer.{c}" for c in extras])

    def test_psycles_volume_channel_aliases(self):
        channels = [c.replace("Volume Direct", "VolumeDir").replace("Volume Indirect", "VolumeInd")
                    for c in self.channels()]
        contract.validate_channels(channels)
        with self.assertRaises(RuntimeError):
            contract.validate_channels(channels + ["ViewLayer.Volume Direct.R"])

    def test_blender_52_header_names(self):
        names = ("Combined", "Normal", "Diffuse Color", "Glossy Color",
                 "Transmission Color", "Diffuse Direct", "Diffuse Indirect",
                 "Glossy Direct", "Glossy Indirect", "Transmission Direct",
                 "Transmission Indirect", "Emission", "Environment",
                 "Volume Direct", "Volume Indirect")
        channels = [f"RenderLayer.{name}.{component}" for name in names
                    for component in ("RGBA" if name == "Combined" else
                                      "XYZ" if name == "Normal" else "RGB")]
        contract.validate_channels(channels)
        with self.assertRaises(RuntimeError):
            contract.validate_channels(channels + ["RenderLayer.DiffCol.R"])

    def test_rejects_missing_duplicate_and_mixed_layer(self):
        channels = self.channels()
        for invalid in (channels[:-1], channels + channels[:1],
                        [channels[0].replace("ViewLayer", "Other")] + channels[1:],
                        channels + self.channels("Other")):
            with self.subTest(channels=invalid):
                with self.assertRaises(RuntimeError):
                    contract.validate_channels(invalid)


if __name__ == "__main__":
    unittest.main()
