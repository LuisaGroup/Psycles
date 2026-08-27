#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np


def load_module(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(
        "psycles_compare_cycles", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChannelResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.comparison = load_module(
            pathlib.Path(sys.argv[1]).resolve()
        )

    def test_flat_exr_combined_channels(self) -> None:
        self.assertEqual(
            self.comparison._REPORT_SCHEMA,
            "psycles.cycles-differential.v2",
        )
        self.assertEqual(
            self.comparison._find_cycles_channels(
                ["R", "G", "B", "A"], "Combined"
            ),
            [0, 1, 2],
        )

    def test_named_multilayer_combined_channels(self) -> None:
        self.assertEqual(
            self.comparison._find_cycles_channels(
                [
                    "ViewLayer.Combined.R",
                    "ViewLayer.Combined.G",
                    "ViewLayer.Combined.B",
                    "ViewLayer.Combined.A",
                ],
                "Combined",
            ),
            [0, 1, 2],
        )

    def test_flat_channels_do_not_alias_named_passes(self) -> None:
        with self.assertRaises(RuntimeError):
            self.comparison._find_cycles_channels(
                ["R", "G", "B"], "DiffDir"
            )

    def test_psycles_compact_volume_pass_names_are_aliases(self) -> None:
        self.assertEqual(
            self.comparison._find_cycles_channels(
                [
                    "ViewLayer.VolumeDir.R",
                    "ViewLayer.VolumeDir.G",
                    "ViewLayer.VolumeDir.B",
                    "ViewLayer.VolumeInd.R",
                    "ViewLayer.VolumeInd.G",
                    "ViewLayer.VolumeInd.B",
                ],
                "Volume Direct",
            ),
            [0, 1, 2],
        )
        self.assertEqual(
            self.comparison._find_cycles_channels(
                [
                    "ViewLayer.VolumeDir.R",
                    "ViewLayer.VolumeDir.G",
                    "ViewLayer.VolumeDir.B",
                    "ViewLayer.VolumeInd.R",
                    "ViewLayer.VolumeInd.G",
                    "ViewLayer.VolumeInd.B",
                ],
                "Volume Indirect",
            ),
            [3, 4, 5],
        )

    def test_comparison_requires_matching_exact_blender_builds(self) -> None:
        identity = {
            "version": "5.2.0 LTS",
            "version_cycle": "release",
            "version_tuple": [5, 2, 0],
            "build_hash": "fbe6228777e7",
            "build_branch": "blender-v5.2-release",
            "build_type": "Release",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            reference = root / "cycles.json"
            actual = root / "scene.json"
            reference.write_text(
                json.dumps({"blender_build": identity}),
                encoding="utf-8",
            )
            actual.write_text(
                json.dumps({"blender_build": identity}),
                encoding="utf-8",
            )
            self.assertEqual(
                self.comparison._validate_build_metadata(
                    reference,
                    actual,
                    allow_unverified=False,
                ),
                identity,
            )

            mismatched = dict(identity)
            mismatched["build_hash"] = "ec438d7429e5"
            actual.write_text(
                json.dumps({"blender_build": mismatched}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError, "different Blender builds"
            ):
                self.comparison._validate_build_metadata(
                    reference,
                    actual,
                    allow_unverified=False,
                )

    def test_unverified_comparison_requires_explicit_diagnostic_mode(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError, "requires exact Blender build metadata"
        ):
            self.comparison._validate_build_metadata(
                None,
                None,
                allow_unverified=False,
            )
        self.assertIsNone(
            self.comparison._validate_build_metadata(
                None,
                None,
                allow_unverified=True,
            )
        )

    def test_invalid_pixels_are_attributed_to_each_renderer(self) -> None:
        reference = np.zeros((2, 2, 3), dtype=np.float32)
        actual = np.zeros_like(reference)
        reference[0, 0, 0] = np.nan
        actual[1, 1, 1] = np.inf
        metrics = self.comparison._metrics(reference, actual)
        self.assertEqual(metrics["valid_pixels"], 2)
        self.assertEqual(metrics["invalid_pixels"], 2)
        self.assertEqual(metrics["reference_invalid_pixels"], 1)
        self.assertEqual(metrics["actual_invalid_pixels"], 1)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
