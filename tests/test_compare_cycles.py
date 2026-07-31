#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


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


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
