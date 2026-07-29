#!/usr/bin/env python3
"""Regression tests for the Cycles/Psycles probe-runner contract."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


def _load_runner(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(
        "psycles_shader_probe_runner", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ShaderProbeRunnerContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if len(sys.argv) != 2:
            raise RuntimeError(
                "expected the probe runner path as the only argument"
            )
        cls.runner = _load_runner(pathlib.Path(sys.argv[1]))

    def test_cycles_uses_scene_owned_seed(self) -> None:
        command = self.runner._cycles_golden_command(
            "/opt/blender",
            pathlib.Path("/probe/scene.blend"),
            pathlib.Path("/tools/render_cycles_golden.py"),
            pathlib.Path("/probe/cycles.exr"),
            width=128,
            height=64,
            samples=4096,
            device="HIP",
            device_name="Radeon RX 9070 XT",
        )
        separator = command.index("--")
        device_option = command.index("--cycles-device")
        self.assertEqual(
            command[separator + 1 : device_option],
            [
                "/probe/cycles.exr",
                "128",
                "64",
                "4096",
            ],
        )

    def test_comparison_uses_declared_host_python(self) -> None:
        command = self.runner._comparison_command(
            "/venv/bin/python",
            pathlib.Path("/tools/compare_cycles.py"),
            pathlib.Path("/probe/cycles.exr"),
            pathlib.Path("/probe/report.json"),
            pathlib.Path("/probe/triptychs"),
            pathlib.Path("/probe/psycles.exr"),
        )
        self.assertEqual(command[0], "/venv/bin/python")
        self.assertEqual(
            command[1], "/tools/compare_cycles.py"
        )
        self.assertNotIn("/opt/blender", command)
        bindings = [
            item for item in command if "=" in item
        ]
        self.assertEqual(
            len(bindings), len(self.runner._REPORT_PASSES)
        )
        self.assertEqual(
            [item.split("=", 1)[0] for item in bindings],
            list(self.runner._REPORT_PASSES),
        )

    def test_indirect_principled_gate_rejects_missing_filter(self) -> None:
        report = {
            "passes": {
                "DiffInd": {"luminance_mean_ratio": 1.0592059},
                "GlossInd": {"luminance_mean_ratio": 1.0457756},
            }
        }
        failures = self.runner._probe_gate_failures(
            "indirect_principled", report
        )
        self.assertEqual(len(failures), 2)

    def test_indirect_principled_gate_accepts_aligned_transport(
        self,
    ) -> None:
        report = {
            "passes": {
                "DiffInd": {"luminance_mean_ratio": 0.9998682},
                "GlossInd": {"luminance_mean_ratio": 1.0001890},
            }
        }
        self.assertEqual(
            self.runner._probe_gate_failures(
                "indirect_principled", report
            ),
            [],
        )


if __name__ == "__main__":
    # unittest would otherwise treat the runner path as a test selector.
    runner_path = sys.argv[1:]
    sys.argv = [sys.argv[0]]
    if len(runner_path) != 1:
        raise SystemExit(
            "usage: test_shader_probe_runner.py "
            "tools/run_cycles_shader_probes.py"
        )
    # Preserve the path for setUpClass without exposing it to unittest.
    sys.argv.append(runner_path[0])
    unittest.main(argv=[sys.argv[0]])
