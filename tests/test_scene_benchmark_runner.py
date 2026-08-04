#!/usr/bin/env python3
"""Regression tests for the canonical scene benchmark matrix."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


def _load_runner(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(
        "psycles_scene_benchmark_runner", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SceneBenchmarkRunnerContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if len(sys.argv) != 2:
            raise RuntimeError(
                "expected the benchmark runner path as the only argument"
            )
        cls.runner = _load_runner(pathlib.Path(sys.argv[1]))

    def test_canonical_luisa_matrix_includes_fallback(self) -> None:
        self.assertEqual(
            self.runner._LUISA_BACKENDS,
            ("fallback", "hip", "vk"),
        )

    def test_native_apple_matrix_is_accepted(self) -> None:
        self.assertEqual(
            self.runner._backend_list("fallback,metal"),
            ("fallback", "metal"),
        )
        self.assertEqual(
            self.runner._device_key("METAL"),
            "metal",
        )
        with self.assertRaises(
            self.runner.argparse.ArgumentTypeError
        ):
            self.runner._backend_list("fallback,metal,metal")

    def test_cycles_commands_select_cpu_and_named_hip(self) -> None:
        common = {
            "blender": pathlib.Path("/opt/blender"),
            "blend": pathlib.Path("/scene/lone-monk.blend"),
            "script": pathlib.Path(
                "/src/tools/render_cycles_golden.py"
            ),
            "width": 640,
            "height": 480,
            "samples": 64,
        }
        cpu = self.runner._cycles_command(
            output=pathlib.Path("/out/cpu.exr"),
            device="CPU",
            device_name="Ryzen 9 9950X3D",
            **common,
        )
        hip = self.runner._cycles_command(
            output=pathlib.Path("/out/hip.exr"),
            device="HIP",
            device_name="Radeon RX 9070 XT",
            **common,
        )
        self.assertEqual(
            cpu[cpu.index("--cycles-device") + 1],
            "CPU",
        )
        self.assertEqual(
            hip[hip.index("--cycles-device") + 1],
            "HIP",
        )
        self.assertEqual(
            hip[hip.index("--device-name") + 1],
            "Radeon RX 9070 XT",
        )

    def test_psycles_command_preserves_matched_settings(self) -> None:
        command = self.runner._psycles_command(
            pathlib.Path("/build/psycles_render_blender_scene"),
            pathlib.Path("/out/export"),
            pathlib.Path("/out/fallback.ppm"),
            "fallback",
            width=640,
            height=480,
            samples=64,
            max_samples_per_dispatch=8,
        )
        self.assertEqual(
            command[3:],
            ["fallback", "640", "480", "64", "8"],
        )

    def test_comparison_labels_each_renderer(self) -> None:
        command = self.runner._comparison_command(
            pathlib.Path("/usr/bin/python"),
            pathlib.Path("/src/tools/compare_cycles.py"),
            pathlib.Path("/out/cycles/hip.exr"),
            pathlib.Path("/out/psycles/fallback.exr"),
            pathlib.Path("/out/report.json"),
            pathlib.Path("/out/triptychs"),
            reference_label="Cycles HIP",
            actual_label="Psycles fallback",
        )
        self.assertEqual(
            command[command.index("--reference-label") + 1],
            "Cycles HIP",
        )
        self.assertEqual(
            command[command.index("--actual-label") + 1],
            "Psycles fallback",
        )
        bindings = [item for item in command if "=" in item]
        self.assertEqual(
            len(bindings),
            len(self.runner._REPORT_PASSES),
        )

    def test_psycles_timing_parser_requires_all_phases(self) -> None:
        output = (
            "Luisa/fallback compiled 350 geometries, "
            "87543 instances, 35 materials in 2.5 s\n"
            "Luisa shader JIT completed in 1.25 s\n"
            "Rendered 640x480 at 64 spp in 40.75 s: image.ppm\n"
        )
        self.assertEqual(
            self.runner._parse_psycles_timings(output),
            {
                "scene_compile_seconds": 2.5,
                "shader_jit_seconds": 1.25,
                "render_seconds": 40.75,
            },
        )
        with self.assertRaises(RuntimeError):
            self.runner._parse_psycles_timings(
                "Rendered 640x480 at 64 spp in 40.75 s: image.ppm\n"
            )

    def test_relative_performance_uses_render_only_times(self) -> None:
        manifest = {
            "renderers": {
                "cycles": {
                    "hip": {"render_seconds": 10.0},
                    "cpu": {"render_seconds": 50.0},
                },
                "psycles": {
                    "fallback": {"render_seconds": 25.0},
                    "hip": {"render_seconds": 20.0},
                    "vk": {"render_seconds": 40.0},
                },
            }
        }
        result = self.runner._relative_performance(manifest)
        self.assertEqual(
            result["fallback"]["slowdown_vs_cycles_hip"],
            2.5,
        )
        self.assertEqual(
            result["fallback"]["speedup_over_cycles_cpu"],
            2.0,
        )
        self.assertEqual(
            result["hip"]["speedup_over_cycles_hip"],
            0.5,
        )

    def test_relative_performance_supports_gpu_only_metal(self) -> None:
        manifest = {
            "renderers": {
                "cycles": {
                    "metal": {"render_seconds": 12.0},
                },
                "psycles": {
                    "fallback": {"render_seconds": 24.0},
                    "metal": {"render_seconds": 6.0},
                },
            }
        }
        result = self.runner._relative_performance(
            manifest, "metal"
        )
        self.assertEqual(
            result["fallback"]["slowdown_vs_cycles_metal"],
            2.0,
        )
        self.assertEqual(
            result["metal"]["speedup_over_cycles_metal"],
            2.0,
        )
        self.assertNotIn("cycles_cpu", result)

    def test_resume_requires_identical_render_configuration(self) -> None:
        expected = {
            "schema": "psycles.scene-benchmark.v1",
            "matrix": {
                "cycles": ["cpu", "hip"],
                "psycles": ["fallback", "hip", "vk"],
            },
            "settings": {"width": 640, "height": 480},
            "scene": {
                "blend": "/scene.blend",
                "sha256": "blend-hash",
                "bundle": "/bundle",
            },
        }
        previous = json.loads(json.dumps(expected))
        previous["scene"]["geometry_bin_sha256"] = "geometry-hash"
        self.runner._validate_resume_configuration(previous, expected)

        changed = json.loads(json.dumps(previous))
        changed["settings"]["width"] = 1920
        with self.assertRaisesRegex(RuntimeError, "different settings"):
            self.runner._validate_resume_configuration(changed, expected)

        changed = json.loads(json.dumps(previous))
        changed["scene"]["sha256"] = "another-scene"
        with self.assertRaisesRegex(RuntimeError, "different scene"):
            self.runner._validate_resume_configuration(changed, expected)

    def test_resume_render_validates_command_output_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            output = root / "cpu.exr"
            metadata = root / "cpu.json"
            output.write_bytes(b"rendered pixels")
            metadata.write_text(
                json.dumps({"elapsed_seconds": 3.5}),
                encoding="utf-8",
            )
            command = ["blender", "scene.blend", "cpu.exr"]
            manifest = {
                "commands": {
                    "cycles_cpu": {
                        "command": command,
                        "returncode": 0,
                    }
                },
                "renderers": {
                    "cycles": {
                        "cpu": {
                            "output": str(output),
                            "sha256": self.runner._sha256(output),
                            "metadata": str(metadata),
                            "metadata_sha256": self.runner._sha256(
                                metadata
                            ),
                            "render_seconds": 3.5,
                        }
                    }
                },
            }

            def can_resume() -> bool:
                return self.runner._can_resume_render(
                    manifest,
                    command_key="cycles_cpu",
                    renderer_group="cycles",
                    renderer_key="cpu",
                    expected_command=command,
                    expected_output=output,
                    required_timings=("render_seconds",),
                    metadata_path=metadata,
                )

            self.assertTrue(can_resume())
            output.write_bytes(b"tampered pixels")
            self.assertFalse(can_resume())
            output.write_bytes(b"rendered pixels")
            manifest["commands"]["cycles_cpu"]["command"] = [
                "different-renderer"
            ]
            self.assertFalse(can_resume())

    def test_resume_export_rejects_changed_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = pathlib.Path(directory)
            scene = bundle / "scene.json"
            geometry = bundle / "geometry.bin"
            scene.write_text("{}", encoding="utf-8")
            geometry.write_bytes(b"geometry")
            command = ["blender", "export.py", str(bundle)]
            manifest = {
                "commands": {
                    "export": {
                        "command": command,
                        "returncode": 0,
                    }
                },
                "scene": {
                    "scene_json_sha256": self.runner._sha256(scene),
                    "geometry_bin_sha256": self.runner._sha256(geometry),
                },
            }
            self.assertTrue(
                self.runner._can_resume_export(
                    manifest, command, bundle
                )
            )
            geometry.write_bytes(b"changed geometry")
            self.assertFalse(
                self.runner._can_resume_export(
                    manifest, command, bundle
                )
            )


if __name__ == "__main__":
    runner_path = sys.argv[1:]
    sys.argv = [sys.argv[0]]
    if len(runner_path) != 1:
        raise SystemExit(
            "usage: test_scene_benchmark_runner.py "
            "tools/run_scene_benchmark.py"
        )
    sys.argv.append(runner_path[0])
    unittest.main(argv=[sys.argv[0]])
