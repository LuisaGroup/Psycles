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

    def test_canonical_report_includes_volume_passes(self) -> None:
        self.assertIn("Volume Direct", self.runner._REPORT_PASSES)
        self.assertIn("Volume Indirect", self.runner._REPORT_PASSES)

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

    def test_experimental_simd_backend_is_accepted(self) -> None:
        self.assertEqual(
            self.runner._backend_list("metal,simd"),
            ("metal", "simd"),
        )

    def test_scheduler_matrix_is_validated(self) -> None:
        schedulers = (
            "megakernel",
            "megakernel-per-sample",
            "wavefront",
            "wavefront-graph",
            "wavefront-staged",
            "persistent",
        )
        self.assertEqual(
            self.runner._scheduler_list(",".join(schedulers)),
            schedulers,
        )
        with self.assertRaises(
            self.runner.argparse.ArgumentTypeError
        ):
            self.runner._scheduler_list("wavefront,wavefront")
        with self.assertRaises(
            self.runner.argparse.ArgumentTypeError
        ):
            self.runner._scheduler_list("unknown")

    def test_scheduler_matrix_crosses_backends_and_schedulers(self) -> None:
        self.assertEqual(
            self.runner._psycles_run_matrix(
                ("fallback", "metal"),
                ("megakernel", "wavefront-graph"),
            ),
            (
                ("fallback-megakernel", "fallback", "megakernel"),
                (
                    "fallback-wavefront-graph",
                    "fallback",
                    "wavefront-graph",
                ),
                ("metal-megakernel", "metal", "megakernel"),
                (
                    "metal-wavefront-graph",
                    "metal",
                    "wavefront-graph",
                ),
            ),
        )
        self.assertEqual(
            self.runner._psycles_run_matrix(("metal",), ()),
            (("metal", "metal", None),),
        )

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

    def test_psycles_command_selects_scheduler_positionally(self) -> None:
        command = self.runner._psycles_command(
            pathlib.Path("/build/psycles_render_blender_scene"),
            pathlib.Path("/out/export"),
            pathlib.Path("/out/metal-wavefront-graph.ppm"),
            "metal",
            width=640,
            height=480,
            samples=64,
            max_samples_per_dispatch=1,
            scheduler="wavefront-graph",
            wavefront_execution_block_size=64,
        )
        self.assertEqual(
            command[8:],
            [
                "-",
                "320",
                "240",
                "0",
                "0",
                "64",
                "-",
                "1",
                "0",
                "wavefront-graph",
                "64",
                "32768",
                "32",
                "1",
                "1",
                "0",
                "4",
                "2",
                "auto",
                "0",
                "0",
                "0",
                "1",
                "1048576",
            ],
        )
        self.assertEqual(self.runner._wavefront_block_size("32"), 32)
        with self.assertRaises(
            self.runner.argparse.ArgumentTypeError
        ):
            self.runner._wavefront_block_size("48")
        self.assertEqual(self.runner._nonnegative_integer("0"), 0)
        self.assertEqual(self.runner._wavefront_tail_threshold("auto"), "auto")
        self.assertEqual(self.runner._wavefront_tail_threshold("49152"), 49152)
        with self.assertRaises(
            self.runner.argparse.ArgumentTypeError
        ):
            self.runner._nonnegative_integer("-1")
        with self.assertRaises(ValueError):
            self.runner._wavefront_tail_threshold("dynamic")

    def test_psycles_command_tracks_tuned_graph_policy(self) -> None:
        command = self.runner._psycles_command(
            pathlib.Path("/build/psycles_render_blender_scene"),
            pathlib.Path("/out/export"),
            pathlib.Path("/out/metal-wavefront-graph.ppm"),
            "metal",
            width=1920,
            height=1080,
            samples=64,
            max_samples_per_dispatch=1,
            scheduler="wavefront-graph",
            wavefront_counter_readback_batch_size=1,
            wavefront_counter_readback_pipeline_depth=1,
            wavefront_graph_worker_count=131072,
            wavefront_graph_selective=True,
        )
        self.assertEqual(
            command[-8:],
            [
                "1",
                "1",
                "auto",
                "131072",
                "1",
                "0",
                "1",
                "1048576",
            ],
        )

    def test_comparison_labels_each_renderer(self) -> None:
        command = self.runner._comparison_command(
            pathlib.Path("/usr/bin/python"),
            pathlib.Path("/src/tools/compare_cycles.py"),
            pathlib.Path("/out/cycles/hip.exr"),
            pathlib.Path("/out/psycles/fallback.exr"),
            pathlib.Path("/out/report.json"),
            pathlib.Path("/out/triptychs"),
            pathlib.Path("/out/cycles/hip.json"),
            pathlib.Path("/out/export/scene.json"),
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
        self.assertEqual(
            command[command.index("--reference-metadata") + 1],
            "/out/cycles/hip.json",
        )
        self.assertEqual(
            command[command.index("--actual-metadata") + 1],
            "/out/export/scene.json",
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
                    "metal-wavefront-graph": {
                        "render_seconds": 6.0
                    },
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
            result["metal-wavefront-graph"][
                "speedup_over_cycles_metal"
            ],
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
            root = pathlib.Path(directory)
            tools = root / "tools"
            tools.mkdir()
            for name in (
                "export_psycles_scene.py",
                "blender_scene_manifest.py",
                "blender_build_identity.py",
                "cycles_hash.py",
                "exporter_identity.py",
            ):
                (tools / name).write_text(name, encoding="utf-8")
            export_script = tools / "export_psycles_scene.py"
            bundle = root / "bundle"
            bundle.mkdir()
            scene = bundle / "scene.json"
            geometry = bundle / "geometry.bin"
            scene.write_text(
                json.dumps(
                    {
                        "exporter": self.runner.exporter_identity.current(
                            export_script
                        )
                    }
                ),
                encoding="utf-8",
            )
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
                    manifest, command, bundle, export_script
                )
            )
            geometry.write_bytes(b"changed geometry")
            self.assertFalse(
                self.runner._can_resume_export(
                    manifest, command, bundle, export_script
                )
            )

            geometry.write_bytes(b"geometry")
            self.assertTrue(
                self.runner._can_resume_export(
                    manifest, command, bundle, export_script
                )
            )
            export_script.write_text("changed exporter", encoding="utf-8")
            self.assertFalse(
                self.runner._can_resume_export(
                    manifest, command, bundle, export_script
                )
            )

    def test_reused_export_requires_current_exporter_closure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for name in (
                "export_psycles_scene.py",
                "blender_scene_manifest.py",
                "blender_build_identity.py",
                "cycles_hash.py",
                "exporter_identity.py",
            ):
                (root / name).write_text(name, encoding="utf-8")
            export_script = root / "export_psycles_scene.py"
            document = {
                "exporter": self.runner.exporter_identity.current(
                    export_script
                )
            }
            self.runner.exporter_identity.require_current(
                document, root / "scene.json", export_script
            )

            (root / "cycles_hash.py").write_text(
                "changed dependency", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                RuntimeError, "different exporter implementation"
            ):
                self.runner.exporter_identity.require_current(
                    document, root / "scene.json", export_script
                )

            with self.assertRaisesRegex(
                RuntimeError, "no exact exporter identity"
            ):
                self.runner.exporter_identity.require_current(
                    {}, root / "legacy-scene.json", export_script
                )

    def test_reused_export_requires_exact_blender_build(self) -> None:
        identity = {
            "version": "5.2.0 LTS",
            "version_cycle": "release",
            "version_tuple": [5, 2, 0],
            "build_hash": "fbe6228777e7",
            "build_branch": "blender-v5.2-release",
            "build_type": "Release",
        }
        source = pathlib.Path("/out/cycles.json")
        parsed = self.runner._blender_build_identity(
            {"blender_build": identity}, source
        )
        self.assertEqual(parsed, identity)
        self.runner._require_same_blender_build(
            identity,
            parsed,
            reference_source=source,
            candidate_source=pathlib.Path("/out/export/scene.json"),
        )

        mismatched = dict(identity)
        mismatched["build_hash"] = "ec438d7429e5"
        with self.assertRaisesRegex(
            RuntimeError, "different Blender builds"
        ):
            self.runner._require_same_blender_build(
                identity,
                mismatched,
                reference_source=source,
                candidate_source=pathlib.Path(
                    "/out/export/scene.json"
                ),
            )

        with self.assertRaisesRegex(
            RuntimeError, "no exact Blender build identity"
        ):
            self.runner._blender_build_identity({}, source)

        invalid = dict(identity)
        invalid["build_hash"] = 0
        with self.assertRaisesRegex(
            RuntimeError, "invalid Blender build identity fields"
        ):
            self.runner._blender_build_identity(
                {"blender_build": invalid}, source
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
