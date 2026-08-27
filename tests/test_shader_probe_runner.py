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
            pathlib.Path("/probe/cycles.json"),
            pathlib.Path("/probe/export/scene.json"),
        )
        self.assertEqual(command[0], "/venv/bin/python")
        self.assertEqual(
            command[1], "/tools/compare_cycles.py"
        )
        self.assertNotIn("/opt/blender", command)
        self.assertEqual(
            command[command.index("--reference-metadata") + 1],
            "/probe/cycles.json",
        )
        self.assertEqual(
            command[command.index("--actual-metadata") + 1],
            "/probe/export/scene.json",
        )
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

    def test_vulkan_probe_forces_native_xir_without_dxc(self) -> None:
        environment = self.runner._psycles_environment(
            "vk",
            {
                "KEEP_ME": "yes",
                "LUISA_VULKAN_USE_XIR": "0",
                "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV": "0",
                "LUISA_VULKAN_DISABLE_DXC": "0",
            },
        )
        self.assertIsNotNone(environment)
        self.assertEqual(environment["KEEP_ME"], "yes")
        self.assertEqual(environment["LUISA_VULKAN_USE_XIR"], "1")
        self.assertEqual(
            environment["LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV"], "1"
        )
        self.assertEqual(environment["LUISA_VULKAN_DISABLE_DXC"], "1")
        self.assertIsNone(
            self.runner._psycles_environment("hip", {"KEEP_ME": "yes"})
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

    def test_glass_gate_rejects_ndf_instead_of_vndf_sampling(
        self,
    ) -> None:
        aligned = {
            "Combined": (1.000001075, 0.000089489),
            "GlossCol": (0.999999642, 0.000028780),
            "GlossDir": (1.000001068, 0.000059442),
            "TransCol": (0.999999191, 0.000051432),
            "TransDir": (1.000001355, 0.000082308),
        }
        report = {
            "passes": {
                pass_name: {
                    "luminance_mean_ratio": ratio,
                    "relative_rmse": rmse,
                }
                for pass_name, (ratio, rmse) in aligned.items()
            }
        }
        self.assertEqual(
            self.runner._probe_gate_failures("glass_transport", report),
            [],
        )

        # The former Beckmann NDF sampler had plausible energy but a different
        # Cycles sample map. These are its retained 256-spp oracle metrics.
        old_ndf = {
            "Combined": (0.999885660, 0.005619914),
            "GlossCol": (0.999350473, 0.001054857),
            "GlossDir": (1.001174338, 0.002185002),
            "TransCol": (1.000022393, 0.000063067),
            "TransDir": (0.999860558, 0.005114500),
        }
        for pass_name, (ratio, rmse) in old_ndf.items():
            report["passes"][pass_name] = {
                "luminance_mean_ratio": ratio,
                "relative_rmse": rmse,
            }
        failures = self.runner._probe_gate_failures(
            "glass_transport", report
        )
        self.assertEqual(len(failures), 8)

    def test_triangle_solid_angle_gate_requires_sample_alignment(
        self,
    ) -> None:
        report = {
            "passes": {
                "Combined": {
                    "luminance_mean_ratio": 1.0000037,
                    "relative_rmse": 0.0014594,
                },
                "DiffDir": {
                    "luminance_mean_ratio": 1.0000038,
                    "relative_rmse": 0.0014594,
                },
            }
        }
        self.assertEqual(
            self.runner._probe_gate_failures(
                "triangle_light_solid_angle", report
            ),
            [],
        )
        report["passes"]["Combined"]["relative_rmse"] = 0.006
        failures = self.runner._probe_gate_failures(
            "triangle_light_solid_angle", report
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("relative RMSE", failures[0])

    def test_camera_gates_require_exact_cycles_sample_mapping(
        self,
    ) -> None:
        for probe, aligned_ratio, aligned_rmse in (
            ("camera_dof_disk", 0.9999951, 0.0001319),
            ("camera_blackman_harris_filter", 1.0000035, 0.0000728),
        ):
            report = {
                "passes": {
                    "Combined": {
                        "luminance_mean_ratio": aligned_ratio,
                        "relative_rmse": aligned_rmse,
                    }
                }
            }
            self.assertEqual(
                self.runner._probe_gate_failures(probe, report),
                [],
            )
            # The former continuous inverse-CDF approximation still converged
            # to the right energy, but did not reproduce Cycles' finite 1024
            # entry lookup table and therefore missed exact edge coverage.
            report["passes"]["Combined"]["relative_rmse"] = 0.001
            failures = self.runner._probe_gate_failures(
                probe, report
            )
            self.assertEqual(len(failures), 1)
            self.assertIn("relative RMSE", failures[0])

    def test_nishita_transport_gate_rejects_world_energy_drift(
        self,
    ) -> None:
        report = {
            "passes": {
                "Combined": {
                    "luminance_mean_ratio": 0.9999106,
                    "relative_rmse": 0.0001402,
                },
                "DiffDir": {
                    "luminance_mean_ratio": 0.9999054,
                    "relative_rmse": 0.0001528,
                },
            }
        }
        self.assertEqual(
            self.runner._probe_gate_failures(
                "nishita_diffuse_transport", report
            ),
            [],
        )
        report["passes"]["DiffDir"]["luminance_mean_ratio"] = 1.001
        failures = self.runner._probe_gate_failures(
            "nishita_diffuse_transport", report
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("energy ratio", failures[0])

    def test_hosek_transport_gate_tracks_energy_not_sample_map(
        self,
    ) -> None:
        report = {
            "passes": {
                "Combined": {
                    "luminance_mean_ratio": 1.000284,
                    "relative_rmse": 0.042405,
                },
                "DiffDir": {
                    "luminance_mean_ratio": 1.000249,
                    "relative_rmse": 0.034036,
                },
            }
        }
        self.assertEqual(
            self.runner._probe_gate_failures(
                "hosek_wilkie_diffuse_transport", report
            ),
            [],
        )
        report["passes"]["Combined"]["luminance_mean_ratio"] = 0.99
        failures = self.runner._probe_gate_failures(
            "hosek_wilkie_diffuse_transport", report
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("energy ratio", failures[0])

    def test_principled_transmission_gate_rejects_missing_backface(
        self,
    ) -> None:
        report = {
            "passes": {
                pass_name: {
                    "luminance_mean_ratio": 1.0,
                    "relative_rmse": 0.0,
                }
                for pass_name in (
                    "Combined",
                    "DiffCol",
                    "GlossCol",
                    "TransCol",
                    "TransDir",
                    "Normal",
                )
            }
        }
        self.assertEqual(
            self.runner._probe_gate_failures(
                "principled_transmission_surface", report
            ),
            [],
        )

        # The former half-vector sign rejection made the rough inverse-eta
        # cell exactly black while the remaining 15 cells still aligned.
        # Preserve both the energy and spatial evidence of that failure.
        report["passes"]["TransDir"] = {
            "luminance_mean_ratio": 0.999641251,
            "relative_rmse": 0.000362996,
        }
        failures = self.runner._probe_gate_failures(
            "principled_transmission_surface", report
        )
        self.assertEqual(len(failures), 2)
        self.assertTrue(
            any("energy ratio" in failure for failure in failures)
        )
        self.assertTrue(
            any("relative RMSE" in failure for failure in failures)
        )

    def test_principled_thin_wall_gate_rejects_selected_bump_shadowing(
        self,
    ) -> None:
        report = {
            "passes": {
                pass_name: {
                    "luminance_mean_ratio": 1.0,
                    "relative_rmse": 0.0,
                }
                for pass_name in (
                    "Combined",
                    "DiffCol",
                    "GlossCol",
                    "TransCol",
                    "GlossDir",
                    "TransDir",
                    "Normal",
                )
            }
        }
        self.assertEqual(
            self.runner._probe_gate_failures(
                "principled_thin_wall_surface", report
            ),
            [],
        )

        # The former implementation applied post-sample bump shadowing to a
        # selected rough-translucent transmission event. Cycles omits that
        # factor for LABEL_TRANSMIT, while retaining it for ordinary closure
        # evaluation. Preserve the measured whole-matrix structural failure.
        report["passes"]["Combined"] = {
            "luminance_mean_ratio": 0.99902799,
            "relative_rmse": 0.00181607,
        }
        failures = self.runner._probe_gate_failures(
            "principled_thin_wall_surface", report
        )
        self.assertEqual(len(failures), 2)
        self.assertTrue(
            any("energy ratio" in failure for failure in failures)
        )
        self.assertTrue(
            any("relative RMSE" in failure for failure in failures)
        )

    def test_thin_film_gate_rejects_a_structural_cell_error(self) -> None:
        pass_names = (
            "Combined",
            "GlossCol",
            "GlossDir",
            "TransCol",
            "Normal",
        )
        report = {
            "passes": {
                pass_name: {
                    "luminance_mean_ratio": 1.0,
                    # Sparse ray/triangle boundary disagreements may dominate
                    # whole-frame RMSE but leave more than 99% of pixels at
                    # the deterministic direct-evaluation result.
                    "relative_rmse": 0.01,
                    "p99_pixel_rmse": 0.00001,
                    "cycles_rms": 1.0,
                }
                for pass_name in pass_names
            }
        }
        self.assertEqual(
            self.runner._probe_gate_failures("thin_film_surface", report),
            [],
        )

        # Each closure occupies 1/16 of the matrix. An incorrect closure thus
        # changes more than 1% of all pixels and must cross the p99 gate even
        # if positive and negative cell errors preserve global energy.
        report["passes"]["GlossDir"]["p99_pixel_rmse"] = 0.01
        failures = self.runner._probe_gate_failures(
            "thin_film_surface", report
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("normalized p99 pixel RMSE", failures[0])

    def test_metallic_gate_rejects_one_wrong_fresnel_cell(self) -> None:
        pass_names = ("Combined", "GlossCol", "GlossDir", "Normal")
        report = {
            "passes": {
                pass_name: {
                    "luminance_mean_ratio": 1.0,
                    "relative_rmse": 0.01,
                    "p99_pixel_rmse": 0.00001,
                    "cycles_rms": 1.0,
                }
                for pass_name in pass_names
            }
        }
        self.assertEqual(
            self.runner._probe_gate_failures(
                "metallic_bsdf_matrix", report
            ),
            [],
        )

        # One wrong model/distribution cell occupies 1/16 of the image, so
        # cancelling errors cannot evade the order-statistic gate even when
        # the whole-frame mean energy happens to remain exact.
        report["passes"]["GlossDir"]["p99_pixel_rmse"] = 0.01
        failures = self.runner._probe_gate_failures(
            "metallic_bsdf_matrix", report
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("normalized p99 pixel RMSE", failures[0])

    def test_sheen_gate_rejects_one_wrong_distribution_cell(self) -> None:
        pass_names = (
            "Combined",
            "DiffCol",
            "DiffDir",
            "GlossCol",
            "GlossDir",
            "Normal",
        )
        report = {
            "passes": {
                pass_name: {
                    "luminance_mean_ratio": 1.0,
                    "relative_rmse": 0.01,
                    "p99_pixel_rmse": 0.00001,
                    "cycles_rms": 1.0,
                    "actual_invalid_pixels": 0,
                }
                for pass_name in pass_names
            }
        }
        self.assertEqual(
            self.runner._probe_gate_failures("sheen_bsdf_matrix", report),
            [],
        )

        # Each static distribution occupies half the matrix and every case
        # occupies 1/16, so a wrong model or pass classification cannot be
        # hidden by cancellation in the mean-energy ratio.
        report["passes"]["GlossDir"]["p99_pixel_rmse"] = 0.01
        failures = self.runner._probe_gate_failures(
            "sheen_bsdf_matrix", report
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("normalized p99 pixel RMSE", failures[0])

        report["passes"]["GlossDir"]["p99_pixel_rmse"] = 0.00001
        report["passes"]["GlossDir"]["actual_invalid_pixels"] = 1
        failures = self.runner._probe_gate_failures(
            "sheen_bsdf_matrix", report
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("non-finite Psycles pixels", failures[0])


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
