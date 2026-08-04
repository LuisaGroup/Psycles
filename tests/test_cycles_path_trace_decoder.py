"""Regression for Blender 5.3 multipart path-trace EXR decoding."""

from __future__ import annotations

import json
import pathlib
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import numpy
    import OpenImageIO as oiio
except ModuleNotFoundError:
    numpy = None
    oiio = None


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import cycles_path_trace_schema as schema


class CyclesPathTraceRenderTests(unittest.TestCase):
    @staticmethod
    def _renderer():
        with mock.patch.dict(
            sys.modules,
            {"bpy": types.ModuleType("bpy")},
        ):
            import render_cycles_path_trace as renderer
        return renderer

    def test_single_pixel_border_survives_blender_float32_storage(self) -> None:
        renderer = self._renderer()

        def float32(value: float) -> float:
            return struct.unpack("f", struct.pack("f", value))[0]

        # 480/1152 include the exact Barbershop trace crop that previously
        # expanded to two rows/columns. The complete small/production extents
        # also cover both image edges and every interior pixel.
        for extent in (1, 2, 3, 7, 480, 1152, 2160, 8192):
            for pixel in range(extent):
                lower, upper = renderer._single_pixel_border(pixel, extent)
                stored_lower = float32(lower)
                stored_upper = float32(upper)
                self.assertEqual(int(stored_lower * extent), pixel)
                self.assertEqual(int(stored_upper * extent), pixel + 1)

    def test_absolute_sample_preserves_the_complete_sequence(self) -> None:
        renderer = self._renderer()
        cycles = types.SimpleNamespace(
            samples=1,
            use_sample_subset=False,
            sample_offset=0,
            sample_subset_length=2048,
        )

        renderer._configure_absolute_sample(cycles, 128, 6)

        self.assertEqual(cycles.samples, 128)
        self.assertTrue(cycles.use_sample_subset)
        self.assertEqual(cycles.sample_offset, 6)
        self.assertEqual(cycles.sample_subset_length, 1)

    def test_absolute_sample_must_belong_to_the_complete_sequence(self) -> None:
        renderer = self._renderer()
        cycles = types.SimpleNamespace()

        with self.assertRaisesRegex(ValueError, r"outside \[0, 128\)"):
            renderer._configure_absolute_sample(cycles, 128, 128)

        self.assertEqual(vars(cycles), {})


class PsyclesRawPathTraceDecoderTests(unittest.TestCase):
    def test_raw_rgba_slots_use_the_shared_schema(self) -> None:
        import decode_cycles_path_trace as decoder

        slots = [[0.0, 0.0, 0.0, 0.0] for _ in schema.SLOTS]
        slots[0] = [
            float(schema.SCHEMA_VERSION),
            17.0,
            23.0,
            1.0,
        ]
        slots[1] = [0.0, 0x4321, 0x8765, 1.0]
        slots[2] = [0.5, 0.5, 0.0, 1.0]
        slots[3] = [0.125, 0.25, 0.75, 1.0]

        with tempfile.TemporaryDirectory(
            prefix="psycles-raw-path-trace-decoder-"
        ) as temporary:
            path = pathlib.Path(temporary) / "trace.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "psycles.cycles-path-trace-raw",
                        "trace_schema_version": schema.SCHEMA_VERSION,
                        "pixel_x": 17,
                        "pixel_y": 23,
                        "sample": 0,
                        "slots": slots,
                    }
                ),
                encoding="utf-8",
            )
            trace = decoder.decode_raw_trace(path)

        self.assertEqual(
            trace["global"]["rng"]["rng_pixel"],
            0x87654321,
        )
        self.assertEqual(
            trace["global"]["lens_time"]["time"],
            0.125,
        )
        self.assertEqual(
            trace["global"]["lens_time"]["lens_u"],
            0.25,
        )
        self.assertEqual(
            trace["global"]["lens_time"]["lens_v"],
            0.75,
        )
        self.assertFalse(trace["events"][0]["written"])


@unittest.skipIf(
    oiio is None or numpy is None,
    "OpenImageIO and NumPy are validation dependencies",
)
class CyclesPathTraceDecoderTests(unittest.TestCase):
    def test_each_aov_may_be_a_separate_exr_part(self) -> None:
        import decode_cycles_path_trace as decoder

        with tempfile.TemporaryDirectory(
            prefix="psycles-path-trace-decoder-"
        ) as temporary:
            path = pathlib.Path(temporary) / "multipart.exr"
            specs = []
            values = []
            for slot in schema.SLOTS:
                spec = oiio.ImageSpec(1, 1, 4, oiio.FLOAT)
                spec.channelnames = [
                    f"ViewLayer.{slot.aov}.{component}"
                    for component in "RGBA"
                ]
                spec.attribute("name", f"ViewLayer.{slot.aov}")
                specs.append(spec)
                value = numpy.array(
                    [
                        [
                            [
                                slot.index + 0.125,
                                slot.index + 0.25,
                                slot.index + 0.5,
                                1.0,
                            ]
                        ]
                    ],
                    dtype=numpy.float32,
                )
                values.append(value)
            values[0][0, 0] = (
                schema.SCHEMA_VERSION,
                17.0,
                23.0,
                1.0,
            )

            output = oiio.ImageOutput.create(str(path))
            self.assertIsNotNone(output)
            self.assertTrue(output.open(str(path), specs))
            for index, (spec, value) in enumerate(zip(specs, values)):
                if index:
                    self.assertTrue(
                        output.open(
                            str(path),
                            spec,
                            "AppendSubimage",
                        )
                    )
                self.assertTrue(output.write_image(value))
            self.assertTrue(output.close())

            trace = decoder.decode_trace(path)
            self.assertEqual(
                trace["global"]["header"]["schema_version"],
                schema.SCHEMA_VERSION,
            )
            self.assertEqual(trace["global"]["header"]["pixel_x"], 17.0)
            self.assertEqual(trace["global"]["header"]["pixel_y"], 23.0)
            self.assertTrue(trace["events"][3]["written"])
            last_closure = trace["events"][3]["closures"][-1]
            self.assertEqual(last_closure["index"], 7)
            self.assertEqual(
                last_closure["normal"]["z"],
                schema.AOV_COUNT - 0.5,
            )


if __name__ == "__main__":
    unittest.main()
