"""Regression for Blender 5.3 multipart path-trace EXR decoding."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

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
