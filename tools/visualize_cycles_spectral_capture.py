"""Build spectral-node oracle triptychs from a backend capture.

Usage:

    python tools/visualize_cycles_spectral_capture.py capture.tsv output-dir

Each output places the external Cycles 5.2.1 values, the captured Psycles
backend values, and a fixed 2^20-amplified absolute difference side by side.
The image is a visual diagnostic; the backend executable performs the numeric
acceptance before writing the capture.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_LAYOUT = {
    "blackbody": (4, 4, "Blackbody"),
    "wavelength": (5, 4, "Wavelength"),
}
_CELL = 96
_GAP = 20
_TITLE = 50
_DIFFERENCE_SCALE = float(1 << 20)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = pathlib.Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    return (
        ImageFont.truetype(str(path), size)
        if path.exists()
        else ImageFont.load_default()
    )


def _display(linear: np.ndarray) -> np.ndarray:
    mapped = np.maximum(linear, 0.0)
    mapped = mapped / (1.0 + mapped)
    return np.where(
        mapped <= 0.0031308,
        mapped * 12.92,
        1.055 * np.power(mapped, 1.0 / 2.4) - 0.055,
    )


def _panel(values: np.ndarray, columns: int, rows: int) -> np.ndarray:
    image = np.zeros((rows * _CELL, columns * _CELL, 3), dtype=np.float32)
    for index, value in enumerate(values):
        row, column = divmod(index, columns)
        image[
            row * _CELL : (row + 1) * _CELL,
            column * _CELL : (column + 1) * _CELL,
        ] = value
    return image


def _write_family(
    family: str,
    records: list[dict[str, str]],
    output_directory: pathlib.Path,
) -> None:
    columns, rows, label = _LAYOUT[family]
    expected = np.asarray(
        [
            [
                float(record["expected_r"]),
                float(record["expected_g"]),
                float(record["expected_b"]),
            ]
            for record in records
        ],
        dtype=np.float32,
    )
    actual = np.asarray(
        [
            [
                float(record["actual_r"]),
                float(record["actual_g"]),
                float(record["actual_b"]),
            ]
            for record in records
        ],
        dtype=np.float32,
    )
    if len(records) != columns * rows:
        raise ValueError(
            f"{family} capture has {len(records)} records, expected "
            f"{columns * rows}"
        )
    difference = np.abs(actual - expected)
    panels = [
        _panel(_display(expected), columns, rows),
        _panel(_display(actual), columns, rows),
        _panel(_display(difference * _DIFFERENCE_SCALE), columns, rows),
    ]
    panel_width = columns * _CELL
    panel_height = rows * _CELL
    canvas = Image.new(
        "RGB",
        (panel_width * 3 + _GAP * 2, panel_height + _TITLE),
        (24, 24, 24),
    )
    for index, panel in enumerate(panels):
        encoded = np.ascontiguousarray(
            np.clip(panel * 255.0 + 0.5, 0.0, 255.0), dtype=np.uint8
        )
        canvas.paste(
            Image.fromarray(encoded, "RGB"),
            (index * (panel_width + _GAP), _TITLE),
        )
    draw = ImageDraw.Draw(canvas)
    font = _font(20)
    headings = (
        f"Cycles 5.2.1 {label}",
        f"Psycles {records[0]['backend'].upper()} SVM",
        "Absolute difference x 2^20",
    )
    for index, heading in enumerate(headings):
        x = index * (panel_width + _GAP) + 8
        draw.text((x, 13), heading, fill=(245, 245, 245), font=font)
    for column in range(1, 3):
        x = column * panel_width + (column - 1) * _GAP + _GAP // 2
        draw.line((x, 0, x, canvas.height), fill=(90, 90, 90), width=1)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{family}-oracle-triptych.png"
    canvas.save(output_path)
    maximum = float(np.max(difference))
    print(f"{family}: max_abs={maximum:.9g} output={output_path}")


def _main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: visualize_cycles_spectral_capture.py capture.tsv output-dir"
        )
    capture_path = pathlib.Path(sys.argv[1])
    output_directory = pathlib.Path(sys.argv[2])
    with capture_path.open(newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    for family in _LAYOUT:
        records = [row for row in rows if row["family"] == family]
        records.sort(key=lambda row: int(row["index"]))
        _write_family(family, records, output_directory)


if __name__ == "__main__":
    _main()
