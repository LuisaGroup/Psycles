"""Losslessly project native observer JSON into the portable C++ fixture ABI.

The only inserted zeroes are explicitly unused union/input slots. No light
parameter, PDF, direction, attenuation or expected shader value is computed.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("inputs", type=Path, nargs="+")
args = parser.parse_args()
rows = []
for path in args.inputs:
    data = json.loads(path.read_text())
    if data["format"] != "Cycles 5.2 light parameter bits" or data["version"] != 1:
        raise ValueError("Not an original parameter observer capture")
    for row in data["lights"]:
        i, k = row["input"], row["kernel"]
        area = "tan_half_spread" in k
        v = i["axis_x"] + i["axis_y"] + i["axis_z"] + i["position"]
        v += ([i["spread"], 0, i["size_u"], i["size_v"]] if area
              else [i["angle"], i["smooth"], i["radius"], 0])
        v += ([k["tan_half_spread"], k["normalize_spread"], 0, 0, 0] if area
              else [k[name] for name in ("cos_half_spot_angle", "half_cot_half_spot_angle",
                                         "spot_smooth", "cos_half_larger_spread", "ray_segment_dp")])
        v += ([k["invarea"], 0, k["len_u"], k["len_v"]] if area
              else [k["eval_fac"], k["radius"], 0, 0])
        v += k["dir"] + k.get("axis_u", [0] * 3) + k.get("axis_v", [0] * 3)
        assert len(v) == 34 and all(type(x) is int and 0 <= x <= 0xffffffff for x in v)
        prefix = [int(not area), row["name_hex"], int(i["normalize"]),
                  int(i.get("ellipse", False)), int(i.get("is_sphere", False))]
        rows.append(" ".join(map(str, prefix + v)))
args.output.write_text(f"PSYLIGHT1 {len(rows)}\n" + "\n".join(rows) + "\n")
