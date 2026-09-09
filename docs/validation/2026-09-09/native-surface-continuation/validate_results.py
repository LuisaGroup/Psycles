"""Check archived claims and retained evidence; never evaluate a shader."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("report", type=Path, nargs="?", default=Path(__file__).with_name("results.json"))
args = parser.parse_args()
report = json.loads(args.report.read_text())
assert report["schema"] == "psycles.native-surface-continuation.v1"
assert [(r["passed"], r["total"]) for r in report["tests"]] == [(178, 178), (190, 190), (192, 192), (9, 9)]
green = report["regression_green"]
assert set(green) == {"hip", "fallback", "vk"}
for backend in green.values():
    assert backend["state_cases"] == 26
    rows = backend["render_observations"]
    assert len(rows) == len({(r["case"], r["scheduler"]) for r in rows}) == 22
    assert all(r["rgb_lanes"] == 768 and r["mismatches"] == 0 for r in rows)
records = report["canaries"]["records"]
assert len(records) == len({(r["scene"], r["repeat"]) for r in records}) == 6
assert all(len(r["passes"]) == 15 for r in records)
assert report["canaries"]["runner_exit_status"] == (0 if all(r["actual_all_finite"] for r in records) else 2)
assert report["qa"]["renderer_and_sdk_changed_since_S1"]
assert report["qa"]["no_exclusive_renderer_causal_timing_claim"]
assert report["qa"]["shadow_task_aos_bytes"] == 224
assert report["qa"]["shadow_task_soa_growth_bytes_per_path"] == 4
checked = set()


def verify(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            path = Path(value["path"])
            with path.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == value["sha256"], path
            checked.add(str(path))
        for child in value.values():
            verify(child)
    elif isinstance(value, list):
        for child in value:
            verify(child)


verify(report)
print(f"Validated {len(checked)} retained identities, all test observations, and explicit campaign limitations.")
