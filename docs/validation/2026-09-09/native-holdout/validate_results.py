"""Recompute a Holdout archive from its retained evidence, without a shader run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
from capture_results import ROOT, capture, embedded_identities, read, require


def validate(path):
    recorded = read(path)
    require(recorded["schema"] == "psycles.native-holdout.v1", "wrong report schema")
    # Publishing the already-validated checkpoint may advance HEAD. Its
    # recorded base must remain a real ancestor; all source/binary hashes,
    # selections, captures and observations are still rechecked below.
    base = recorded["snapshot"]["root_base"]
    require(subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", base, "HEAD"],
                           check=False).returncode == 0, "recorded root base is not an ancestor")
    actual = capture(Path(recorded["evidence_directory"]), Path(recorded["canaries_directory"]))
    actual["snapshot"]["root_base"] = base
    actual = json.loads(json.dumps(actual, allow_nan=False))
    changed = sorted(key for key in set(recorded) | set(actual) if recorded.get(key) != actual.get(key))
    require(not changed, ("archive differs from retained/current evidence", changed))
    identities = {entry["path"] for entry in embedded_identities(actual)}
    print(f"Validated {len(identities)} identities; exact CTest selections, red/green observations and six canaries. "
          f"Large-image finite gate passed: {actual['canaries']['finite_gate_passed']}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, nargs="?", default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    try:
        validate(args.report)
    except (OSError, ValueError, KeyError, IndexError) as error:
        sys.exit(f"Holdout archive validation failed: {error}")
