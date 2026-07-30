#!/usr/bin/env python3
"""Enforce Psycles' first-party source-size maintenance budget."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


LINE_LIMIT = 2_000

# Existing debt is allowed to shrink in place, but not grow. Remove an entry
# as soon as its semantic decomposition brings it under LINE_LIMIT.
DEBT_BUDGETS = {
    "tools/create_cycles_shader_probe.py": 5_518,
}

SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".inl",
    ".py",
    ".cmake",
}


def is_source(path: Path) -> bool:
    return path.name == "CMakeLists.txt" or path.suffix.lower() in SOURCE_SUFFIXES


def is_excluded(relative_path: Path) -> bool:
    if not relative_path.parts:
        return True
    top_level = relative_path.parts[0]
    return (
        top_level in {".git", "third_party"}
        or top_level == "build"
        or top_level.startswith("build-")
        or top_level.startswith("cmake-build-")
    )


def line_count(path: Path) -> int:
    return len(path.read_bytes().splitlines())


def check_tree(root: Path) -> tuple[list[str], list[tuple[str, int, int]], int, int]:
    violations: list[str] = []
    remaining_debt: list[tuple[str, int, int]] = []
    files_checked = 0
    lines_checked = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root)
        if is_excluded(relative_path) or not is_source(path):
            continue

        relative = relative_path.as_posix()
        count = line_count(path)
        files_checked += 1
        lines_checked += count

        budget = DEBT_BUDGETS.get(relative)
        if budget is not None:
            if count > budget:
                violations.append(
                    f"{relative}: {count} lines exceeds its debt budget of {budget}"
                )
            elif count > LINE_LIMIT:
                remaining_debt.append((relative, count, budget))
            continue

        if count > LINE_LIMIT:
            violations.append(
                f"{relative}: {count} lines exceeds the {LINE_LIMIT}-line limit"
            )

    return violations, remaining_debt, files_checked, lines_checked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Psycles source root (defaults to the script's parent repository)",
    )
    return parser.parse_args()


def main() -> int:
    root = parse_args().root.resolve()
    if not (root / "CMakeLists.txt").is_file():
        print(f"error: {root} is not a Psycles source root", file=sys.stderr)
        return 2

    violations, remaining_debt, files_checked, lines_checked = check_tree(root)
    print(
        f"checked {files_checked} first-party source files "
        f"({lines_checked} total lines; limit {LINE_LIMIT})"
    )
    for relative, count, budget in remaining_debt:
        print(f"debt: {relative}: {count}/{budget} lines")

    if violations:
        for violation in violations:
            print(f"error: {violation}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
