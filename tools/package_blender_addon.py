#!/usr/bin/env python3
"""Stage and package the Psycles Blender extension (.zip) with native libraries."""

from __future__ import annotations

import argparse
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ADDON_SOURCE = _REPO_ROOT / "blender_addon" / "psycles"

_EXCLUDE_DIR_NAMES = {"__pycache__", ".git", ".cache", ".data", "tests"}
_EXCLUDE_FILE_SUFFIXES = {".pyc", ".pyo", ".pdb", ".lib", ".exp", ".ilk", ".exe", ".a"}
_BLENDER_PROVIDED_PREFIXES = ("openimageio", "embree", "tbb")


def detect_platform_tag() -> str:
    machine = platform.machine().lower()
    if sys.platform == "win32":
        return "windows-arm64" if "arm" in machine or "aarch64" in machine else "windows-x64"
    if sys.platform == "darwin":
        return "macos-arm64" if "arm" in machine or "aarch64" in machine else "macos-x64"
    return "linux-arm64" if "arm" in machine or "aarch64" in machine else "linux-x64"


def _is_runtime_payload(path: pathlib.Path) -> bool:
    if not path.is_file():
        return False
    lower_name = path.name.lower()
    if path.suffix.lower() in _EXCLUDE_FILE_SUFFIXES:
        return False
    if any(lower_name.startswith(prefix) for prefix in _BLENDER_PROVIDED_PREFIXES):
        return False
    if sys.platform == "win32":
        return path.suffix.lower() in {".dll", ".bc", ".hipfb"}
    if sys.platform == "darwin":
        return path.suffix.lower() in {".dylib", ".metallib"}
    return ".so" in path.suffixes or path.suffix.lower() in {".bc", ".hipfb"}


def _copy_addon_tree(src: pathlib.Path, dst: pathlib.Path) -> None:
    def _ignore(_directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            if name in _EXCLUDE_DIR_NAMES or name == ".DS_Store":
                ignored.add(name)
            elif name.endswith((".pyc", ".pyo")):
                ignored.add(name)
        return ignored

    shutil.copytree(src, dst, ignore=_ignore, dirs_exist_ok=True)


def _copy_runtime_binaries(build_dir: pathlib.Path, dst_bin: pathlib.Path) -> int:
    dst_bin.mkdir(parents=True, exist_ok=True)
    search_dirs = [build_dir / "bin", build_dir]
    copied: set[str] = set()
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            if entry.name in copied or not _is_runtime_payload(entry):
                continue
            shutil.copy2(entry, dst_bin / entry.name)
            copied.add(entry.name)
    required = (
        "psycles_blender.dll"
        if sys.platform == "win32"
        else ("libpsycles_blender.dylib" if sys.platform == "darwin" else "libpsycles_blender.so")
    )
    if required not in copied:
        raise FileNotFoundError(
            f"Required bridge library {required!r} not found under {build_dir}"
        )
    return len(copied)


def _inject_manifest_platform(manifest_path: pathlib.Path, platform_tag: str) -> None:
    text = manifest_path.read_text(encoding="utf-8")
    if "platforms =" in text:
        return
    lines = text.splitlines()
    insert_at = 0
    for idx, line in enumerate(lines):
        if line.startswith("blender_version_min"):
            insert_at = idx + 1
            break
    if insert_at == 0:
        insert_at = len(lines)
    lines.insert(insert_at, f'platforms = ["{platform_tag}"]')
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_zip_fallback(source_dir: pathlib.Path, output_zip: pathlib.Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(source_dir).as_posix())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=pathlib.Path,
        required=True,
        help="CMake build directory containing bin/psycles_blender and Luisa backends",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output .zip path (defaults to psycles-<platform>.zip in cwd)",
    )
    parser.add_argument(
        "--blender",
        type=pathlib.Path,
        default=None,
        help="Optional Blender executable to validate/build (and --install) the extension",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install the built .zip into Blender's user_default repository and enable it",
    )
    parser.add_argument(
        "--platform",
        default=detect_platform_tag(),
        help="Blender extension platform identifier (default: host platform)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.install and args.blender is None:
        raise SystemExit("--install requires --blender")

    build_dir = args.build_dir.resolve()
    output_zip = (
        args.output.resolve()
        if args.output is not None
        else pathlib.Path.cwd() / f"psycles-{args.platform}.zip"
    )
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="psycles_ext_stage_") as tmp_dir:
        staging = pathlib.Path(tmp_dir) / "psycles"
        _copy_addon_tree(_ADDON_SOURCE, staging)
        binary_count = _copy_runtime_binaries(build_dir, staging / "bin")
        _inject_manifest_platform(staging / "blender_manifest.toml", args.platform)
        print(f"Staged extension at {staging} ({binary_count} runtime binaries)")

        if args.blender is not None:
            blender = str(args.blender.resolve())
            subprocess.run(
                [blender, "--command", "extension", "validate", str(staging)],
                check=True,
            )
            subprocess.run(
                [
                    blender,
                    "--command",
                    "extension",
                    "build",
                    "--source-dir",
                    str(staging),
                    "--output-filepath",
                    str(output_zip),
                ],
                check=True,
            )
            if args.install:
                subprocess.run(
                    [
                        blender,
                        "--command",
                        "extension",
                        "install-file",
                        "-r",
                        "user_default",
                        "-e",
                        str(output_zip),
                    ],
                    check=True,
                )
        else:
            _build_zip_fallback(staging, output_zip)

    size_mb = output_zip.stat().st_size / (1024.0 * 1024.0)
    print(f"Built extension archive: {output_zip} ({size_mb:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
