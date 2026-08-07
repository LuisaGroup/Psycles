#!/usr/bin/env python3
"""Configure and build Psycles on Windows with the CUDA, DirectX, and Vulkan
LuisaCompute backends only.

The LLVM/Embree ``fallback`` backend and every other LuisaCompute backend
(Metal, HIP, CPU, Remote, Rust) is explicitly disabled, so this script needs
no LLVM/Embree installation and never falls back to a reduced build.

No machine-specific file path is hard-coded.  Tool and SDK locations are
discovered at runtime:

* ``cmake`` is taken from ``PATH``;
* the MSVC toolchain is located through ``vswhere`` (or ``PATH``) and the
  ``Visual Studio`` generator is used when found, otherwise ``Ninja``;
* the CUDA toolkit is taken from ``CUDA_PATH``/``CUDA_HOME`` or the location
  of ``nvcc`` on ``PATH``;
* the Vulkan SDK is taken from the ``VULKAN_SDK`` environment variable.

The build directory, configuration, and job count are command-line options
with the same defaults the repository documentation uses.

Usage:
    python scripts/build_windows.py [options]

Options:
    --build-dir DIR     Build tree root (default: <repo>/build).
    --config NAME       Build configuration (default: Release).
    --parallel N        Parallel jobs (default: number of CPUs).
    --generator NAME    "auto" (default), "Visual Studio 17 2022", or "Ninja".
    --platform NAME     Target platform for the Visual Studio generator
                        (default: x64; ignored by single-config generators).
    --clean             Delete the build directory before configuring.
    --openimageio       Enable OpenImageIO/OpenEXR output (default; requires
                        an OpenImageIO package discoverable by CMake).
    --no-openimageio    Disable the OpenImageIO dependency.
    --tests             Build the C++ tests (default).
    --no-tests          Do not build the C++ tests.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# LuisaCompute backends that Psycles must build on Windows.  The value is
# passed to both the Psycles-level option (when one exists) and the
# LuisaCompute-level option so the requested set cannot be silently reduced.
_ENABLED_BACKENDS = {
    "cuda": {
        "psycles": None,  # no top-level Psycles toggle; Luisa default is ON
        "luisa": "LUISA_COMPUTE_ENABLE_CUDA",
    },
    "dx": {
        "psycles": None,
        "luisa": "LUISA_COMPUTE_ENABLE_DX",
    },
    "vk": {
        "psycles": "PSYCLES_ENABLE_LUISA_VULKAN",
        "luisa": "LUISA_COMPUTE_ENABLE_VULKAN",
    },
}

# Every other backend is disabled on both levels.
_DISABLED_BACKENDS = {
    "fallback": ("PSYCLES_ENABLE_LUISA_FALLBACK", "LUISA_COMPUTE_ENABLE_FALLBACK"),
    "metal": ("PSYCLES_ENABLE_LUISA_METAL", "LUISA_COMPUTE_ENABLE_METAL"),
    "hip": ("PSYCLES_ENABLE_LUISA_HIP", "LUISA_COMPUTE_ENABLE_HIP"),
    "cpu": (None, "LUISA_COMPUTE_ENABLE_CPU"),
    "remote": (None, "LUISA_COMPUTE_ENABLE_REMOTE"),
    "rust": (None, "LUISA_COMPUTE_ENABLE_RUST"),
}

_BACKEND_LIBRARY_NAMES = {
    "cuda": "luisa-backend-cuda",
    "dx": "luisa-backend-dx",
    "vk": "luisa-backend-vk",
    "fallback": "luisa-backend-fallback",
    "hip": "luisa-backend-hip",
    "metal": "luisa-backend-metal",
    "cpu": "luisa-backend-cpu",
    "remote": "luisa-backend-remote",
}


def log(message: str) -> None:
    print(f"[build_windows] {message}", flush=True)


def run(command: list[str], cwd: Path | None = None) -> int:
    """Run a command, streaming its output, and return its exit code."""
    log("+ " + " ".join(command))
    return subprocess.call(command, cwd=str(cwd) if cwd else None)


def require_tool(name: str) -> Path:
    """Return the absolute path of a tool found on PATH, or exit."""
    found = shutil.which(name)
    if not found:
        log(f"error: required tool '{name}' was not found on PATH")
        sys.exit(1)
    return Path(found).resolve()


def find_vswhere() -> Path | None:
    """Locate vswhere: on PATH first, then in the standard VS Installer dir."""
    on_path = shutil.which("vswhere")
    if on_path:
        return Path(on_path).resolve()
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    if program_files_x86:
        candidate = (
            Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        )
        if candidate.is_file():
            return candidate
    return None


def find_visual_studio_generator() -> tuple[str, Path] | None:
    """Return ("Visual Studio 17 2022", install_dir) when an MSVC-capable
    Visual Studio 2022 installation exists, else None."""
    vswhere = find_vswhere()
    if not vswhere:
        return None
    command = [
        str(vswhere),
        "-latest",
        "-products",
        "*",
        "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "-property",
        "installationPath",
    ]
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, encoding="utf-8"
        )
    except OSError:
        return None
    install_dir = result.stdout.strip()
    if not install_dir or result.returncode != 0:
        return None
    return "Visual Studio 17 2022", Path(install_dir)


def find_cuda_root() -> Path | None:
    """Locate the CUDA toolkit root from the environment or nvcc on PATH."""
    for variable in ("CUDA_PATH", "CUDA_HOME"):
        value = os.environ.get(variable)
        if value and Path(value).is_dir():
            return Path(value)
    nvcc = shutil.which("nvcc")
    if nvcc:
        root = Path(nvcc).resolve().parents[1]  # <root>/bin/nvcc.exe
        if root.is_dir():
            return root
    return None


def generator_supported(name: str) -> bool:
    """Whether the discovered cmake supports the named generator.

    ``cmake --help`` marks the generators available on this platform with a
    leading ``*``, so both plain and starred entries are matched.
    """
    cmake = require_tool("cmake")
    result = subprocess.run(
        [str(cmake), "--help"],
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    for line in result.stdout.splitlines():
        cleaned = line.strip().lstrip("*").strip()
        if "=" in cleaned and cleaned.split("=", 1)[0].strip() == name:
            return True
    return False


def resolve_generator(choice: str, platform: str) -> tuple[list[str], str]:
    """Return the cmake generator arguments and a short label.

    auto prefers an MSVC-capable Visual Studio installation (whose compiler
    CMake configures itself), falling back to Ninja.
    """
    if choice != "auto":
        if not generator_supported(choice):
            log(f"error: generator '{choice}' is not supported by this cmake")
            sys.exit(1)
        args = ["-G", choice]
        if choice.startswith("Visual Studio"):
            args += ["-A", platform]
        return args, choice
    visual = find_visual_studio_generator()
    if visual and generator_supported(visual[0]):
        log(f"using Visual Studio installation at {visual[1]}")
        return ["-G", visual[0], "-A", platform], visual[0]
    if generator_supported("Ninja") and shutil.which("ninja"):
        return ["-G", "Ninja"], "Ninja"
    log("error: no MSVC-capable Visual Studio or Ninja generator found; "
        "install the Visual Studio C++ workload or Ninja")
    sys.exit(1)


def cache_value(build_dir: Path, key: str) -> str | None:
    """Read a cached CMake variable from build_dir/CMakeCache.txt."""
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return None
    prefix = key + ":"
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix) and "=" in line:
            return line.split("=", 1)[1].strip()
    return None


def check_cached_flags(build_dir: Path) -> bool:
    """Verify the requested backend policy is exactly what CMake cached."""
    ok = True
    for name, entry in _ENABLED_BACKENDS.items():
        value = cache_value(build_dir, entry["luisa"])
        if value != "ON":
            log(f"error: backend '{name}' is not enabled (cache: {value!r})")
            ok = False
    for name, (psycles_opt, luisa_opt) in _DISABLED_BACKENDS.items():
        for option in (psycles_opt, luisa_opt):
            if option is None:
                continue
            value = cache_value(build_dir, option)
            if value != "OFF":
                log(f"error: backend '{name}' is not disabled (cache: {value!r})")
                ok = False
    return ok


def verify_artifacts(build_dir: Path) -> bool:
    """Check that the enabled backend libraries were produced and the
    disabled ones were not."""
    ok = True
    windows = os.name == "nt"
    for name, library in _BACKEND_LIBRARY_NAMES.items():
        candidates = []
        for pattern in (library + ".dll", library + ".lib", library + ".so", library + ".dylib"):
            candidates.extend(str(p) for p in build_dir.rglob(pattern))
        present = bool(candidates)
        if name in _ENABLED_BACKENDS:
            if not present:
                log(f"error: expected backend library '{library}' was not produced")
                ok = False
            else:
                log(f"ok: {library} -> {candidates[0]}")
        elif present:
            log(f"error: disabled backend library '{library}' was produced; "
                "the backend policy was not honored")
            ok = False
        elif windows:
            log(f"ok: disabled backend '{library}' is absent")
    return ok


def build_arguments(args: argparse.Namespace) -> list[str]:
    generator_args, generator_label = resolve_generator(args.generator, args.platform)
    arguments = [
        str(require_tool("cmake")),
        "-S",
        str(REPO_ROOT),
        "-B",
        str(args.build_dir),
    ]
    arguments += generator_args
    arguments += [f"-D{option}=ON" for entry in _ENABLED_BACKENDS.values()
                  if (option := entry["psycles"])]
    arguments += [f"-D{option}=ON" for entry in _ENABLED_BACKENDS.values()
                  if (option := entry["luisa"])]
    for _, (psycles_opt, luisa_opt) in _DISABLED_BACKENDS.items():
        for option in (psycles_opt, luisa_opt):
            if option is not None:
                arguments.append(f"-D{option}=OFF")
    arguments += [
        "-DPSYCLES_ENABLE_LUISA=ON",
        f"-DPSYCLES_ENABLE_OPENIMAGEIO={'ON' if args.openimageio else 'OFF'}",
        f"-DPSYCLES_BUILD_TESTS={'ON' if args.tests else 'OFF'}",
        "-DLUISA_COMPUTE_ENABLE_DSL=ON",
        "-DLUISA_COMPUTE_ENABLE_GUI=OFF",
    ]
    cuda_root = find_cuda_root()
    if cuda_root:
        log(f"CUDA toolkit: {cuda_root}")
        arguments.append(f"-DCUDAToolkit_ROOT={cuda_root}")
    if generator_label.startswith("Visual Studio"):
        arguments.append(f"-DCMAKE_CONFIGURATION_TYPES={args.config}")
    vulkan_sdk = os.environ.get("VULKAN_SDK")
    if vulkan_sdk:
        log(f"Vulkan SDK: {vulkan_sdk}")
    else:
        log("warning: VULKAN_SDK is not set; the Vulkan backend needs the "
            "Vulkan SDK headers at configure time")
    return arguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--build-dir", type=Path, default=REPO_ROOT / "build",
        help="build tree root (default: <repo>/build)",
    )
    parser.add_argument("--config", default="Release", help="build configuration")
    parser.add_argument(
        "--parallel", type=int, default=os.cpu_count() or 1,
        help="parallel jobs (default: CPU count)",
    )
    parser.add_argument(
        "--generator", default="auto",
        choices=["auto", "Visual Studio 17 2022", "Ninja"],
    )
    parser.add_argument("--platform", default="x64", help="Visual Studio platform")
    parser.add_argument(
        "--clean", action="store_true",
        help="delete the build directory before configuring",
    )
    parser.add_argument(
        "--openimageio", dest="openimageio", action="store_true", default=True,
        help="enable OpenImageIO EXR output (default)",
    )
    parser.add_argument(
        "--no-openimageio", dest="openimageio", action="store_false",
        help="disable the OpenImageIO dependency",
    )
    parser.add_argument(
        "--tests", dest="tests", action="store_true", default=True,
        help="build the C++ tests (default)",
    )
    parser.add_argument(
        "--no-tests", dest="tests", action="store_false",
        help="do not build the C++ tests",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not REPO_ROOT.is_dir() or not (REPO_ROOT / "CMakeLists.txt").is_file():
        log(f"error: {REPO_ROOT} is not a Psycles source tree")
        return 1
    cmake = require_tool("cmake")
    if args.clean and args.build_dir.exists():
        log(f"removing build directory {args.build_dir}")
        shutil.rmtree(args.build_dir)
    args.build_dir.mkdir(parents=True, exist_ok=True)

    log(f"configuring Psycles (backends: {', '.join(sorted(_ENABLED_BACKENDS))})")
    configure = build_arguments(args)
    if run(configure, cwd=REPO_ROOT) != 0:
        log("error: cmake configure failed")
        return 1
    if not check_cached_flags(args.build_dir):
        log("error: configured backend set does not match the required policy")
        return 1

    log(f"building {args.config} with {args.parallel} parallel jobs")
    build = [
        str(cmake),
        "--build",
        str(args.build_dir),
        "--config",
        args.config,
        "--parallel",
        str(args.parallel),
    ]
    if run(build, cwd=REPO_ROOT) != 0:
        log("error: cmake build failed")
        return 1

    if not verify_artifacts(args.build_dir):
        log("error: post-build backend verification failed")
        return 1
    log("build succeeded: CUDA, DirectX, and Vulkan backends only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
