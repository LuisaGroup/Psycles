"""Export Blender image resources without evaluating or replacing shader graphs."""

from __future__ import annotations

import array
import hashlib
import pathlib
import shutil
from typing import Any

import bpy


def _safe_name(name: str) -> str:
    clean = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in name
    )
    return clean[:120] or "unnamed"


def _image_extension(image: Any) -> str:
    if image.source == "GENERATED":
        return ".png"
    suffix = pathlib.Path(image.filepath).suffix.lower()
    if suffix in {
        ".jpg",
        ".jpeg",
        ".png",
        ".tga",
        ".bmp",
        ".hdr",
        ".exr",
    }:
        return suffix
    return {
        "JPEG": ".jpg",
        "PNG": ".png",
        "TARGA": ".tga",
        "BMP": ".bmp",
        "HDR": ".hdr",
        "OPEN_EXR": ".exr",
        "OPEN_EXR_MULTILAYER": ".exr",
    }.get(image.file_format, ".bin")


def _external_image_path(image: Any) -> pathlib.Path:
    # Linked image datablocks retain paths relative to the library .blend
    # that owns them, not the currently open main file. This distinction is
    # observable in Blender's official Classroom scene, whose linked assets
    # use paths such as ``//../../textures/_baseTextures/...``. Delegate the
    # base selection to Blender so export and Cycles resolve the same file.
    return pathlib.Path(
        bpy.path.abspath(
            image.filepath,
            library=image.library,
        )
    )


def _save_generated_image(
    image: Any,
    destination: pathlib.Path,
) -> None:
    # GENERATED images have no backing file, but their pixel buffer is a real
    # Cycles texture input. Save a temporary datablock copy through Blender's
    # image codec so the original filepath/format and source scene remain
    # untouched. ``Image.save`` applies the datablock color-space encoding;
    # Psycles decodes that declared space when sampling the exported texture.
    encoded = image.copy()
    try:
        pixels = array.array("f", [0.0]) * len(image.pixels)
        image.pixels.foreach_get(pixels)
        encoded.pixels.foreach_set(pixels)
        encoded.update()
        encoded.filepath_raw = str(destination)
        encoded.file_format = "PNG"
        encoded.save()
    finally:
        bpy.data.images.remove(encoded)


def export_images(output: pathlib.Path) -> list[dict[str, Any]]:
    texture_directory = output / "textures"
    texture_directory.mkdir(parents=True, exist_ok=True)
    result: list[dict[str, Any]] = []
    for index, image in enumerate(
        sorted(bpy.data.images, key=lambda candidate: candidate.name)
    ):
        if image.type in {"RENDER_RESULT", "COMPOSITING"}:
            continue
        metadata = {
            "name": image.name,
            "width": int(image.size[0]),
            "height": int(image.size[1]),
            "source": image.source,
            "colorspace": image.colorspace_settings.name,
            "alpha_mode": image.alpha_mode,
        }
        # ImageManager retains assigned images even when their source cannot
        # load. Dropping the datablock collapses distinct handles to resource
        # zero and loses colorspace/alpha metadata. Do not synthesize pixels:
        # a failed full image returns before CLIP/other wrapping in Cycles.
        if min(image.size) <= 0:
            result.append({**metadata, "load_failed": True})
            continue
        extension = _image_extension(image)
        destination = texture_directory / (
            f"{index:03d}-{_safe_name(image.name)}{extension}"
        )
        if image.packed_file is not None:
            destination.write_bytes(bytes(image.packed_file.data))
        elif image.source == "GENERATED":
            _save_generated_image(image, destination)
        else:
            source = _external_image_path(image)
            if not source.is_file():
                result.append({**metadata, "load_failed": True})
                continue
            shutil.copyfile(source, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        result.append(
            {
                **metadata,
                "path": destination.relative_to(output).as_posix(),
                "sha256": digest,
            }
        )
    return result
