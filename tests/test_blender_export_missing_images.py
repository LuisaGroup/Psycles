"""Missing image datablocks must retain identity and metadata during export."""

from pathlib import Path
import runpy
import sys
import tempfile

import bpy


def main() -> None:
    exporter = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    create_probe = runpy.run_path(str(exporter.with_name(
        "create_cycles_missing_image_probe.py")))["main"]
    export_images = runpy.run_path(str(exporter))["_export_images"]
    with tempfile.TemporaryDirectory(prefix="psycles-missing-image-") as temporary:
        directory = Path(temporary)
        original_argv = sys.argv
        try:
            sys.argv = ["probe", "--", str(directory / "probe.blend")]
            create_probe()
        finally:
            sys.argv = original_argv
        assert len(bpy.data.images) == 2
        assert all(tuple(image.size) == (0, 0) for image in bpy.data.images)
        # Blender may still have cached pixels after a FILE source disappears;
        # Cycles' external loader nevertheless sees a failed source.
        png = runpy.run_path(str(exporter.parent.parent / "tests" /
                                 "test_blender_export_linked_images.py"))["_TEST_PNG"]
        cached_source = directory / "cached-file.png"
        cached_source.write_bytes(png)
        cached = bpy.data.images.load(str(cached_source))
        assert min(cached.size) > 0
        cached_source.unlink()
        exported = export_images(directory)
        assert len(exported) == 3, "exporter lost failed image identity"
        for item in exported:
            image = bpy.data.images[item["name"]]
            assert item["load_failed"] is True
            assert item["colorspace"] == image.colorspace_settings.name
            assert item["alpha_mode"] == image.alpha_mode
            assert (item["width"], item["height"]) == tuple(image.size)
        assert not list((directory / "textures").iterdir()), "fake image payload was created"
    print("Failed-image export identity and metadata passed")


if __name__ == "__main__":
    main()
