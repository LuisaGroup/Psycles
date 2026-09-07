"""Check the oracle sampler contract against genuine Blender RNA properties."""

from pathlib import Path
import math
import runpy
import sys
import tempfile

import bpy

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    with tempfile.TemporaryDirectory(prefix="psycles-cycles-sampler-") as directory:
        for script in (
            "render_cycles_golden.py",
            "render_cycles_path_trace.py",
            "create_cycles_area_light_surface_oracle.py",
        ):
            module = runpy.run_path(str(TOOLS / script), run_name="sampler_test")
            scene.cycles.sampling_pattern = "AUTOMATIC"
            scene.cycles.scrambling_distance = 0.25
            scene.cycles.auto_scrambling_distance = True
            if "_configure_sampler" in module:
                module["_configure_sampler"](scene, "TABULATED_SOBOL", 1.0)
            else:
                # Configuration only: no CPU render and no output image.
                module["configure_scene"](
                    Path(directory) / "unused.exr", "RECTANGLE", math.pi, 16
                )
            assert scene.cycles.sampling_pattern == "TABULATED_SOBOL", script
            assert scene.cycles.scrambling_distance == 1.0, script
            assert not scene.cycles.auto_scrambling_distance, script
            print(f"{script}: pinned actual Cycles sampler properties")


if __name__ == "__main__":
    main()
