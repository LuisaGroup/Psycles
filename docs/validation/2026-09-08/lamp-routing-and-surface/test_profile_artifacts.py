"""Permanent negative controls for stage-to-dump attribution, not shader evaluation."""
from pathlib import Path
import tempfile
import unittest

from audit_inline_experiment import identify_stage_artifacts
from audit_surface import isa_functions


class ProfileArtifacts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="psycles-stage-artifacts-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.kernel = "kernel_deadbeef"
        self.definition = b"define amdgpu_kernel void @kernel_deadbeef() {\n}\n"
        for phase, index in (("before_opt", 3), ("final", 57)):
            (self.directory / f"hip_kernel_{phase}_{index}.ll").write_bytes(self.definition)
            (self.directory / f"hip_kernel_{phase}_12.ll").write_bytes(
                b"define amdgpu_kernel void @kernel_decoy() {\n}\n")
        # Only tests the symbol-name selector; these are not valid code objects
        # and are never consumed as renderer/compiler oracle inputs.
        (self.directory / "hip_isa_91.co").write_bytes(b"\0kernel_deadbeef\0")
        (self.directory / "hip_isa_12.co").write_bytes(b"\0kernel_decoy\0")

    def test_independent_counters(self):
        result = identify_stage_artifacts(self.directory, self.kernel)
        self.assertEqual({key: path.name for key, path in result.items()}, {
            "before": "hip_kernel_before_opt_3.ll", "final": "hip_kernel_final_57.ll",
            "code_object": "hip_isa_91.co"})

    def test_ambiguous_definition_rejected(self):
        (self.directory / "hip_kernel_final_58.ll").write_bytes(self.definition)
        with self.assertRaises(AssertionError):
            identify_stage_artifacts(self.directory, self.kernel)

    def test_missing_stage_rejected(self):
        with self.assertRaises(AssertionError):
            identify_stage_artifacts(self.directory, "kernel_missing")

    def test_ambiguous_code_object_rejected(self):
        (self.directory / "hip_isa_92.co").write_bytes(b"\0kernel_deadbeef\0")
        with self.assertRaises(AssertionError):
            identify_stage_artifacts(self.directory, self.kernel)

    def test_instruction_counts_exclude_alignment_padding(self):
        path = self.directory / "isa.txt"
        path.write_text("""00000100 <first>:
  s_nop 0 // 00000100: BF800000
  s_endpgm // 00000104: BFB00000
  s_code_end // 00000108: BF9F0000
  s_code_end // 0000010C: BF9F0000
00000110 <second>:
  s_endpgm // 00000110: BFB00000
  s_code_end // 00000114: BF9F0000
""")
        bounds = {"first": {"address": 0x100, "bytes": 8},
                  "second": {"address": 0x110, "bytes": 4}}
        rows = isa_functions(path, bounds)
        self.assertEqual(rows["first"]["static_instructions"], 2)
        self.assertEqual(rows["second"]["static_instructions"], 1)
        with self.assertRaises(AssertionError):
            isa_functions(path, {"first": bounds["first"]})
        with self.assertRaises(AssertionError):
            isa_functions(path, {**bounds, "first": {"address": 0x104, "bytes": 8}})


if __name__ == "__main__":
    unittest.main()
