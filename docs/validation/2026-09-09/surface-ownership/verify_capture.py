"""Verify a frozen IR observation; do not execute or emulate shader semantics."""
import hashlib
from pathlib import Path
import re

EVIDENCE = Path("/var/tmp/psycles-surface-ownership-FeArPk")
HASHES = {
    "hip_kernel_final_12.ll": "8122e2dfafc71e9683fcc08c376f47383350fb32d8550adf5e68f6d5488db7ed",
    "barbershop-before.log": "437c87ee721c817f1676100bd13d07c66ad42507950025e65efd26612417b1a6",
    "barbershop-before.exr": "e6cbf31b892202ee8c516a7dc2c8feeb3bc2bb0a72979e88eae6b28e18e510f9",
}
for name, expected in HASHES.items():
    with (EVIDENCE / name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, name

module = (EVIDENCE / "hip_kernel_final_12.ll").read_text()
entry = "kernel_11eae98514a796c2"
match = re.search(r"^define amdgpu_kernel void @" + entry + r"\(.*?^}",
                  module, re.M | re.S)
assert match is not None
body = match[0]
assert "alloca [33 x float]" in body and "alloca [60 x [4 x i32]]" in body
value = "%.unpack12967.i"
uses = [line.strip() for line in body.splitlines() if value in line]
assert uses == [
    "%.unpack12967.i = load i32, ptr addrspace(1) %.elt12966.i, align 16",
    "%_reg_159.i = phi i32 [ %.unpack12967.i, %721 ], [ 1, %1810 ]",
    "%2102 = and i32 %.unpack12967.i, 524288",
    "%2185 = or i32 %2083, %.unpack12967.i",
    "%2186 = and i32 %.unpack12967.i, 20",
]
for operation in (
    "%722 = and i32 %_reg_158.i, 541065215",
    "%723 = extractvalue { ptr addrspace(1), i64 } %12, 0",
    "%724 = zext nneg i32 %722 to i64",
    "%725 = shl nuw nsw i64 %724, 5",
    "%726 = getelementptr inbounds nuw i8, ptr addrspace(1) %723, i64 %725",
    "%.elt12966.i = getelementptr inbounds nuw i8, ptr addrspace(1) %726, i64 16",
    "%2187 = icmp eq i32 %2186, 0",
    "call void @llvm.assume(i1 %2187)",
):
    assert operation in body, operation
log = (EVIDENCE / "barbershop-before.log").read_text()
assert "subroutines=6 frame_fields=92 frame_bytes=416 capacity=1048576" in log
assert "Rendered 2048x858 at 256 spp" in log
print("Verified three frozen artifacts and the shared shader-flags load/use witness.")
