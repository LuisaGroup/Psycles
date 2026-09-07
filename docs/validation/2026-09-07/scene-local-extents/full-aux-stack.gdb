set pagination off
set confirm off
set debuginfod enabled off
set print thread-events off
handle SIGPIPE nostop noprint pass
python
import gdb
import struct

# Diagnostic A/B only: keep the same binary and compiled scene but restore
# the old conservative 255-slot extent at the auxiliary JIT entrypoints.
# Do not modify the word image, main surface entrypoint, or device execution.
# This script is specific to x86-64 System V and the signature below.
gdb.execute("start")
observations = 0

class FullAuxStack(gdb.Breakpoint):
    def stop(self):
        global observations
        # eval_nodes has eleven integer-class arguments. Argument 11 is
        # stack_size, at entry RSP + 40 after the return address and args 7-10.
        address = int(gdb.parse_and_eval("$rsp")) + 40
        inferior = gdb.selected_inferior()
        extent = struct.unpack("<Q", inferior.read_memory(address, 8))[0]
        if not 1 <= extent <= 255:
            raise RuntimeError("unexpected scene/ABI stack extent: %d" % extent)
        inferior.write_memory(address, struct.pack("<Q", 255))
        observations += 1
        caller = gdb.newest_frame().older().name()
        gdb.write("AUX_STACK_EXTENT %d -> 255 caller=%s\n" % (extent, caller))
        return False

FullAuxStack("*_ZN7psycles13luisa_backend10cycles_svm10eval_nodesERKNS1_13KernelGlobalsEN5luisa7compute4ExprINS6_6BufferIjEEEENS_8compiler10cycles_svm10ShaderTypeEjjRKSt5arrayIbLm110EERKNS1_14TransformStateERNS1_10ShaderDataERKNS1_9PathStateERNS1_16EvaluationResultEm", internal=True)
gdb.execute("continue")
if gdb.selected_inferior().pid or observations < 2:
    raise RuntimeError("expected a successful render and both auxiliary entrypoints")
gdb.write("AUX_STACK_RECORDINGS %d\n" % observations)
gdb.execute("quit %d" % int(gdb.parse_and_eval("$_exitcode")))
end
