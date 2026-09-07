set pagination off
set confirm off
set debuginfod enabled off
set breakpoint pending on
set print thread-events off
handle SIGPIPE nostop noprint pass
python
import gdb
observed_return = False

class CountReturn(gdb.Breakpoint):
    def __init__(self, label):
        address = int(gdb.parse_and_eval("*(unsigned long long *)$rsp"))
        super().__init__("*%#x" % address, internal=True, temporary=True)
        self.thread = gdb.selected_thread().global_num
        self.label = label

    def stop(self):
        # x86-64 System V ABI: observe the original function's integer result
        # after it returns. This writes no inferior data or renderer settings.
        value = int(gdb.parse_and_eval("$eax"))
        gdb.write("%s %d\n" % (self.label, value))
        global observed_return
        observed_return = True
        # Let GDB retire its temporary breakpoint at a real stop before the
        # driver continues. Returning False keeps it live in current GDB.
        return True

class CountEntry(gdb.Breakpoint):
    def __init__(self, function, label):
        super().__init__("*'%s'" % function, internal=True)
        self.label = label

    def stop(self):
        CountReturn(self.label)
        return False

CountEntry("ccl::Scene::get_max_closure_count()", "CYCLES_MAX_CLOSURES")
CountEntry("ccl::ShaderGraph::get_num_closures()", "CYCLES_GRAPH_CLOSURES")
gdb.execute("run")
while gdb.selected_inferior().pid:
    if not observed_return:
        raise RuntimeError("unexpected inferior stop; oracle did not complete")
    observed_return = False
    gdb.execute("continue")
gdb.execute("quit %d" % int(gdb.parse_and_eval("$_exitcode")))
end
