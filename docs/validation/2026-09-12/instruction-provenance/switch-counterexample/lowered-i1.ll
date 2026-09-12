target datalayout = "e-p:64:64-p4:64:64-i64:64-n32:64"
@luisa.switch.table.0 = private addrspace(4) constant [2 x i32] [i32 20, i32 10]
define i32 @dispatch(i1 %x) alwaysinline {
entry:
  %switch.table.inrange = icmp ult i1 %x, 1
  %switch.table.index = select i1 %switch.table.inrange, i1 %x, i1 1
  %switch.table.ptr = getelementptr inbounds [2 x i32], ptr addrspace(4) @luisa.switch.table.0, i1 0, i1 %switch.table.index
  %switch.table.value = load i32, ptr addrspace(4) %switch.table.ptr
  br label %merge
default:
  br label %merge
zero:
  br label %merge
merge:
  ret i32 %switch.table.value
}
define i32 @default_case() {
entry:
  %result = call i32 @dispatch(i1 true)
  ret i32 %result
}
