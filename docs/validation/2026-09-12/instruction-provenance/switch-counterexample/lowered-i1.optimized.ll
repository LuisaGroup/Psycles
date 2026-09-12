; ModuleID = '/var/tmp/psycles-switch-retraction-20260912/lowered-i1.ll'
source_filename = "/var/tmp/psycles-switch-retraction-20260912/lowered-i1.ll"
target datalayout = "e-p:64:64-p4:64:64-i64:64-n32:64"

@luisa.switch.table.0 = private addrspace(4) constant [2 x i32] [i32 20, i32 10]

; Function Attrs: alwaysinline
define i32 @dispatch(i1 %x) #0 {
entry:
  %0 = sext i1 %x to i64
  %switch.table.ptr = getelementptr inbounds i32, ptr addrspace(4) @luisa.switch.table.0, i64 %0
  %switch.table.value = load i32, ptr addrspace(4) %switch.table.ptr, align 4
  ret i32 %switch.table.value
}

define i32 @default_case() {
entry:
  ret i32 poison
}

attributes #0 = { alwaysinline }
