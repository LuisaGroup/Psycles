; ModuleID = '/var/tmp/psycles-switch-retraction-20260912/original-i1.ll'
source_filename = "/var/tmp/psycles-switch-retraction-20260912/original-i1.ll"
target datalayout = "e-p:64:64-p4:64:64-i64:64-n32:64"

; Function Attrs: alwaysinline
define i32 @dispatch(i1 %x) #0 {
entry:
  %cond = icmp eq i1 %x, false
  %. = select i1 %cond, i32 20, i32 10
  ret i32 %.
}

define i32 @default_case() {
entry:
  ret i32 10
}

attributes #0 = { alwaysinline }
