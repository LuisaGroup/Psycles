target datalayout = "e-p:64:64-p4:64:64-i64:64-n32:64"
define i32 @dispatch(i1 %x) alwaysinline {
entry:
  switch i1 %x, label %default [ i1 0, label %zero ]
default:
  br label %merge
zero:
  br label %merge
merge:
  %value = phi i32 [10, %default], [20, %zero]
  ret i32 %value
}
define i32 @default_case() {
entry:
  %result = call i32 @dispatch(i1 true)
  ret i32 %result
}
