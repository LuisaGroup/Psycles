# Textual LLVM instruction inventory. Only instruction opcodes at the start
# of an instruction line count: multiline switch operands, labels, declarations,
# attributes and linked-library tables do not. Print module totals separately
# from each complete amdgpu_kernel body so dead OCML imports are not mistaken
# for renderer/device work. The inputs remain immutable full modules.
BEGIN {
  split("ret br switch indirectbr invoke callbr resume catchswitch catchret cleanupret unreachable fneg add fadd sub fsub mul fmul udiv sdiv fdiv urem srem frem shl lshr ashr and or xor extractelement insertelement shufflevector extractvalue insertvalue alloca load store fence cmpxchg atomicrmw getelementptr trunc zext sext fptrunc fpext fptoui fptosi uitofp sitofp ptrtoint inttoptr bitcast addrspacecast icmp fcmp phi select freeze call va_arg landingpad catchpad cleanuppad", words);
  for (i in words) valid[words[i]] = 1;
  PROCINFO["sorted_in"] = "@ind_str_asc";
}
/^define / {
  active = 1; module_functions++;
  match($0, /@[^ (]+/, name); current = substr(name[0], 2);
  is_kernel = ($0 ~ /amdgpu_kernel/);
  if (is_kernel) { kernels[current] = 1; blocks[current] = 1; }
  next;
}
active && /^}/ { active = 0; next; }
active {
  line = $0;
  sub(/^[ \t]+/, "", line);
  if (line ~ /^[A-Za-z0-9$._-]+:/) {
    if (is_kernel && instructions[current] > 0) blocks[current]++;
    next;
  }
  sub(/^%[^=]+=[ \t]*/, "", line);
  sub(/^(musttail|notail|tail)[ \t]+/, "", line);
  split(line, fields, /[ \t]+/); op = fields[1];
  if (!(op in valid)) next;
  module_instructions++;
  if (!is_kernel) next;
  instructions[current]++;
  opcodes[current, op]++;
  if (line ~ /(^|[ \t])fast([ \t]|$)/) fast[current]++;
  if (op == "call" || op == "invoke" || op == "callbr") {
    if (match(line, /@[^ (]+/, called)) callee = substr(called[0], 2);
    else callee = "<indirect>";
    calls[current, callee]++;
  }
}
END {
  print "module", FILENAME, "functions=" module_functions, "instructions=" module_instructions;
  for (kernel in kernels) {
    print "kernel", kernel, "instructions=" instructions[kernel], "blocks=" blocks[kernel], "fast=" fast[kernel];
    for (key in opcodes) {
      split(key, parts, SUBSEP);
      if (parts[1] == kernel) print "opcode", parts[2], opcodes[key];
    }
    for (key in calls) {
      split(key, parts, SUBSEP);
      if (parts[1] == kernel) print "callee", parts[2], calls[key];
    }
  }
}
