# GPT-5.5 Takehome Hints

These hints are advisory. The live source and tests are authoritative.

## Objective

Lower `CYCLES` below `147734` without changing the benchmark, tests, machine
constants, or correctness criteria.

## Strong Guidance

- Prefer a conservative VLIW packing change over an algorithm rewrite.
- The current `KernelBuilder.build()` emits one instruction bundle per slot.
  That is the obvious cycle leak.
- Respect the machine model: reads observe old values, writes commit at the end
  of the cycle, and each engine has a slot limit.
- Pack only slots that are independent under that model.
- If dependency analysis is uncertain, leave the slots separate.

## Known Bad Directions

- Broad `build_kernel` replacement.
- First-pass vectorization with `vload`, `vstore`, or `valu` over data-dependent
  tree/index accesses.
- Unsupported invented operations.
- Stale `<search>` text copied from an earlier failed candidate.
- Cosmetic refactors or precompute-only patches that still report `147734`.

## Concrete First Probe

Try to make `build(slots, vliw=True)` emit packed bundles and call it from
`build_kernel`, while keeping the existing slot order and debug comparisons.
At minimum, pack the two independent ALU slots at the start of each
`HASH_STAGES` step, then keep the dependent ALU and debug compare after them.
